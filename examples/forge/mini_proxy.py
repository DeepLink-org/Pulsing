#!/usr/bin/env python3
"""mini_proxy v6: codex(Responses API) -> z.ai(Chat Completions)

v6: complete bidirectional MCP tool conversion.
- codex sends tools as {type:"namespace", name:"mcp__pulsing", tools:[...]}:
  flatten each nested function to chat tool "mcp__pulsing__pulsing_send"
- codex history input has function_call/function_call_output items WITH a
  separate namespace field: join to flat chat names, join outputs by call_id
- z.ai tool_calls back: split flat name into (namespace, name) so codex
  router (ToolName::new(namespace,name)) dispatches to the MCP server
"""
import json, os, sys, time, urllib.request, urllib.error, socketserver
from http.server import BaseHTTPRequestHandler

UPSTREAM = os.environ.get("MINI_UPSTREAM", "https://api.z.ai/api/paas/v4/chat/completions")


def _egress_default():
    """Internal proxy IP is stable; sandbox DNS for the hostname is lossy.
    Reuse the credentials the platform injected in env proxy vars, but pin
    the internal IP. Never hardcode credentials here."""
    if os.environ.get("MINI_PROXY"):
        return os.environ["MINI_PROXY"]
    for v in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        u = os.environ.get(v, "")
        if "@" in u:
            creds = u.split("//", 1)[1].split("@", 1)[0]
            return "http://%s@proxy.internal:8080" % creds
    return "http://proxy.internal:8080"


EGRESS_PROXY = _egress_default()
API_KEY = os.environ.get("BIGMODEL_API_KEY", "")
PORT = int(os.environ.get("PROXY_PORT", "4020"))
RETRIES = int(os.environ.get("MINI_RETRIES", "3"))

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({
    "http": EGRESS_PROXY, "https": EGRESS_PROXY,
}))

# flat chat name -> (namespace or None, tool name); filled by conv_tools
FLAT_MAP = {}


def call_upstream(payload):
    body = json.dumps(payload).encode()
    last = None
    # 429 = z.ai per-key concurrency limit: back off hard and retry
    backoffs = [3, 6, 12, 20, 30, 45, 60, 60]
    attempt = 0
    while True:
        req = urllib.request.Request(UPSTREAM, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY,
        })
        try:
            resp = _opener.open(req, timeout=120)
            return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode(errors="replace")
            if e.code == 429 and attempt < len(backoffs):
                w = backoffs[attempt]
                attempt += 1
                print("429 backoff %ds (try %d)" % (w, attempt), file=sys.stderr, flush=True)
                time.sleep(w)
                continue
            raise RuntimeError("upstream HTTP %d: %s" % (e.code, detail))
        except Exception as e:
            last = e
            if attempt < 3:
                attempt += 1
                print("RETRY %d after %s" % (attempt, e), file=sys.stderr, flush=True)
                time.sleep(1 + attempt * 2)
                continue
            raise RuntimeError("upstream failed: %s" % last)


def flat(ns, name):
    return (ns + "__" + name) if ns else name


def unflat(fname):
    """chat flat name -> (namespace or None, tool name)"""
    if fname in FLAT_MAP:
        return FLAT_MAP[fname]
    if "__" in fname:
        ns, _, t = fname.partition("__")
        return (ns, t)
    return (None, fname)


def conv_tools(tools):
    """Responses tool defs -> Chat nested {type:function, function:{...}}."""
    out = []

    def add_flat(ns, name, desc, params):
        key = flat(ns, name)
        FLAT_MAP[key] = (ns, name)
        out.append({"type": "function", "function": {
            "name": key, "description": desc, "parameters": params,
        }})

    for t in tools or []:
        tt = t.get("type", "?")
        if tt == "function":
            if "function" in t:  # already chat-shaped, passthrough
                out.append(t)
            else:
                add_flat(None, t.get("name", ""), t.get("description", ""), t.get("parameters", {}))
        elif tt == "namespace":
            ns = t.get("name", "")
            for nt in t.get("tools", []):
                if nt.get("type") == "function":
                    add_flat(ns, nt.get("name", ""), nt.get("description", ""), nt.get("parameters", {}))
                else:
                    print("DROP nested tool type=%s" % nt.get("type"), file=sys.stderr, flush=True)
        else:
            print("DROP tool type=%s name=%s" % (tt, t.get("name", "?")), file=sys.stderr, flush=True)
    return out


def content_to_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    return str(c)


def conv_input(body):
    """Responses input -> Chat messages, merging consecutive function_calls."""
    msgs = []
    if body.get("instructions"):
        msgs.append({"role": "system", "content": body["instructions"]})
    inp = body.get("input", [])
    if isinstance(inp, str):
        msgs.append({"role": "user", "content": inp})
        return msgs
    pending = []

    def flush():
        if pending:
            msgs.append({"role": "assistant", "content": "", "tool_calls": list(pending)})
            pending.clear()

    for it in inp:
        if not isinstance(it, dict):
            msgs.append({"role": "user", "content": str(it)})
            continue
        t = it.get("type", "")
        if t == "function_call":
            ns = it.get("namespace")
            pending.append({"id": it.get("call_id", it.get("id", "")),
                            "type": "function",
                            "function": {"name": flat(ns, it.get("name", "")),
                                         "arguments": it.get("arguments", "{}")}})
        else:
            flush()
            if t == "function_call_output":
                msgs.append({"role": "tool",
                             "tool_call_id": it.get("call_id", ""),
                             "content": content_to_text(it.get("output", ""))})
            elif t == "message" or "content" in it:
                role = it.get("role", "user")
                text = content_to_text(it.get("content", ""))
                if role == "assistant" and not text:
                    continue
                msgs.append({"role": role, "content": text})
    flush()
    return msgs


def J(obj):
    return json.dumps(obj, ensure_ascii=False)


def build_sse(body):
    msg = body["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    rid = body.get("id", "resp_proxy")
    model = body.get("model", "m")
    created = body.get("created", 0)
    events = [{"type": "response.created", "response": {
        "id": rid, "object": "response", "created_at": created,
        "status": "in_progress", "model": model, "output": []}}]
    oi = 0
    outputs = []
    if reasoning:
        item = {"type": "reasoning", "id": "r0", "summary": [{"type": "summary_text", "text": reasoning[:2000]}]}
        events.append({"type": "response.output_item.added", "output_index": oi, "item": item})
        events.append({"type": "response.output_item.done", "output_index": oi, "item": item})
        outputs.append(item)
        oi += 1
    for tc in tcs:
        fn = tc["function"]
        ns, tname = unflat(fn["name"])
        cid = tc.get("id", "c" + str(oi))
        call_item = {"type": "function_call", "id": cid, "call_id": cid,
                     "name": tname, "arguments": fn["arguments"], "status": "completed"}
        if ns:
            call_item["namespace"] = ns
        events.append({"type": "response.output_item.added", "output_index": oi, "item": call_item})
        events.append({"type": "response.output_item.done", "output_index": oi, "item": call_item})
        outputs.append(call_item)
        oi += 1
    if text or not tcs:
        msg_item = {"type": "message", "id": "m", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": text}]}
        events.append({"type": "response.output_item.added", "output_index": oi,
                       "item": {"type": "message", "id": "m", "role": "assistant",
                                "status": "in_progress", "content": []}})
        events.append({"type": "response.output_item.done", "output_index": oi, "item": msg_item})
        outputs.append(msg_item)
        oi += 1
    events.append({"type": "response.completed", "response": {
        "id": rid, "object": "response", "created_at": created,
        "status": "completed", "model": model, "output": outputs}})
    lines = ["data: " + J(e) for e in events]
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""
        print("REQ " + self.path + " " + str(n) + "B", file=sys.stderr, flush=True)
        try:
            body = json.loads(raw)
        except Exception:
            body = {}
        payload = {"model": body.get("model", "glm-5-turbo"),
                   "messages": conv_input(body)}
        tools = conv_tools(body.get("tools"))
        if tools:
            payload["tools"] = tools
        try:
            resp_body = call_upstream(payload)
            sse = build_sse(resp_body)
            ntc = len(resp_body["choices"][0]["message"].get("tool_calls") or [])
            print("RESP %dB tool_calls=%d" % (len(sse), ntc), file=sys.stderr, flush=True)
        except Exception as e:
            print("ERR " + str(e), file=sys.stderr, flush=True)
            sse = build_sse({"choices": [{"message": {"content": "proxy error: " + str(e)}}],
                             "id": "err", "model": payload["model"], "created": 0})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(sse)))
        self.end_headers()
        self.wfile.write(sse)

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", "3")
        self.end_headers()
        self.wfile.write(b"ok\n")

    def log_message(self, *a):
        pass


class TS(socketserver.ForkingTCPServer):
    """v8: fork per connection + bounded children.

    max_children caps concurrent upstream request streams, keeping us under
    the z.ai per-key concurrency limit (429 code 1302) instead of tripping it.
    Waiting codex clients queue at accept — waves of MINI_MAX_CONC streams.
    """
    allow_reuse_address = True
    max_children = int(os.environ.get("MINI_MAX_CONC", "8"))

    def handle_error(self, *a):
        pass


if __name__ == "__main__":
    print("mini_proxy v8(fork,cap=%d) :%d -> %s" % (TS.max_children, PORT, UPSTREAM), flush=True)
    TS(("127.0.0.1", PORT), H).serve_forever()
