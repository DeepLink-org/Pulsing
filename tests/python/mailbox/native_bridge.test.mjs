import assert from "node:assert/strict";
import { test } from "node:test";
import { MailboxBridge, resultValue, threadIdentity } from "../../../python/pulsing/mailbox/plugins/bridge.mjs";

test("harness IDs cannot fall back to a parent or collide across harnesses", () => {
  assert.notEqual(threadIdentity("pi", "one"), threadIdentity("dsh", "one"));
  for (const value of [undefined, null, "", " ", 123]) {
    assert.throws(() => threadIdentity("pi", value), /authoritative/);
  }
  assert.throws(() => threadIdentity("pi", "x".repeat(256)), /256/);
});

test("tool errors stay errors, including results without structuredContent", () => {
  assert.deepEqual(resultValue({ content: [{ type: "text", text: "[1,2]" }] }), [1, 2]);
  assert.throws(() => resultValue({ isError: true, content: [{ type: "text", text: "denied" }] }), /denied/);
  assert.throws(() => resultValue({ content: [] }), /no JSON/);
});

test("native context supplies metadata, never model arguments", async () => {
  const bridge = new MailboxBridge();
  bridge.start = async () => [];
  const calls = [];
  bridge.post = async (...args) => { calls.push(args); return {}; };
  await bridge.call("mail_identity", { threadId: "model-spoof" }, threadIdentity("pi", "native"));
  assert.equal(calls[0][1]._meta.threadId, "pi:native");
  await assert.rejects(bridge.call("mail_identity", {}, null), /authoritative/);
  const abort = new AbortController();
  abort.abort();
  await assert.rejects(bridge.call("mail_send", {}, "pi:native", abort.signal), { name: "AbortError" });
  assert.equal(calls.length, 1);
  await assert.rejects(bridge.call("shell", {}, "pi:native"), /Not a Pulsing/);
});

test("transport forwards cancellation and does not retry uncertain writes", async () => {
  const original = globalThis.fetch;
  const bridge = new MailboxBridge();
  bridge.endpoint = { url: "http://127.0.0.1:1/mcp", token: "test-only" };
  let calls = 0;
  globalThis.fetch = async (_url, options) => {
    calls++;
    assert.equal(options.redirect, "error");
    return new Promise((_resolve, reject) => {
      options.signal.addEventListener("abort", () => reject(options.signal.reason), { once: true });
    });
  };
  try {
    const abort = new AbortController();
    const pending = bridge.post("tools/call", { name: "mail_send" }, abort.signal);
    abort.abort();
    await assert.rejects(pending, { name: "AbortError" });
    assert.equal(calls, 1);
    assert.equal(bridge.pending.size, 0);
  } finally {
    globalThis.fetch = original;
    await bridge.close();
  }
});

test("borrowed endpoints fail closed on missing credentials or non-loopback URLs", async () => {
  const old = { url: process.env.PULSING_MCP_URL, token: process.env.PULSING_MCP_TOKEN };
  try {
    process.env.PULSING_MCP_URL = "https://example.com/mcp";
    process.env.PULSING_MCP_TOKEN = "not-sent";
    await assert.rejects(new MailboxBridge().start(), /loopback/);
    delete process.env.PULSING_MCP_TOKEN;
    await assert.rejects(new MailboxBridge().start(), /Both/);
  } finally {
    for (const [key, value] of [
      ["PULSING_MCP_URL", old.url], ["PULSING_MCP_TOKEN", old.token],
    ]) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
});
