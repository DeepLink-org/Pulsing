// Shared, dependency-free MCP transport for native Pi and DSH plugins.
// Mailbox semantics and schemas belong to Python; do not duplicate them here.
import { spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const TOOL_NAMES = new Set([
  "mail_identity", "mail_roles", "mail_send", "mail_list", "mail_read",
  "mail_claim", "mail_finish", "mail_reply", "mail_status",
]);
const SHARED = Symbol.for("pulsing.mailbox.bridge.v1");
const INTERNAL_TOOLS = new Set(["mail_wait", "mail_dispatch_event"]);

export async function runtimeConfig() {
  try {
    return JSON.parse(await readFile(new URL("./runtime.json", import.meta.url), "utf8"));
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    return {};
  }
}

export async function dispatchOptions() {
  const config = await runtimeConfig();
  const enabled = process.env.PULSING_DISPATCH === undefined
    ? config.dispatch === true : process.env.PULSING_DISPATCH === "1";
  const maxTurns = Number(process.env.PULSING_DISPATCH_MAX_TURNS ?? config.dispatchMaxTurns ?? 8);
  if (!Number.isInteger(maxTurns) || maxTurns < 1 || maxTurns > 64) {
    throw new Error("Pulsing dispatch max turns must be 1–64");
  }
  return { enabled, maxTurns };
}

export function threadIdentity(harness, id) {
  if (typeof id !== "string" || !id.trim()) {
    throw new Error(`Pulsing requires an authoritative ${harness} session ID`);
  }
  const identity = `${harness}:${id}`;
  if (Buffer.byteLength(identity, "utf8") > 256) {
    throw new Error("Pulsing session ID exceeds 256 UTF-8 bytes");
  }
  return identity;
}

export async function protocol() {
  if (process.env.PULSING_PROTOCOL) {
    return readFile(process.env.PULSING_PROTOCOL, "utf8");
  }
  try {
    return await readFile(new URL("./PROTOCOL.md", import.meta.url), "utf8");
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    return readFile(new URL("../COMPANION.md", import.meta.url), "utf8");
  }
}

export function resultValue(result) {
  if (result.isError) {
    throw new Error(result.content?.find(item => item.type === "text")?.text || "Pulsing tool failed");
  }
  const text = result.content?.find(item => item.type === "text")?.text;
  if (typeof text !== "string") throw new Error("Pulsing returned no JSON tool result");
  return JSON.parse(text);
}

export class MailboxBridge {
  constructor() {
    this.sequence = 0;
    this.closed = false;
    this.pending = new Set();
    this.owners = new Set();
    this.dispatchStates = new Map();
  }

  async start() {
    if (this.closed) throw new Error("Pulsing plugin is closed");
    if (!this.starting) {
      this.starting = this.connect().catch(async error => {
        await this.stopChild();
        this.starting = undefined;
        throw error;
      });
    }
    return this.starting;
  }

  async connect() {
    const url = process.env.PULSING_MCP_URL;
    const token = process.env.PULSING_MCP_TOKEN;
    if (url || token) {
      if (!url || !token) throw new Error("Both PULSING_MCP_URL and PULSING_MCP_TOKEN are required");
      this.endpoint = { url, token };
    } else {
      this.endpoint = await this.bootstrap();
    }
    const parsed = new URL(this.endpoint.url);
    if (parsed.protocol !== "http:" || !["127.0.0.1", "localhost", "[::1]"].includes(parsed.hostname)
      || parsed.username || parsed.password) {
      throw new Error("Pulsing plugins only connect to an authenticated loopback HTTP companion");
    }
    const initialized = await this.post("initialize", {
      protocolVersion: "2025-03-26",
      capabilities: {},
      clientInfo: { name: "pulsing-native-plugin", version: "0.1.0" },
    });
    this.version = initialized.protocolVersion;
    await this.post("notifications/initialized", {}, undefined, true);
    const result = await this.post("tools/list", {});
    this.capabilities = new Set(result.tools.map(tool => tool.name));
    this.definitions = result.tools.filter(tool => TOOL_NAMES.has(tool.name));
    if (this.definitions.length !== TOOL_NAMES.size) {
      throw new Error("Pulsing companion does not expose the expected mailbox tools");
    }
    return this.definitions;
  }

  async bootstrap() {
    const config = await runtimeConfig();
    if (this.closed) throw new Error("Pulsing plugin is closed");
    const child = spawn(process.env.PULSING_PYTHON || config.python || "python3", [
      "-m", "pulsing.cli", "--adapter", "generic", ...(config.launchArgs || []),
      "--", process.execPath, fileURLToPath(new URL("./host.mjs", import.meta.url)),
    ], {
      // A terminal Ctrl-C may abort a Pi/DSH turn without closing the session.
      // Do not deliver that foreground-group signal to its owned companion.
      detached: true,
      // This helper is infrastructure, not an inference adapter. The owning
      // Pi extension reads the operator's dispatch option in its own process.
      env: { ...process.env, PULSING_DISPATCH: "0" },
      stdio: ["pipe", "pipe", "inherit"],
    });
    this.child = child;
    // No credentials are written to a file or a log. Read the readiness frame
    // over an anonymous pipe; stdout is never inherited by the coding harness.
    return new Promise((resolve, reject) => {
      let bytes = "";
      const timer = setTimeout(() => fail(new Error("Pulsing companion startup timed out")), 45_000);
      const fail = error => { clearTimeout(timer); reject(error); };
      child.once("error", () => fail(new Error("Cannot start Pulsing; install pulsing[mailbox] and configure its Python executable")));
      child.once("exit", () => fail(new Error("Pulsing companion exited")));
      child.stdout.on("data", chunk => {
        bytes += chunk.toString("utf8");
        if (bytes.length > 16384) return fail(new Error("Invalid Pulsing readiness frame"));
        const newline = bytes.indexOf("\n");
        if (newline < 0) return;
        clearTimeout(timer);
        try {
          const frame = JSON.parse(bytes.slice(0, newline));
          if (typeof frame.url !== "string" || typeof frame.token !== "string" || !frame.token) {
            throw new Error("Invalid Pulsing readiness frame");
          }
          resolve(frame);
        } catch {
          reject(new Error("Invalid Pulsing readiness frame"));
        }
      });
    });
  }

  async post(method, params, signal, notification = false) {
    if (this.closed) throw new Error("Pulsing plugin is closed");
    const controller = new AbortController();
    this.pending.add(controller);
    const timeout = AbortSignal.timeout(30_000);
    try {
      const body = { jsonrpc: "2.0", method, params };
      if (!notification) body.id = ++this.sequence;
      const response = await fetch(this.endpoint.url, {
        method: "POST",
        redirect: "error",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json, text/event-stream",
          Authorization: `Bearer ${this.endpoint.token}`,
          ...(this.version ? { "MCP-Protocol-Version": this.version } : {}),
        },
        body: JSON.stringify(body),
        signal: AbortSignal.any([controller.signal, timeout, ...(signal ? [signal] : [])]),
      });
      if (!response.ok) throw new Error(`Pulsing MCP HTTP ${response.status}`);
      if (notification) {
        await response.body?.cancel();
        return;
      }
      const result = await response.json();
      if (result.id !== body.id || result.jsonrpc !== "2.0") {
        throw new Error("Invalid Pulsing MCP response");
      }
      if (result.error) throw new Error(result.error.message || "Pulsing MCP request failed");
      return result.result;
    } finally {
      this.pending.delete(controller);
    }
  }

  async call(name, args, thread, signal) {
    if (!TOOL_NAMES.has(name) && !INTERNAL_TOOLS.has(name)) throw new Error("Not a Pulsing mailbox tool");
    // Validate before connecting: never silently fall back to a parent mailbox.
    if (typeof thread !== "string" || !thread.trim() || Buffer.byteLength(thread, "utf8") > 256) {
      throw new Error("Pulsing requires an authoritative session ID");
    }
    signal?.throwIfAborted();
    await this.start();
    return this.post("tools/call", {
      name, arguments: args, _meta: { threadId: thread },
    }, signal);
  }

  async stopChild() {
    const child = this.child;
    this.child = undefined;
    if (!child || child.exitCode !== null || child.signalCode !== null) return;
    await new Promise(resolve => {
      const kill = setTimeout(() => child.kill("SIGTERM"), 5000);
      const hardKill = setTimeout(() => child.kill("SIGKILL"), 12000);
      child.once("exit", () => {
        clearTimeout(kill);
        clearTimeout(hardKill);
        resolve();
      });
      child.stdin.end();
    });
  }

  async close() {
    this.closed = true;
    for (const controller of this.pending) controller.abort();
    await this.stopChild();
  }

  retain(owner) {
    this.owners.add(owner);
  }

  async release(owner, closeWhenUnused = true) {
    this.owners.delete(owner);
    if (closeWhenUnused && this.owners.size === 0) await this.close();
  }
}

export function sharedBridge() {
  // Pi replaces extension instances on /new, /fork and /reload. The companion
  // belongs to the OS process, not to the replaceable extension instance.
  let bridge = globalThis[SHARED];
  if (!bridge || bridge.closed) {
    bridge = new MailboxBridge();
    globalThis[SHARED] = bridge;
    process.once("exit", () => bridge.child?.kill("SIGTERM"));
  }
  return bridge;
}
