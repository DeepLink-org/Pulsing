// Deterministic host-API probe. No model SDK, credentials, or inference.
import { createInterface } from "node:readline";
import { EventEmitter } from "node:events";

const harness = process.argv[2];
const base = process.argv[3]
  ? new URL(`file://${process.argv[3]}/`)
  : new URL("../../../python/pulsing/mailbox/plugins/", import.meta.url);
const tools = new Map();
const handlers = new Map();
const disposers = [];
let currentSession = "root";
const context = () => ({
  sessionManager: { getSessionId: () => currentSession },
  ui: { setStatus() {}, notify() {} },
  isIdle: () => true,
  hasPendingMessages: () => false,
});

if (harness === "pi") {
  const { default: extension } = await import(new URL("pi.mjs", base));
  extension({
    events: new EventEmitter(),
    on: (name, callback) => handlers.set(name, callback),
    registerTool: tool => tools.set(tool.name, tool),
    registerCommand() {},
  });
  await handlers.get("session_start")({}, context());
  const prompt = handlers.get("before_agent_start")({ systemPrompt: "Original prompt" });
  if (!prompt.systemPrompt.startsWith("Original prompt\n") || !prompt.systemPrompt.includes("untrusted")) {
    throw new Error("Pi protocol was not injected");
  }
} else {
  const { apply } = await import(new URL("dsh.mjs", base));
  const sections = [];
  await apply({
    effect: setup => { disposers.push(setup()); },
    tools: {
      register: tool => tools.set(tool.name, tool),
      get: name => tools.get(name),
    },
    systemPrompt: { section: section => sections.push(section) },
    agents: { list: () => [{ id: currentSession }] },
    on: (name, callback) => handlers.set(name, callback),
  });
  if (!sections[0].text.includes("untrusted") || sections[0].interpolate !== false) {
    throw new Error("DSH protocol was not injected");
  }
}

async function call(name, args = {}, session = currentSession) {
  const signal = new AbortController().signal;
  const tool = tools.get(name);
  if (!tool) throw new Error(`Missing native tool: ${name}`);
  if (harness === "pi") {
    const ctx = { sessionManager: { getSessionId: () => session } };
    const result = await tool.execute("native-test", args, signal, undefined, ctx);
    return result.details;
  }
  return tool.execute(args, { agent: session === null ? undefined : { id: session }, signal });
}

process.stdout.write(JSON.stringify({ ready: await call("mail_identity"), tools: [...tools.keys()] }) + "\n");
for await (const line of createInterface({ input: process.stdin })) {
  const request = JSON.parse(line);
  try {
    if (request.action === "switch") {
      await handlers.get("session_shutdown")?.({ reason: "new" }, context());
      currentSession = request.session;
      await handlers.get("session_start")?.({ reason: "new" }, context());
    }
    const result = await call(request.tool, request.arguments, request.session);
    process.stdout.write(JSON.stringify({ result }) + "\n");
  } catch (error) {
    process.stdout.write(JSON.stringify({ error: error.message }) + "\n");
  }
}
await handlers.get("session_shutdown")?.({ reason: "quit" }, context());
for (const dispose of disposers.reverse()) await dispose();
