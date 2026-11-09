// The real Pi harness runs deterministic in-memory model events. No provider
// network, real API key, or paid inference is involved.
import assert from "node:assert/strict";
import { mkdir, readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { resultValue, sharedBridge } from "../../../python/pulsing/mailbox/plugins/bridge.mjs";

process.env.PULSING_DISPATCH = "1";
process.env.PULSING_DISPATCH_MAX_TURNS = "4";
const cwd = resolve(process.argv[3]);
await mkdir(cwd, { recursive: true });
async function sdk(name) {
  const root = join(resolve(process.argv[2]), "node_modules", name);
  const manifest = JSON.parse(await readFile(join(root, "package.json"), "utf8"));
  return import(pathToFileURL(join(root, manifest.main)).href);
}
const {
  createAgentSessionServices, createAgentSessionFromServices,
  ModelRuntime, SettingsManager, SessionManager,
} = await sdk("@earendil-works/pi-coding-agent");
const { createAssistantMessageEventStream } = await sdk("@earendil-works/pi-ai");
const runtime = await ModelRuntime.create({
  authPath: join(cwd, "auth.json"), modelsPath: null,
  modelsStorePath: join(cwd, "models.json"), refreshOnCreate: false,
  allowModelNetwork: false,
});
let phase = "read";
let hold = false;
let held;
let requests = 0;
runtime.registerProvider("pulsing-fixture", {
  api: "openai-completions",
  baseUrl: "http://127.0.0.1:1/never-used",
  apiKey: "deterministic-test-only",
  streamSimple(model, context, options) {
    requests++;
    const stream = createAssistantMessageEventStream();
    const reply = {
      role: "assistant", api: model.api, provider: model.provider, model: model.id,
      timestamp: Date.now(), stopReason: "stop", content: [],
      usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
    };
    queueMicrotask(() => {
      stream.push({ type: "start", partial: reply });
      if (hold) {
        held?.();
        const abort = () => {
          const error = { ...reply, stopReason: "aborted", errorMessage: "Fixture cancelled" };
          stream.push({ type: "error", reason: "aborted", error }); stream.end(error);
        };
        if (options.signal.aborted) abort();
        else options.signal.addEventListener("abort", abort, { once: true });
        return;
      }
      const text = message => typeof message.content === "string" ? message.content
        : message.content.filter(c => c.type === "text").map(c => c.text).join("");
      const notification = [...context.messages].reverse().map(text)
        .find(value => value.includes("Pulsing delivery notification."));
      const id = notification.match(/[0-9a-f]{32}/)[0];
      let tool;
      let args;
      if (phase === "read") { tool = "mail_read"; args = { message_id: id }; phase = "claim"; }
      else if (phase === "claim") { tool = "mail_claim"; args = { message_id: id }; phase = "finish"; }
      else if (phase === "finish") {
        const result = JSON.parse(text(context.messages.at(-1)));
        tool = "mail_finish"; args = { message_id: id, claim_token: result.claim_token, result: "Fixture checked" };
        phase = "done";
      } else { phase = "read"; }
      reply.content = tool
        ? [{ type: "toolCall", id: `fixture-${requests}`, name: tool, arguments: args }]
        : [{ type: "text", text: "Mail processed within the approved fixture scope." }];
      reply.stopReason = tool ? "toolUse" : "stop";
      stream.push({ type: "done", reason: reply.stopReason, message: reply }); stream.end(reply);
    });
    return stream;
  },
  models: [{
    id: "dispatch", name: "Deterministic dispatch fixture", reasoning: false,
    input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 32768, maxTokens: 512,
  }],
});
const services = await createAgentSessionServices({
  cwd, agentDir: cwd, modelRuntime: runtime, settingsManager: SettingsManager.inMemory(),
  resourceLoaderOptions: {
    noExtensions: true, noSkills: true, noThemes: true, noContextFiles: true,
    noPromptTemplates: true,
    additionalExtensionPaths: [new URL("../../../python/pulsing/mailbox/plugins/pi.mjs", import.meta.url).pathname],
  },
});
const { session } = await createAgentSessionFromServices({
  services, sessionManager: SessionManager.inMemory(cwd), noTools: "builtin",
  model: runtime.getModel("pulsing-fixture", "dispatch"),
});
const errors = [];
function settled() {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { off(); reject(new Error("Pi did not settle")); }, 10_000);
    const off = session.subscribe(event => {
      if (event.type === "agent_settled") { off(); clearTimeout(timeout); resolve(event); }
    });
  });
}
try {
  await session.bindExtensions({ mode: "rpc", onError: error => errors.push(error) });
  const bridge = sharedBridge();
  const thread = `pi:${session.sessionManager.getSessionId()}`;
  const recipient = resultValue(await bridge.call("mail_identity", {}, thread));
  const sender = "pi:fixture-operator";
  const send = key => bridge.call("mail_send", {
    to: recipient.identity, subject: "Fixture task", body: "Read, claim and finish this fixture mail.",
    idempotency_key: key,
  }, sender).then(resultValue);
  const finished = settled();
  const first = await send("first");
  await finished;
  assert.equal(resultValue(await bridge.call("mail_status", {
    to: recipient.identity, message_id: first.id,
  }, sender)).state, "completed");
  assert.equal(requests, 4);
  // Cancel a genuine native Pi run. Its unread mail must not immediately wake
  // another run; only the explicit operator command resumes delivery.
  hold = true;
  const started = new Promise(resolve => { held = resolve; });
  const second = await send("second");
  await started;
  const aborted = settled();
  await session.abort();
  await aborted;
  assert.equal(bridge.dispatchStates.get(thread).paused, "cancelled");
  const callsAfterAbort = requests;
  await send("second"); // An uncertain-send retry is not a fresh delivery.
  assert.equal(requests, callsAfterAbort);
  hold = false;
  const resumed = settled();
  await session.prompt("/pulsing resume");
  await resumed;
  assert.equal(resultValue(await bridge.call("mail_status", {
    to: recipient.identity, message_id: second.id,
  }, sender)).state, "completed");
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ realPiHarness: true, completed: 2, cancellationPaused: true, operatorResume: true, externalModelRequests: 0 }));
} finally {
  await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
  session.dispose();
}
