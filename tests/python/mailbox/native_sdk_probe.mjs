// Opt-in compatibility test against installed upstream SDKs, without inference.
import assert from "node:assert/strict";
import { mkdir, readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const sdk = async name => {
  const root = join(resolve(process.argv[2]), "node_modules", name);
  const manifest = JSON.parse(await readFile(join(root, "package.json"), "utf8"));
  return import(pathToFileURL(join(root, manifest.main)).href);
};
const plugins = new URL("../../../python/pulsing/mailbox/plugins/", import.meta.url);
const cwd = resolve(process.argv[3]);
await mkdir(cwd, { recursive: true });

if (process.argv[4] === "pi") {
  const {
    createAgentSessionRuntime, createAgentSessionServices, createAgentSessionFromServices,
    SessionManager, SettingsManager, ModelRuntime,
  } = await sdk("@earendil-works/pi-coding-agent");
  const runtime = await ModelRuntime.create({
    authPath: join(cwd, "auth.json"), modelsPath: null,
    modelsStorePath: join(cwd, "models.json"),
    allowModelNetwork: false, refreshOnCreate: false,
  });
  const errors = [];
  const factory = async input => {
    const services = await createAgentSessionServices({
      cwd: input.cwd, agentDir: cwd, modelRuntime: runtime,
      settingsManager: SettingsManager.inMemory(),
      resourceLoaderOptions: {
        noExtensions: true, noSkills: true, noThemes: true,
        noPromptTemplates: true, noContextFiles: true,
        additionalExtensionPaths: [new URL("pi.mjs", plugins).pathname],
      },
    });
    assert.equal(services.resourceLoader.getExtensions().errors.length, 0);
    const result = await createAgentSessionFromServices({
      services, sessionManager: input.sessionManager,
      sessionStartEvent: input.sessionStartEvent, noTools: "builtin",
    });
    return { ...result, services, diagnostics: services.diagnostics };
  };
  const host = await createAgentSessionRuntime(factory, {
    cwd, agentDir: cwd, sessionManager: SessionManager.inMemory(cwd),
  });
  const bind = session => session.bindExtensions({ mode: "rpc", onError: error => errors.push(error) });
  host.setRebindSession(bind);
  let disposed = false;
  try {
    await bind(host.session);
    assert.deepEqual(errors, []);
    assert.equal(host.session.getAllTools().filter(t => t.name.startsWith("mail_")).length, 9);
    assert.ok(host.session.getActiveToolNames().includes("mail_send"));
    const identity = async (session = host.session) => {
      const runner = session.extensionRunner;
      return (await runner.getToolDefinition("mail_identity").execute(
        "sdk-call", {}, undefined, undefined, runner.createToolContext("sdk-call"),
      )).details;
    };
    const first = await identity();
    await host.session.reload();
    assert.equal((await identity()).identity, first.identity);
    await host.newSession();
    assert.notEqual((await identity()).identity, first.identity);
    const second = await createAgentSessionRuntime(factory, {
      cwd, agentDir: cwd, sessionManager: SessionManager.inMemory(cwd),
    });
    try {
      await bind(second.session);
      const secondIdentity = await identity(second.session);
      assert.notEqual(secondIdentity.identity, (await identity()).identity);
      await host.dispose();
      disposed = true;
      assert.equal((await identity(second.session)).identity, secondIdentity.identity);
    } finally {
      await second.dispose();
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ harness: "pi", sdkLoaded: true, identity: first.identity, tools: 9, reload: true, newSession: true, concurrentSessions: true }));
  } finally {
    if (!disposed) await host.dispose();
  }
} else {
  const { Context } = await sdk("@deepseek-ai/cordis");
  const { default: SystemPrompt } = await sdk("@deepseek-ai/dsh-system-prompt");
  const { default: ToolsService } = await sdk("@deepseek-ai/dsh-tools");
  const plugin = await import(new URL("dsh.mjs", plugins));
  const ctx = new Context();
  const agents = [{ id: "dsh-sdk-parent" }, { id: "dsh-sdk-child" }];
  const agentService = ctx.provide("agents", { list: () => agents });
  const promptService = ctx.plugin(SystemPrompt, {});
  await promptService;
  const toolsService = ctx.plugin(ToolsService, {});
  await toolsService;
  const nativePlugin = ctx.plugin(plugin);
  try {
    await nativePlugin;
    assert.equal(ctx.tools.definitions().length, 9);
    const duplicate = ctx.plugin({ name: "pulsing-duplicate", inject: plugin.inject, apply: plugin.apply });
    await duplicate;
    await duplicate.dispose();
    assert.equal(ctx.tools.definitions().length, 9);
    const call = (name, args, agent) => ctx.tools.get(name).execute(
      args, { agent, signal: new AbortController().signal },
    );
    const parent = await call("mail_identity", {}, agents[0]);
    const child = await call("mail_identity", {}, agents[1]);
    assert.notEqual(parent.identity, child.identity);
    await assert.rejects(call("mail_identity", {}, undefined), /authoritative/);
    const { createScope } = await sdk("@deepseek-ai/dsh-scope");
    const policy = ctx.inject(["tools"], async policyContext => {
      const scope = createScope(policyContext, agents[1]);
      try {
        scope.ctx.tools.restrict({ deny: ["mail_send"] });
        assert.equal(policyContext.tools.get("mail_send", agents[1]), undefined);
        assert.ok(policyContext.tools.get("mail_send", agents[0]));
      } finally {
        await scope.dispose();
      }
    });
    await policy;
    await policy.dispose();
    console.log(JSON.stringify({ harness: "dsh", sdkLoaded: true, tools: 9, isolated: true }));
  } finally {
    await nativePlugin.dispose();
    assert.equal(ctx.tools.definitions().length, 0);
    await toolsService.dispose();
    await promptService.dispose();
    await agentService();
  }
}
