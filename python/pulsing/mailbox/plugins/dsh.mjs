import { dispatchOptions, protocol, resultValue, sharedBridge, threadIdentity } from "./bridge.mjs";

export const name = "pulsing";
export const inject = ["agents", "tools", "systemPrompt"];
const OWNER = Symbol.for("pulsing.dsh.tool.v1");

export async function apply(ctx) {
  if ((await dispatchOptions()).enabled) {
    throw new Error("Automatic Pulsing dispatch currently requires the Pi adapter");
  }
  const bridge = sharedBridge();
  const rules = await protocol();
  const tools = await bridge.start();
  // Installing a bundle and using the invocation overlay may load two copies.
  // Only the registration owner tears down the shared connection.
  if (ctx.tools.get("mail_identity")?.[OWNER]) return;
  const owner = Symbol("pulsing-dsh-plugin");
  bridge.retain(owner);
  ctx.effect(() => () => bridge.release(owner));
  for (const tool of tools) {
    ctx.tools.register({
      [OWNER]: true,
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema,
      output: {
        schema: {},
        render: (_args, value) => [{ type: "text", text: JSON.stringify(value) }],
      },
      async execute(args, execution) {
        // Read the actual caller, including spawned/forked subagents. Inherited
        // tool registrations must never capture the creator's session ID.
        const thread = threadIdentity("dsh", execution.agent?.id);
        return resultValue(await bridge.call(tool.name, args, thread, execution.signal));
      },
    });
  }
  ctx.systemPrompt.section({
    name: "pulsing-collaboration",
    order: 9000,
    interpolate: false,
    text: rules,
  });
  const bind = async agent => {
    await bridge.call("mail_identity", {}, threadIdentity("dsh", agent.id));
  };
  ctx.on("agent/created", async ({ agent }) => { await bind(agent); });
  for (const agent of ctx.agents.list()) await bind(agent);
}
