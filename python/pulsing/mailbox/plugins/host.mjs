// Private bootstrap pipe, never a user-facing command. The Python wrapper owns
// networking and shutdown; closing this process's stdin releases that wrapper.
const { PULSING_MCP_URL: url, PULSING_MCP_TOKEN: token } = process.env;
if (!url || !token) throw new Error("Pulsing companion did not provide its endpoint");
process.stdout.write(JSON.stringify({ url, token }) + "\n");
process.stdin.resume();
process.stdin.on("end", () => process.exit(0));
