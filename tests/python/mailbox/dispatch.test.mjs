import assert from "node:assert/strict";
import { test } from "node:test";
import { MessageDispatcher, dispatchState } from "../../../python/pulsing/mailbox/plugins/dispatch.mjs";

const tick = () => new Promise(resolve => setImmediate(resolve));
const id = n => n.toString(16).padStart(32, "0");
const result = data => ({ content: [{ type: "text", text: JSON.stringify(data) }] });

class Mail {
  constructor() { this.rows = []; this.epoch = "one"; this.waiters = new Set(); this.events = []; }
  push(n) { this.rows.push({ id: id(n) }); for (const wake of [...this.waiters]) wake(); }
  async call(tool, args, _thread, signal) {
    if (tool === "mail_dispatch_event") { this.events.push(args); return result({ recorded: true }); }
    assert.equal(tool, "mail_wait");
    if (args.epoch === this.epoch && args.after === this.rows.length) {
      await new Promise((resolve, reject) => {
        const cleanup = () => { this.waiters.delete(wake); signal.removeEventListener("abort", abort); };
        const wake = () => { cleanup(); resolve(); };
        const abort = () => { cleanup(); reject(new DOMException("Aborted", "AbortError")); };
        this.waiters.add(wake);
        signal.addEventListener("abort", abort, { once: true });
        if (signal.aborted) abort();
      });
    }
    const after = args.epoch === this.epoch ? args.after : 0;
    let cursor = after;
    const items = [];
    while (cursor < this.rows.length && items.length < args.limit) {
      const row = this.rows[cursor++];
      if (!row.completed) items.push({ id: row.id, state: "unread" });
    }
    return result({ epoch: this.epoch, cursor, items, more: cursor < this.rows.length });
  }
}

function fixture(options = {}) {
  const mail = new Mail();
  const batches = [];
  const errors = [];
  let idle = false;
  const driver = new MessageDispatcher({
    bridge: mail, thread: "pi:session", isIdle: () => idle,
    dispatch: batch => batches.push(batch), onError: error => errors.push(error),
    ...options,
  });
  return { mail, batches, errors, driver, idle: value => { idle = value; } };
}

test("busy bursts coalesce, idle dispatches once, and unread mail does not loop", async () => {
  const f = fixture();
  try {
    f.driver.start();
    await tick();
    f.mail.push(1); f.mail.push(2); f.mail.push(3);
    await tick();
    assert.equal(f.batches.length, 0);
    f.idle(true); f.driver.settled();
    await tick();
    assert.deepEqual(f.batches[0].ids, [id(1), id(2), id(3)]);
    f.driver.started(f.batches[0].id);
    f.driver.settled();
    await tick();
    assert.equal(f.batches.length, 1);
  } finally { await f.driver.stop(); }
});

test("pending capacity bounds prefetch without losing later deliveries", async () => {
  const f = fixture({ batchSize: 2, maxPending: 4 });
  try {
    f.driver.start();
    for (let n = 1; n <= 5; n++) f.mail.push(n);
    await tick();
    assert.equal(f.driver.state.pending.size, 4);
    f.idle(true);
    for (let i = 0; i < 3; i++) {
      f.driver.settled(); await tick();
      assert.ok(f.driver.state.pending.size + f.driver.active.ids.length <= 4);
      f.driver.started(f.batches[i].id);
    }
    assert.deepEqual(f.batches.flatMap(b => b.ids), [1, 2, 3, 4, 5].map(id));
  } finally { await f.driver.stop(); }
});

test("cancel pauses until an explicit resume, and resume only retries unfinished mail", async () => {
  const f = fixture();
  try {
    f.idle(true); f.driver.start(); f.mail.push(1); await tick();
    f.driver.started(f.batches[0].id);
    f.driver.settled({ aborted: true });
    f.mail.push(2); await tick();
    assert.equal(f.driver.status().reason, "cancelled");
    assert.equal(f.batches.length, 1);
    f.mail.rows[0].completed = true;
    f.driver.resume(); await tick();
    assert.deepEqual(f.batches[1].ids, [id(2)]);
  } finally { await f.driver.stop(); }
});

test("a hard automatic-turn budget stops reply storms", async () => {
  const f = fixture({ maxTurns: 1, batchSize: 1 });
  try {
    f.idle(true); f.driver.start(); f.mail.push(1); f.mail.push(2); await tick();
    f.driver.started(f.batches[0].id); f.driver.settled(); await tick();
    assert.equal(f.batches.length, 1);
    assert.equal(f.driver.status().reason, "budget");
    assert.equal(f.driver.state.pending.size, 1);
  } finally { await f.driver.stop(); }
});

test("dispatch failure restores pending work without an automatic retry", async () => {
  let attempts = 0;
  const f = fixture({ dispatch: () => { attempts++; throw new Error("harness unavailable"); } });
  try {
    f.idle(true); f.driver.start(); f.mail.push(1); await tick();
    assert.equal(f.driver.status().reason, "harness");
    assert.equal(attempts, 1);
    assert.equal(f.driver.state.pending.size, 1);
    f.mail.push(2); await tick();
    assert.equal(attempts, 1);
  } finally { await f.driver.stop(); }
});

test("reload retains cursor and budget, and mailbox restart fails closed", async () => {
  const state = dispatchState();
  const f = fixture({ state });
  f.idle(true); f.driver.start(); f.mail.push(1); await tick();
  f.driver.started(f.batches[0].id); f.driver.settled(); await f.driver.stop();
  const next = new MessageDispatcher({
    bridge: f.mail, thread: "pi:session", state,
    isIdle: () => true, dispatch: batch => f.batches.push(batch),
    onError: error => f.errors.push(error),
  });
  try {
    next.start(); await tick();
    assert.equal(f.batches.length, 1);
    assert.equal(next.status().turns, 1);
    f.mail.epoch = "two";
    for (const wake of [...f.mail.waiters]) wake();
    await tick();
    assert.equal(next.status().reason, "transport");
    assert.equal(f.batches.length, 1);
  } finally { await next.stop(); }
});

test("pause during the audit round trip prevents a late model submission", async () => {
  const f = fixture();
  let accept;
  const call = f.mail.call.bind(f.mail);
  f.mail.call = (tool, args, ...rest) => args.phase === "requested"
    ? new Promise(resolve => { accept = resolve; }) : call(tool, args, ...rest);
  f.idle(true); f.driver.start(); f.mail.push(1); await tick();
  f.driver.pause("operator");
  accept(result({ recorded: true }));
  await tick();
  assert.equal(f.batches.length, 0);
  assert.equal(f.driver.state.pending.size, 1);
  await f.driver.stop();
  assert.equal(f.mail.waiters.size, 0);
});

test("a second runtime cannot reset an active session's cursor or budget", async () => {
  const state = dispatchState();
  const f = fixture({ state });
  f.driver.start(); f.mail.push(1); await tick();
  state.turns = 3;
  const other = new MessageDispatcher({
    bridge: f.mail, thread: "pi:session", state,
    isIdle: () => true, dispatch: () => assert.fail("duplicate dispatcher"),
  });
  assert.throws(() => other.resume(), /already has/);
  assert.equal(state.after, 1);
  assert.equal(state.turns, 3);
  assert.equal(state.pending.size, 1);
  await f.driver.stop();
});
