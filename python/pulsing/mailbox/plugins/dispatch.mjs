// A volatile, bounded message-to-harness scheduler. No model loop or recovery.
import { randomUUID } from "node:crypto";
import { resultValue } from "./bridge.mjs";

export function dispatchState() {
  return { epoch: null, after: 0, pending: new Map(), turns: 0, paused: null };
}

export class MessageDispatcher {
  constructor({ bridge, thread, isIdle, dispatch, state = dispatchState(),
    maxTurns = 8, batchSize = 20, maxPending = 128,
    onError = () => {}, onChange = () => {} }) {
    if (!Number.isInteger(maxTurns) || maxTurns < 1 || maxTurns > 64
      || !Number.isInteger(batchSize) || batchSize < 1 || batchSize > 20
      || !Number.isInteger(maxPending) || maxPending < batchSize) {
      throw new Error("Invalid Pulsing dispatcher limits");
    }
    Object.assign(this, { bridge, thread, isIdle, dispatch, state, maxTurns,
      batchSize, maxPending, onError, onChange });
    this.waiters = new Set();
    this.closed = false;
    this.owner = Symbol("dispatcher");
  }

  status() {
    return {
      state: this.closed ? "stopped" : this.state.paused ? "paused"
        : this.active ? "running" : this.state.pending.size ? "queued" : "waiting",
      reason: this.state.paused, pending: this.state.pending.size,
      turns: this.state.turns, maxTurns: this.maxTurns,
      dispatchId: this.active?.id ?? null,
    };
  }

  changed() { this.onChange(this.status()); }

  async record(phase, batch = this.active, reason = "") {
    return this.bridge.call("mail_dispatch_event", {
      phase, dispatch_id: batch?.id ?? "", message_ids: batch?.ids ?? [], reason,
    }, this.thread);
  }

  report(phase, batch, reason) {
    // Observations cannot recursively trigger work or retries on transport failure.
    void this.record(phase, batch, reason).catch(() => {});
  }

  start() {
    if (this.closed) throw new Error("Dispatcher is stopped");
    if (this.state.owner && this.state.owner !== this.owner) {
      throw new Error("This Pi session already has an active Pulsing dispatcher");
    }
    this.state.owner = this.owner;
    if (this.state.paused) { this.changed(); return; }
    if (this.controller && !this.controller.signal.aborted) return;
    this.controller = new AbortController();
    const signal = this.controller.signal;
    this.watching = this.watch(signal).catch(error => {
      if (!signal.aborted && !this.closed) {
        this.pause("transport");
        this.onError(error);
      }
    });
    this.pump();
    this.changed();
  }

  pause(reason = "operator") {
    this.state.paused = reason;
    this.controller?.abort();
    this.releaseCapacity();
    this.report("paused", this.active, reason);
    this.changed();
  }

  resume() {
    if (this.closed) throw new Error("Dispatcher is stopped");
    if (this.state.owner && this.state.owner !== this.owner) {
      throw new Error("This Pi session already has an active Pulsing dispatcher");
    }
    if (this.active || !this.isIdle()) throw new Error("Wait for the Pi run to settle before resuming");
    // Only an explicit operator resume retries unseen/read work. Processing and
    // terminal messages remain excluded by the mailbox snapshot.
    this.state.paused = null;
    this.state.turns = 0;
    this.state.epoch = null;
    this.state.after = 0;
    this.state.pending.clear();
    this.report("resumed", undefined, "operator");
    this.start();
  }

  releaseCapacity() {
    for (const wake of this.waiters) wake();
    this.waiters.clear();
  }

  async capacity(signal) {
    if (signal.aborted) return;
    await new Promise(resolve => {
      const wake = () => {
        this.waiters.delete(wake);
        signal.removeEventListener("abort", wake);
        resolve();
      };
      this.waiters.add(wake);
      signal.addEventListener("abort", wake, { once: true });
    });
  }

  async watch(signal) {
    while (!this.closed && !signal.aborted && !this.state.paused) {
      const space = this.maxPending - this.state.pending.size - (this.active?.ids.length ?? 0);
      if (space <= 0) { await this.capacity(signal); continue; }
      const response = resultValue(await this.bridge.call("mail_wait", {
        epoch: this.state.epoch, after: this.state.after,
        limit: Math.min(space, this.batchSize), timeout_seconds: 20,
      }, this.thread, signal));
      if (signal.aborted || this.closed) return;
      if (typeof response.epoch !== "string" || !Number.isInteger(response.cursor)
        || response.cursor < 0 || !Array.isArray(response.items)
        || response.items.length > Math.min(space, this.batchSize)) {
        throw new Error("Invalid Pulsing delivery cursor");
      }
      if (this.state.epoch && response.epoch !== this.state.epoch) {
        throw new Error("Mailbox lifetime changed; explicitly resume after checking lost work");
      }
      if (this.state.epoch && response.cursor < this.state.after) {
        throw new Error("Pulsing delivery cursor moved backwards");
      }
      this.state.epoch = response.epoch;
      this.state.after = response.cursor;
      for (const item of response.items) {
        if (typeof item.id !== "string" || !/^[0-9a-f]{32}$/.test(item.id)) {
          throw new Error("Invalid Pulsing message ID");
        }
        this.state.pending.set(item.id, true);
      }
      this.pump();
      this.changed();
    }
  }

  pump() {
    if (this.closed || this.state.paused || this.active
      || !this.state.pending.size || !this.isIdle()) return;
    if (this.state.turns >= this.maxTurns) { this.pause("budget"); return; }
    const ids = [...this.state.pending.keys()].slice(0, this.batchSize);
    const batch = { id: randomUUID(), ids, started: false };
    this.active = batch;
    this.state.turns++;
    for (const id of ids) this.state.pending.delete(id);
    this.changed();
    this.submitting = this.submit(batch);
  }

  restore(batch) {
    if (this.active !== batch) return;
    this.state.pending = new Map([
      ...batch.ids.map(id => [id, true]), ...this.state.pending,
    ]);
    this.active = undefined;
    this.releaseCapacity();
    this.changed();
  }

  async submit(batch) {
    try {
      await this.record("requested", batch, "delivery");
      if (this.closed || this.state.paused || this.active !== batch) {
        this.restore(batch);
        return;
      }
      // A human may have started a run during the audit round trip.
      if (!this.isIdle()) { this.restore(batch); return; }
      await this.dispatch(batch);
      this.report("submitted", batch, "delivery");
    } catch (error) {
      this.restore(batch);
      if (!this.closed) {
        this.pause("harness");
        this.onError(error);
      }
    }
  }

  started(dispatchId) {
    if (this.active?.id !== dispatchId) return;
    this.active.started = true;
    this.report("started", this.active, "delivery");
    this.changed();
  }

  settled({ aborted = false, failed = false } = {}) {
    if (aborted || failed) {
      this.active = undefined;
      this.releaseCapacity();
      this.pause(aborted ? "cancelled" : "harness");
      return;
    }
    if (this.active && !this.active.started) return;
    if (this.active) this.report("settled", this.active);
    this.active = undefined;
    this.releaseCapacity();
    this.pump();
    this.changed();
  }

  async stop() {
    this.closed = true;
    this.controller?.abort();
    this.releaseCapacity();
    try {
      await this.watching;
      await this.submitting;
    } finally {
      this.report("stopped", undefined, "session");
      if (this.state.owner === this.owner) delete this.state.owner;
      this.changed();
    }
  }
}
