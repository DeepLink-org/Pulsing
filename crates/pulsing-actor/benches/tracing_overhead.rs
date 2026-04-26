//! Benchmark: memtable write overhead for each tracing store.
//!
//! Measures the **per-operation cost** of writing to the four memtable-backed
//! stores, so we know how much overhead tracing adds to actor messaging.
//!
//! Run: `cargo bench -p pulsing-actor --bench tracing_overhead`

use std::hint::black_box;
use std::time::Instant;

use probing_memtable::discover::ExposedHashTable;
use probing_memtable::discover::ExposedTable;
use probing_memtable::{DType, Schema, Value};

// ── Helpers ───────────────────────────────────────────────────────────

fn bench_loop(name: &str, iterations: u64, mut f: impl FnMut(u64)) {
    // Warmup
    for i in 0..iterations.min(1000) {
        f(i);
    }

    let start = Instant::now();
    for i in 0..iterations {
        f(i);
    }
    let elapsed = start.elapsed();

    let per_op = elapsed / iterations as u32;
    let ops_sec = if elapsed.as_secs_f64() > 0.0 {
        iterations as f64 / elapsed.as_secs_f64()
    } else {
        f64::INFINITY
    };

    println!("  {name:<40} {elapsed:>10.3?}  {per_op:>8.0?}/op  {ops_sec:>12.0} ops/s",);
}

// ── Span store (MEMT ring buffer, wide row) ──────────────────────────

fn bench_span_write() {
    let mut schema = Schema::new()
        .col("trace_id", DType::Str)
        .col("span_id", DType::Str)
        .col("parent_span_id", DType::Str)
        .col("name", DType::Str)
        .col("kind", DType::Str)
        .col("start_us", DType::I64)
        .col("end_us", DType::I64)
        .col("duration_us", DType::I64)
        .col("status_code", DType::Str)
        .col("instrumentation_scope", DType::Str);
    for name in &[
        "attr_actor_name",
        "attr_message_type",
        "attr_target_addr",
        "attr_target_path",
        "attr_pulsing_op",
        "attr_http_method",
        "attr_http_route",
        "attr_http_url",
        "attr_http_peer",
    ] {
        schema = schema.col(name, DType::Str);
    }
    schema = schema
        .col("events_json", DType::Str)
        .col("links_json", DType::Str);

    let mut table = ExposedTable::create("__bench_spans", &schema, 65536, 16).unwrap();

    let n = 100_000u64;
    bench_loop("span_write (MEMT, 21-col row)", n, |i| {
        let tid = format!("trace-{i:08x}");
        let sid = format!("span-{i:08x}");
        let row: Vec<Value> = vec![
            Value::Str(&tid),
            Value::Str(&sid),
            Value::Str(""),
            Value::Str("actor.ask"),
            Value::Str("INTERNAL"),
            Value::I64(1_700_000_000_000_000 + i as i64),
            Value::I64(1_700_000_000_001_000 + i as i64),
            Value::I64(1000),
            Value::Str("OK"),
            Value::Str("pulsing"),
            Value::Str("actors/counter_0"),
            Value::Str("ask"),
            Value::Str(""),
            Value::Str(""),
            Value::Str("ask"),
            Value::Str(""),
            Value::Str(""),
            Value::Str(""),
            Value::Str(""),
            Value::Str("[]"),
            Value::Str("[]"),
        ];
        let mut w = table.writer();
        w.push_row(black_box(&row));
    });
}

// ── Actor store (MEMH hash table) ────────────────────────────────────

fn bench_actor_upsert() {
    let mut table = ExposedHashTable::create("__bench_actors", 256, 65536, 0).unwrap();

    let names: Vec<String> = (0..256).map(|i| format!("actors/worker_{i}")).collect();
    let n = 100_000u64;

    bench_loop("actor_upsert (MEMH, 256 actors)", n, |i| {
        let key = &names[(i % 256) as usize];
        let val = format!("{i}|1|Worker|mymod");
        let mut w = table.writer();
        let _ = w.insert(black_box(key), black_box(&Value::Str(&val)));
    });
}

fn bench_actor_remove() {
    let mut table = ExposedHashTable::create("__bench_actors_rm", 256, 65536, 0).unwrap();

    let names: Vec<String> = (0..256).map(|i| format!("actors/worker_{i}")).collect();

    // Pre-populate
    {
        let mut w = table.writer();
        for name in &names {
            let _ = w.insert(name, &Value::Str("0|1|Worker|mymod"));
        }
    }

    let n = 100_000u64;
    bench_loop("actor_remove (MEMH, 256 actors)", n, |i| {
        let key = &names[(i % 256) as usize];
        let mut w = table.writer();
        w.remove(black_box(key));
        // Re-insert so next remove has something to do
        let _ = w.insert(key, &Value::Str("0|1|Worker|mymod"));
    });
}

// ── Metrics store (MEMT ring buffer) ─────────────────────────────────

fn bench_metrics_write() {
    let schema = Schema::new()
        .col("timestamp_us", DType::I64)
        .col("node_id", DType::Str)
        .col("actors_count", DType::I64)
        .col("messages_total", DType::I64)
        .col("actors_created", DType::I64)
        .col("actors_stopped", DType::I64)
        .col("uptime_secs", DType::I64);

    let mut table = ExposedTable::create("__bench_metrics", &schema, 65536, 16).unwrap();

    let n = 100_000u64;
    bench_loop("metrics_write (MEMT, 7-col row)", n, |i| {
        let row: Vec<Value> = vec![
            Value::I64(1_700_000_000_000_000 + i as i64),
            Value::Str("node-1"),
            Value::I64(42),
            Value::I64(i as i64),
            Value::I64(100),
            Value::I64(5),
            Value::I64(3600),
        ];
        let mut w = table.writer();
        w.push_row(black_box(&row));
    });
}

// ── Members store (MEMH hash table) ──────────────────────────────────

fn bench_member_upsert() {
    let mut table = ExposedHashTable::create("__bench_members", 64, 16384, 0).unwrap();

    let keys: Vec<String> = (0..32).map(|i| format!("{i}")).collect();
    let n = 100_000u64;

    bench_loop("member_upsert (MEMH, 32 nodes)", n, |i| {
        let key = &keys[(i % 32) as usize];
        let val = format!("10.0.0.{}:9000|online|{}", i % 32, i);
        let mut w = table.writer();
        let _ = w.insert(black_box(key), black_box(&Value::Str(&val)));
    });
}

// ── Span write with u64 IDs (no format!) ─────────────────────────────

fn bench_span_write_u64_ids() {
    let mut schema = Schema::new()
        .col("trace_id_hi", DType::U64)
        .col("trace_id_lo", DType::U64)
        .col("span_id", DType::U64)
        .col("parent_span_id", DType::U64)
        .col("name", DType::Str)
        .col("kind", DType::Str)
        .col("start_us", DType::I64)
        .col("end_us", DType::I64)
        .col("duration_us", DType::I64)
        .col("status_code", DType::Str)
        .col("instrumentation_scope", DType::Str);
    for name in &[
        "attr_actor_name",
        "attr_message_type",
        "attr_target_addr",
        "attr_target_path",
        "attr_pulsing_op",
        "attr_http_method",
        "attr_http_route",
        "attr_http_url",
        "attr_http_peer",
    ] {
        schema = schema.col(name, DType::Str);
    }
    schema = schema
        .col("events_json", DType::Str)
        .col("links_json", DType::Str);

    let mut table = ExposedTable::create("__bench_spans_u64", &schema, 65536, 16).unwrap();

    let n = 100_000u64;
    bench_loop("span_write (MEMT, u64 IDs)", n, |i| {
        let row: Vec<Value> = vec![
            Value::U64(0xdeadbeef00000000 + i), // trace_id_hi
            Value::U64(i),                      // trace_id_lo
            Value::U64(i * 7),                  // span_id
            Value::U64(0),                      // parent_span_id
            Value::Str("actor.ask"),
            Value::Str("INTERNAL"),
            Value::I64(1_700_000_000_000_000 + i as i64),
            Value::I64(1_700_000_000_001_000 + i as i64),
            Value::I64(1000),
            Value::Str("OK"),
            Value::Str("pulsing"),
            Value::Str("actors/counter_0"),
            Value::Str("ask"),
            Value::Str(""),
            Value::Str(""),
            Value::Str("ask"),
            Value::Str(""),
            Value::Str(""),
            Value::Str(""),
            Value::Str(""),
            Value::Str("[]"),
            Value::Str("[]"),
        ];
        let mut w = table.writer();
        w.push_row(black_box(&row));
    });
}

// ── Isolate: format! cost alone ──────────────────────────────────────

fn bench_format_overhead() {
    let n = 100_000u64;
    bench_loop("format!(trace_id + span_id) only", n, |i| {
        let _tid = black_box(format!("trace-{i:08x}"));
        let _sid = black_box(format!("span-{i:08x}"));
    });
}

// ── Isolate: raw MEMT push_row (no alloc) ────────────────────────────

fn bench_memt_raw_push() {
    let schema = Schema::new()
        .col("a", DType::I64)
        .col("b", DType::I64)
        .col("c", DType::Str);

    let mut table = ExposedTable::create("__bench_raw_push", &schema, 65536, 16).unwrap();
    let n = 100_000u64;
    bench_loop("memt_push (new writer each time)", n, |i| {
        let row = [Value::I64(i as i64), Value::I64(42), Value::Str("hello")];
        let mut w = table.writer();
        w.push_row(black_box(&row));
    });
}

fn bench_memt_direct_push() {
    let schema = Schema::new()
        .col("a", DType::I64)
        .col("b", DType::I64)
        .col("c", DType::Str);

    let mut table = ExposedTable::create("__bench_direct_push", &schema, 65536, 16).unwrap();
    let n = 100_000u64;
    bench_loop("memt_push (ExposedTable::push_row)", n, |i| {
        let row = [Value::I64(i as i64), Value::I64(42), Value::Str("hello")];
        table.push_row(black_box(&row));
    });
}

// ── Baseline: mutex acquire/release ──────────────────────────────────

fn bench_mutex_baseline() {
    let m = std::sync::Mutex::new(0u64);
    let n = 1_000_000u64;
    bench_loop("mutex lock/unlock (baseline)", n, |_| {
        let mut g = m.lock().unwrap();
        *g = black_box(*g + 1);
    });
}

// ── Main ─────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("Pulsing tracing overhead benchmark");
    println!("===================================");
    println!();
    println!(
        "  {:<40} {:>10}  {:>10}  {:>12}",
        "test", "total", "latency", "throughput"
    );
    println!("  {}", "-".repeat(76));

    bench_mutex_baseline();
    bench_format_overhead();
    bench_memt_raw_push();
    bench_memt_direct_push();
    println!();
    bench_span_write();
    bench_span_write_u64_ids();
    bench_metrics_write();
    println!();
    bench_actor_upsert();
    bench_actor_remove();
    bench_member_upsert();

    println!();
}
