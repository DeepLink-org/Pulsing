//! System metrics time-series backed by a MEMT ring buffer (mmap'd file).
//!
//! Each `GetMetrics` call appends one row to `<tmpdir>/probing/<pid>/pulsing.metrics`.
//! Older rows are overwritten when the ring buffer wraps.

use probing_memtable::discover::ExposedTable;
use probing_memtable::Value;
use std::sync::{Mutex, Once, OnceLock};

static METRICS_MEMTABLE: OnceLock<Mutex<ExposedTable>> = OnceLock::new();
static INIT_METRICS_MEMTABLE: Once = Once::new();

pub(crate) fn init_metrics_memtable() {
    use probing_memtable::{DType, Schema};

    INIT_METRICS_MEMTABLE.call_once(|| {
        let schema = Schema::new()
            .col("timestamp_us", DType::I64)
            .col("node_id", DType::Str)
            .col("actors_count", DType::I64)
            .col("messages_total", DType::I64)
            .col("actors_created", DType::I64)
            .col("actors_stopped", DType::I64)
            .col("uptime_secs", DType::I64);

        match ExposedTable::create("pulsing.metrics", &schema, 65536, 16) {
            Ok(table) => {
                let _ = METRICS_MEMTABLE.set(Mutex::new(table));
            }
            Err(e) => {
                eprintln!("pulsing: failed to create metrics memtable: {e}");
            }
        }
    });
}

pub(crate) fn write_metrics_snapshot(
    ts_unix_micros: u64,
    node_id: &str,
    actors_count: u64,
    messages_total: u64,
    actors_created: u64,
    actors_stopped: u64,
    uptime_secs: u64,
) {
    let Some(mt) = METRICS_MEMTABLE.get() else {
        return;
    };
    let Ok(mut table) = mt.lock() else {
        return;
    };

    let row: Vec<Value> = vec![
        Value::I64(ts_unix_micros as i64),
        Value::Str(node_id),
        Value::I64(actors_count as i64),
        Value::I64(messages_total as i64),
        Value::I64(actors_created as i64),
        Value::I64(actors_stopped as i64),
        Value::I64(uptime_secs as i64),
    ];

    table.push_row(&row);
}
