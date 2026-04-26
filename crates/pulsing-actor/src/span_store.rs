//! Span storage backed by a memtable (mmap'd ring buffer).
//!
//! When tracing is initialized, an [`ExposedTable`] is created at
//! `<tmpdir>/probing/<pid>/pulsing.spans`. Any tool that understands the memtable
//! format (e.g. probing's DataFusion engine) can read spans in real time via mmap.

use futures::future::BoxFuture;
use opentelemetry::trace::{SpanId, Status};
use opentelemetry::KeyValue;
use opentelemetry_sdk::export::trace::{ExportResult, SpanData, SpanExporter};
use probing_memtable::discover::ExposedTable;
use probing_memtable::Value;
use std::sync::{Mutex, Once, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const MAX_ATTR_ENTRIES: usize = 128;
const MAX_ATTR_VALUE_BYTES: usize = 1024;
const MAX_EVENTS_SERIALIZED: usize = 32;
const MAX_EVENT_ATTR_ENTRIES: usize = 16;
const MAX_LINKS_SERIALIZED: usize = 32;

const SPAN_ATTR_KEYS: &[(&str, &str)] = &[
    ("actor.name", "attr_actor_name"),
    ("message.type", "attr_message_type"),
    ("target.addr", "attr_target_addr"),
    ("target.path", "attr_target_path"),
    ("pulsing.op", "attr_pulsing_op"),
    ("http.method", "attr_http_method"),
    ("http.route", "attr_http_route"),
    ("http.url", "attr_http_url"),
    ("http.peer", "attr_http_peer"),
];

// ── memtable storage ───────────────────────────────────────────────────

static SPAN_MEMTABLE: OnceLock<Mutex<ExposedTable>> = OnceLock::new();
static INIT_SPAN_MEMTABLE: Once = Once::new();

pub(crate) fn init_span_memtable() {
    use probing_memtable::{DType, Schema};

    INIT_SPAN_MEMTABLE.call_once(|| {
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

        for &(_, col_name) in SPAN_ATTR_KEYS {
            schema = schema.col(col_name, DType::Str);
        }
        schema = schema
            .col("events_json", DType::Str)
            .col("links_json", DType::Str);

        match ExposedTable::create("pulsing.spans", &schema, 65536, 16) {
            Ok(table) => {
                let _ = SPAN_MEMTABLE.set(Mutex::new(table));
            }
            Err(e) => {
                eprintln!("pulsing: failed to create span memtable: {e}");
            }
        }
    });
}

fn write_span_to_memtable(record: &SpanRecord) {
    let Some(mt) = SPAN_MEMTABLE.get() else {
        return;
    };
    let Ok(mut table) = mt.lock() else {
        return;
    };

    let attrs: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&record.attributes_json).unwrap_or_default();

    let mut attr_vals: Vec<String> = Vec::with_capacity(SPAN_ATTR_KEYS.len());
    for &(src_key, _) in SPAN_ATTR_KEYS {
        let v = attrs.get(src_key).and_then(|v| v.as_str()).unwrap_or("");
        attr_vals.push(v.to_string());
    }

    let mut row: Vec<Value> = Vec::with_capacity(10 + SPAN_ATTR_KEYS.len() + 2);
    row.push(Value::Str(&record.trace_id));
    row.push(Value::Str(&record.span_id));
    row.push(Value::Str(&record.parent_span_id));
    row.push(Value::Str(&record.name));
    row.push(Value::Str(&record.kind));
    row.push(Value::I64((record.start_unix_nanos / 1000) as i64));
    row.push(Value::I64((record.end_unix_nanos / 1000) as i64));
    row.push(Value::I64(record.duration_nanos as i64 / 1000));
    row.push(Value::Str(&record.status_code));
    row.push(Value::Str(&record.instrumentation_scope));
    for v in &attr_vals {
        row.push(Value::Str(v));
    }
    row.push(Value::Str(&record.events_json));
    row.push(Value::Str(&record.links_json));

    table.push_row(&row);
}

// ── exporter ───────────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct InMemorySpanExporter;

impl SpanExporter for InMemorySpanExporter {
    fn export(&mut self, batch: Vec<SpanData>) -> BoxFuture<'static, ExportResult> {
        for sd in batch {
            write_span_to_memtable(&span_data_to_record(sd));
        }
        Box::pin(std::future::ready(Ok(())))
    }

    fn shutdown(&mut self) {}
}

// ── internal SpanRecord (used by span_data_to_record / write_span_to_memtable) ──

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct SpanRecord {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: String,
    pub sampled: bool,
    pub name: String,
    pub kind: String,
    pub start_unix_nanos: u128,
    pub end_unix_nanos: u128,
    pub duration_nanos: u64,
    pub status_code: String,
    pub status_message: String,
    pub instrumentation_scope: String,
    pub instrumentation_version: String,
    pub dropped_attributes_count: u32,
    pub events_dropped_count: u32,
    pub links_dropped_count: u32,
    pub attributes_json: String,
    pub events_json: String,
    pub links_json: String,
}

// ── helpers ─────────────────────────────────────────────────────────────

fn system_time_unix_nanos(st: SystemTime) -> u128 {
    st.duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

fn truncate_bytes(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max.saturating_sub(1);
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

fn kv_slice_to_json_map(attrs: &[KeyValue], max_entries: usize, max_val_bytes: usize) -> String {
    let mut m = serde_json::Map::new();
    for kv in attrs.iter().take(max_entries) {
        let k = kv.key.to_string();
        let v = truncate_bytes(&kv.value.to_string(), max_val_bytes);
        m.insert(k, serde_json::Value::String(v));
    }
    serde_json::Value::Object(m).to_string()
}

fn span_data_to_record(sd: SpanData) -> SpanRecord {
    let sc = sd.span_context;
    let trace_id = sc.trace_id().to_string();
    let span_id = sc.span_id().to_string();
    let parent = sd.parent_span_id;
    let parent_span_id = if parent == SpanId::INVALID {
        String::new()
    } else {
        parent.to_string()
    };

    let start_n = system_time_unix_nanos(sd.start_time);
    let end_n = system_time_unix_nanos(sd.end_time);
    let duration_nanos = sd
        .end_time
        .duration_since(sd.start_time)
        .map(|d: Duration| d.as_nanos().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0);

    let (status_code, status_message) = match &sd.status {
        Status::Unset => ("unset".to_string(), String::new()),
        Status::Ok => ("ok".to_string(), String::new()),
        Status::Error { description } => (
            "error".to_string(),
            truncate_bytes(description.as_ref(), 4096),
        ),
    };

    let scope = &sd.instrumentation_scope;
    let instrumentation_version = scope.version().map(|s| s.to_string()).unwrap_or_default();

    let kind = format!("{:?}", sd.span_kind);
    let name = sd.name.into_owned();

    let attributes_json =
        kv_slice_to_json_map(&sd.attributes, MAX_ATTR_ENTRIES, MAX_ATTR_VALUE_BYTES);

    let events: Vec<serde_json::Value> = sd
        .events
        .iter()
        .take(MAX_EVENTS_SERIALIZED)
        .map(|ev| {
            let ts = system_time_unix_nanos(ev.timestamp);
            let attrs =
                kv_slice_to_json_map(&ev.attributes, MAX_EVENT_ATTR_ENTRIES, MAX_ATTR_VALUE_BYTES);
            serde_json::json!({
                "name": ev.name.as_ref(),
                "timestamp_unix_nanos": ts.to_string(),
                "attributes": serde_json::from_str::<serde_json::Value>(&attrs).unwrap_or_default(),
            })
        })
        .collect();
    let events_json = serde_json::to_string(&events).unwrap_or_else(|_| "[]".to_string());

    let links: Vec<serde_json::Value> = sd
        .links
        .iter()
        .take(MAX_LINKS_SERIALIZED)
        .map(|lk| {
            let lsc = lk.span_context.clone();
            serde_json::json!({
                "trace_id": lsc.trace_id().to_string(),
                "span_id": lsc.span_id().to_string(),
            })
        })
        .collect();
    let links_json = serde_json::to_string(&links).unwrap_or_else(|_| "[]".to_string());

    SpanRecord {
        trace_id,
        span_id,
        parent_span_id,
        sampled: sc.is_sampled(),
        name,
        kind,
        start_unix_nanos: start_n,
        end_unix_nanos: end_n,
        duration_nanos,
        status_code,
        status_message,
        instrumentation_scope: scope.name().to_string(),
        instrumentation_version,
        dropped_attributes_count: sd.dropped_attributes_count,
        events_dropped_count: sd.events.dropped_count,
        links_dropped_count: sd.links.dropped_count,
        attributes_json,
        events_json,
        links_json,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::trace::{SpanContext, SpanKind, TraceFlags, TraceId, TraceState};
    use opentelemetry::InstrumentationScope;
    use opentelemetry_sdk::trace::{SpanEvents, SpanLinks};
    use std::borrow::Cow;

    fn sample_span_data(name: &str) -> SpanData {
        let trace_id = TraceId::from_bytes([9u8; 16]);
        let span_id = SpanId::from_bytes([8u8; 8]);
        let sc = SpanContext::new(
            trace_id,
            span_id,
            TraceFlags::SAMPLED,
            true,
            TraceState::default(),
        );
        SpanData {
            span_context: sc,
            parent_span_id: SpanId::INVALID,
            span_kind: SpanKind::Internal,
            name: Cow::Owned(name.to_string()),
            start_time: UNIX_EPOCH + Duration::from_secs(1),
            end_time: UNIX_EPOCH + Duration::from_secs(2),
            attributes: vec![KeyValue::new("actor.name", "test_actor")],
            dropped_attributes_count: 0,
            events: SpanEvents::default(),
            links: SpanLinks::default(),
            status: Status::Ok,
            instrumentation_scope: InstrumentationScope::builder("test.scope").build(),
        }
    }

    #[tokio::test]
    async fn exporter_writes_to_memtable() {
        init_span_memtable();
        let mut exp = InMemorySpanExporter;
        let _ = exp.export(vec![sample_span_data("x")]).await;

        let mt = SPAN_MEMTABLE.get().unwrap().lock().unwrap();
        let view = mt.view();
        assert!(view.num_rows(view.write_chunk()) >= 1);
    }
}
