//! JSONL trace — same format as ``python/pulsing/forge/repl/trace.py``.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecord {
    pub seq: u64,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session: Option<Value>,
}

pub fn load_trace(path: &Path) -> Result<Vec<TraceRecord>> {
    let file = File::open(path).with_context(|| format!("open trace {}", path.display()))?;
    let mut out = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        out.push(serde_json::from_str(line)?);
    }
    Ok(out)
}

pub fn save_trace(path: &Path, records: &[TraceRecord]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut file = File::create(path)?;
    for rec in records {
        writeln!(file, "{}", serde_json::to_string(rec)?)?;
    }
    Ok(())
}

pub fn tool_calls(records: &[TraceRecord]) -> Vec<&TraceRecord> {
    records.iter().filter(|r| r.kind == "tool_call").collect()
}
