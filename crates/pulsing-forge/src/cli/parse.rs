//! Line parsing — tool args (Nushell-style flags + JSON).

use anyhow::Result;
use serde_json::{Map, Value, json};

pub fn parse_tool_args(text: &str) -> Result<Value> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(json!({}));
    }
    if text.starts_with('{') {
        return Ok(serde_json::from_str(text)?);
    }
    let mut out = Map::new();
    let mut parts = text.split_whitespace().peekable();
    while parts.peek().is_some() {
        let tok = parts.next().unwrap();
        if !tok.starts_with("--") {
            continue;
        }
        let key = tok.trim_start_matches("--");
        if let Some((k, v)) = key.split_once('=') {
            out.insert(k.to_string(), Value::String(v.to_string()));
            continue;
        }
        if let Some(v) = parts.next() {
            if !v.starts_with("--") {
                out.insert(key.to_string(), Value::String(v.to_string()));
            }
        } else {
            out.insert(key.to_string(), Value::Bool(true));
        }
    }
    Ok(Value::Object(out))
}

pub fn parse_tool_invocation(rest: &str) -> Result<(String, Value)> {
    let (tool, args_text) = rest
        .split_once(|c: char| c.is_whitespace())
        .map(|(a, b)| (a.to_string(), b.trim()))
        .unwrap_or((rest.to_string(), ""));
    Ok((tool, parse_tool_args(args_text)?))
}
