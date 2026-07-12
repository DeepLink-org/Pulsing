//! MCP tool metadata, filtering, and model-visible name normalization.

use std::collections::HashSet;

use rmcp::model::Tool;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha1::{Digest, Sha1};
use tracing::warn;

use super::LEGACY_MCP_TOOL_NAME_PREFIX;
use super::config::McpServerConfig;

const MCP_UI_META_KEY: &str = "ui";
const MCP_UI_VISIBILITY_META_KEY: &str = "visibility";
const MCP_UI_MODEL_VISIBILITY: &str = "model";
const MCP_TOOL_NAME_DELIMITER: &str = "__";
const MAX_TOOL_NAME_BYTES: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    pub server_name: String,
    #[serde(default)]
    pub supports_parallel_tool_calls: bool,
    #[serde(default)]
    pub server_origin: Option<String>,
    #[serde(rename = "tool_name", alias = "callable_name")]
    pub callable_name: String,
    #[serde(rename = "tool_namespace", alias = "callable_namespace")]
    pub callable_namespace: String,
    #[serde(default, alias = "connector_description")]
    pub namespace_description: Option<String>,
    pub tool: Tool,
    pub connector_id: Option<String>,
    pub connector_name: Option<String>,
    #[serde(default)]
    pub plugin_display_names: Vec<String>,
}

impl ToolInfo {
    pub fn model_tool_name(&self, prefix_mcp_tool_names: bool) -> String {
        if prefix_mcp_tool_names {
            format!(
                "{}{}{}",
                self.callable_namespace, MCP_TOOL_NAME_DELIMITER, self.callable_name
            )
        } else {
            format!("{}/{}", self.callable_namespace, self.callable_name)
        }
    }
}

pub fn tool_is_model_visible(tool: &ToolInfo) -> bool {
    let Some(visibility) = tool
        .tool
        .meta
        .as_deref()
        .and_then(|meta| meta.get(MCP_UI_META_KEY))
        .and_then(Value::as_object)
        .and_then(|ui| ui.get(MCP_UI_VISIBILITY_META_KEY))
        .and_then(Value::as_array)
    else {
        return true;
    };
    visibility
        .iter()
        .any(|t| t.as_str() == Some(MCP_UI_MODEL_VISIBILITY))
}

#[derive(Default, Clone)]
pub struct ToolFilter {
    enabled: Option<HashSet<String>>,
    disabled: HashSet<String>,
}

impl ToolFilter {
    pub fn from_config(cfg: &McpServerConfig) -> Self {
        Self {
            enabled: cfg
                .enabled_tools
                .as_ref()
                .map(|t| t.iter().cloned().collect()),
            disabled: cfg
                .disabled_tools
                .as_ref()
                .map(|t| t.iter().cloned().collect())
                .unwrap_or_default(),
        }
    }

    pub fn allows(&self, tool_name: &str) -> bool {
        if let Some(enabled) = &self.enabled {
            if !enabled.contains(tool_name) {
                return false;
            }
        }
        !self.disabled.contains(tool_name)
    }
}

pub fn filter_tools(tools: Vec<ToolInfo>, filter: &ToolFilter) -> Vec<ToolInfo> {
    tools
        .into_iter()
        .filter(|t| filter.allows(&t.tool.name))
        .collect()
}

pub fn sanitize_responses_api_tool_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() { "tool".into() } else { out }
}

pub fn normalize_tools_for_model<I>(tools: I, prefix_mcp_tool_names: bool) -> Vec<ToolInfo>
where
    I: IntoIterator<Item = ToolInfo>,
{
    let mut seen_identity = HashSet::new();
    let mut seen_model_names = HashSet::new();
    let mut out = Vec::new();
    for mut tool in tools {
        let identity = format!("{}\0{}", tool.server_name, tool.tool.name);
        if !seen_identity.insert(identity) {
            warn!(
                server = %tool.server_name,
                tool = %tool.tool.name,
                "skipping duplicated MCP tool from same server"
            );
            continue;
        }
        tool.callable_namespace = if prefix_mcp_tool_names {
            format!(
                "{}{}",
                LEGACY_MCP_TOOL_NAME_PREFIX,
                sanitize_responses_api_tool_name(&tool.server_name)
            )
        } else {
            sanitize_responses_api_tool_name(&tool.server_name)
        };
        tool.callable_name = sanitize_responses_api_tool_name(&tool.tool.name);
        tool.callable_namespace = truncate_to_bytes(&tool.callable_namespace, MAX_TOOL_NAME_BYTES);
        tool.callable_name = truncate_to_bytes(&tool.callable_name, MAX_TOOL_NAME_BYTES);
        let model_name = tool.model_tool_name(prefix_mcp_tool_names);
        if !seen_model_names.insert(model_name) {
            warn!(
                server = %tool.server_name,
                tool = %tool.tool.name,
                model_name = %tool.model_tool_name(prefix_mcp_tool_names),
                "skipping MCP tool with colliding model-visible name"
            );
            continue;
        }
        out.push(tool);
    }
    out
}

fn truncate_to_bytes(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let hash = format!("{:x}", Sha1::digest(s.as_bytes()));
    let suffix = &hash[..8.min(hash.len())];
    let keep = max.saturating_sub(suffix.len() + 1);
    format!(
        "{}_{}",
        &s[..s.floor_char_boundary(keep.min(s.len()))],
        suffix
    )
}

pub fn tool_spec_for_model(tool: &ToolInfo, prefix_mcp_tool_names: bool) -> Value {
    serde_json::json!({
        "name": tool.model_tool_name(prefix_mcp_tool_names),
        "description": tool
            .tool
            .description
            .as_ref()
            .map(|d| d.to_string())
            .unwrap_or_default(),
        "input_schema": tool_input_schema_json(&tool.tool),
        "server_name": tool.server_name,
        "tool_name": tool.tool.name,
    })
}

pub fn tool_input_schema_json(tool: &Tool) -> Value {
    let mut schema = if tool.input_schema.as_ref().is_empty() {
        Map::from_iter([("type".into(), Value::String("object".into()))])
    } else {
        tool.input_schema.as_ref().clone()
    };
    // OpenAI models require `properties`; some MCP servers omit or null it (Codex parity).
    if schema.get("properties").is_none_or(|v| v.is_null()) {
        schema.insert("properties".into(), Value::Object(Map::new()));
    }
    Value::Object(schema)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::Tool;
    use std::sync::Arc;

    fn sample_tool(name: &str) -> Tool {
        Tool::new_with_raw(name.to_string(), None, Arc::new(serde_json::Map::new()))
    }

    #[test]
    fn tool_input_schema_inserts_empty_properties() {
        let tool = sample_tool("no_props");
        let schema = tool_input_schema_json(&tool);
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"].is_object());
    }

    #[test]
    fn normalize_skips_colliding_model_names() {
        let mk = |server: &str| ToolInfo {
            server_name: server.into(),
            supports_parallel_tool_calls: false,
            server_origin: None,
            callable_name: "search".into(),
            callable_namespace: server.into(),
            namespace_description: None,
            tool: sample_tool("search"),
            connector_id: None,
            connector_name: None,
            plugin_display_names: vec![],
        };
        // Distinct servers, same sanitized namespace → one model-visible name.
        let out = normalize_tools_for_model(vec![mk("my.server"), mk("my_server")], true);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].model_tool_name(true), "mcp__my_server__search");
    }

    #[test]
    fn tool_spec_for_model_includes_properties() {
        let mut tool = sample_tool("search");
        tool.description = Some("Search issues".into());
        let info = ToolInfo {
            server_name: "github".into(),
            supports_parallel_tool_calls: false,
            server_origin: None,
            callable_name: "search".into(),
            callable_namespace: "github".into(),
            namespace_description: None,
            tool,
            connector_id: None,
            connector_name: None,
            plugin_display_names: vec![],
        };
        let out = normalize_tools_for_model(vec![info], true);
        let spec = tool_spec_for_model(&out[0], true);
        assert_eq!(spec["name"], "mcp__github__search");
        assert_eq!(spec["description"], "Search issues");
        assert!(spec["input_schema"]["properties"].is_object());
    }

    #[test]
    fn normalize_adds_mcp_prefix() {
        let info = ToolInfo {
            server_name: "github".into(),
            supports_parallel_tool_calls: false,
            server_origin: None,
            callable_name: "search".into(),
            callable_namespace: "github".into(),
            namespace_description: None,
            tool: sample_tool("search"),
            connector_id: None,
            connector_name: None,
            plugin_display_names: vec![],
        };
        let out = normalize_tools_for_model(vec![info], true);
        assert_eq!(out[0].callable_namespace, "mcp__github");
        assert_eq!(out[0].model_tool_name(true), "mcp__github__search");
    }
}
