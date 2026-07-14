//! Default tool schemas for the Forge agent loop (Anthropic wire format).

use serde_json::{Value, json};

pub const DEFAULT_TOOL_NAMES: &[&str] = &["update_plan", "Glob", "Read", "Grep", "shell_command"];

/// Tools for LLM-guided ``pulsing init`` (includes write/edit).
pub const INIT_TOOL_NAMES: &[&str] = &[
    "update_plan",
    "Glob",
    "Read",
    "Grep",
    "Write",
    "Edit",
    "shell_command",
];

pub fn forge_tool_definitions(names: &[&str]) -> Vec<Value> {
    names.iter().filter_map(|name| tool_schema(name)).collect()
}

fn tool_schema(name: &str) -> Option<Value> {
    let (description, schema) = match name {
        "update_plan" => (
            "Update the multi-step plan visible to the user.",
            json!({
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": { "type": "string" },
                                "status": { "type": "string" }
                            },
                            "required": ["step"]
                        }
                    }
                },
                "required": ["plan"]
            }),
        ),
        "Glob" => (
            "Find files by glob pattern.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string" },
                    "path": { "type": "string" }
                },
                "required": ["pattern"]
            }),
        ),
        "Read" => (
            "Read a file from the workspace.",
            json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string" },
                    "offset": { "type": "integer" },
                    "limit": { "type": "integer" }
                },
                "required": ["file_path"]
            }),
        ),
        "Grep" => (
            "Search file contents with ripgrep.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string" },
                    "path": { "type": "string" },
                    "glob": { "type": "string" }
                },
                "required": ["pattern"]
            }),
        ),
        "shell_command" => (
            "Run a shell command in the workspace.",
            json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "workdir": { "type": "string" },
                    "timeout_ms": { "type": "integer" }
                },
                "required": ["command"]
            }),
        ),
        "Write" => (
            "Create or overwrite a file.",
            json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["file_path", "content"]
            }),
        ),
        "Edit" => (
            "Replace a unique old_string with new_string in a file.",
            json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string" },
                    "old_string": { "type": "string" },
                    "new_string": { "type": "string" }
                },
                "required": ["file_path", "old_string", "new_string"]
            }),
        ),
        _ => return None,
    };
    Some(json!({
        "name": name,
        "description": description,
        "input_schema": schema,
    }))
}
