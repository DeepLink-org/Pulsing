use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

use crate::execpolicy::Decision;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum ApprovalPolicy {
    /// Prompt on execpolicy Prompt / sandbox escalation (default).
    #[default]
    OnRequest,
    /// Auto-approve all prompts (tests / trusted hosts).
    Always,
    /// Deny all prompts.
    Never,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecApprovalRequest {
    pub command: Vec<String>,
    pub cwd: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sandbox_permissions: Option<String>,
    pub policy_decision: Decision,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub justification: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposed_execpolicy_amendment: Option<Vec<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReviewDecision {
    Approved,
    ApprovedExecpolicyAmendment {
        proposed_execpolicy_amendment: Vec<String>,
    },
    ApprovedForSession,
    Denied,
    Abort,
}

impl ReviewDecision {
    pub fn is_approved(&self) -> bool {
        matches!(
            self,
            Self::Approved | Self::ApprovedExecpolicyAmendment { .. } | Self::ApprovedForSession
        )
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestPermissionProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_system: Option<serde_json::Value>,
}

impl RequestPermissionProfile {
    pub fn is_empty(&self) -> bool {
        self.network.is_none() && self.file_system.is_none()
    }

    /// True when every present section is null/{} or has no effective grant
    /// (Codex `NetworkPermissions::is_empty` / `FileSystemPermissions::is_empty`).
    pub fn is_effectively_empty(&self) -> bool {
        let net_empty = self
            .network
            .as_ref()
            .map(is_permission_section_empty)
            .unwrap_or(true);
        let fs_empty = self
            .file_system
            .as_ref()
            .map(is_permission_section_empty)
            .unwrap_or(true);
        net_empty && fs_empty
    }

    pub fn has_actionable_request(&self) -> bool {
        !self.is_empty() && !self.is_effectively_empty()
    }

    /// Reject file_system paths that resolve outside `cwd` (P0 sandbox safety).
    pub fn validate_paths(&self, cwd: &Path) -> Result<(), String> {
        if let Some(fs) = &self.file_system {
            validate_file_system_paths(cwd, fs)?;
        }
        Ok(())
    }

    pub fn network_enabled(&self) -> bool {
        self.network
            .as_ref()
            .and_then(|v| v.get("enabled"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }
}

fn validate_file_system_paths(cwd: &Path, fs: &serde_json::Value) -> Result<(), String> {
    let root = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
    for raw in collect_file_system_path_strings(fs) {
        let resolved = resolve_permission_path(&root, &raw)?;
        if resolved != root && !resolved.starts_with(&root) {
            return Err(format!(
                "file_system permission path outside working directory: {raw}"
            ));
        }
    }
    Ok(())
}

fn collect_file_system_path_strings(fs: &serde_json::Value) -> Vec<String> {
    let mut out = Vec::new();
    let Some(obj) = fs.as_object() else {
        return out;
    };
    for key in ["read", "write"] {
        if let Some(arr) = obj.get(key).and_then(|v| v.as_array()) {
            for item in arr {
                if let Some(s) = item.as_str() {
                    out.push(s.to_string());
                }
            }
        }
    }
    if let Some(entries) = obj.get("entries").and_then(|v| v.as_array()) {
        for entry in entries {
            let Some(entry_obj) = entry.as_object() else {
                continue;
            };
            let Some(path_val) = entry_obj.get("path") else {
                continue;
            };
            if let Some(s) = path_val.as_str() {
                out.push(s.to_string());
                continue;
            }
            let Some(path_obj) = path_val.as_object() else {
                continue;
            };
            if path_obj.get("type").and_then(|v| v.as_str()) == Some("path")
                && let Some(s) = path_obj.get("path").and_then(|v| v.as_str())
            {
                out.push(s.to_string());
            }
        }
    }
    out
}

fn resolve_permission_path(cwd: &Path, raw: &str) -> Result<PathBuf, String> {
    let path = Path::new(raw);
    if path.is_absolute() {
        return path
            .canonicalize()
            .or_else(|_| Ok::<_, std::io::Error>(path.to_path_buf()))
            .map_err(|e| format!("invalid file_system path {raw:?}: {e}"));
    }
    for component in path.components() {
        if matches!(component, Component::ParentDir) {
            return Err(format!(
                "file_system permission path must not contain '..': {raw}"
            ));
        }
    }
    Ok(cwd.join(path))
}

fn is_permission_section_empty(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => true,
        serde_json::Value::Object(map) => {
            if map.is_empty() {
                return true;
            }
            if map.len() == 1 && map.get("enabled").is_some_and(|enabled| enabled.is_null()) {
                return true;
            }
            if let Some(entries) = map.get("entries").and_then(|v| v.as_array()) {
                return entries.is_empty();
            }
            let read_empty = map
                .get("read")
                .map(|v| v.as_array().is_none_or(|a| a.is_empty()))
                .unwrap_or(true);
            let write_empty = map
                .get("write")
                .map(|v| v.as_array().is_none_or(|a| a.is_empty()))
                .unwrap_or(true);
            if map.contains_key("read") || map.contains_key("write") {
                return read_empty && write_empty;
            }
            false
        }
        _ => false,
    }
}

#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn rejects_unknown_fields_on_args() {
        let err = serde_json::from_value::<RequestPermissionsArgs>(serde_json::json!({
            "permissions": {"network": {"enabled": true}},
            "extra": true,
        }))
        .unwrap_err();
        assert!(err.to_string().contains("unknown field"));
    }

    #[test]
    fn rejects_file_system_path_outside_cwd() {
        let profile = RequestPermissionProfile {
            network: None,
            file_system: Some(serde_json::json!({
                "write": ["/etc/passwd"]
            })),
        };
        let err = profile
            .validate_paths(Path::new("/tmp/workspace"))
            .unwrap_err();
        assert!(err.contains("outside working directory"));
    }

    #[test]
    fn rejects_parent_dir_in_relative_path() {
        let profile = RequestPermissionProfile {
            network: None,
            file_system: Some(serde_json::json!({
                "read": ["../secret"]
            })),
        };
        let err = profile
            .validate_paths(Path::new("/tmp/workspace"))
            .unwrap_err();
        assert!(err.contains(".."));
    }

    #[test]
    fn allows_in_cwd_relative_path() {
        let profile = RequestPermissionProfile {
            network: None,
            file_system: Some(serde_json::json!({
                "write": ["subdir/file.txt"]
            })),
        };
        profile
            .validate_paths(Path::new("/tmp"))
            .expect("in-cwd path");
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestPermissionsArgs {
    #[serde(
        default,
        rename = "environment_id",
        alias = "environmentId",
        skip_serializing_if = "Option::is_none"
    )]
    pub environment_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub permissions: RequestPermissionProfile,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestPermissionsResponse {
    pub permissions: RequestPermissionProfile,
    #[serde(default)]
    pub scope: String,
    #[serde(default)]
    pub strict_auto_review: bool,
}
