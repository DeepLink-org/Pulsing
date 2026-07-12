//! Exec approval types and shell gate (Codex-aligned).

mod gate;
mod types;

pub use gate::{
    args_dangerously_disable_sandbox, effective_sandbox_policy, ensure_shell_allowed,
    new_exec_policy, tokenize_shell_command,
};
pub use types::{
    ApprovalPolicy, ExecApprovalRequest, RequestPermissionProfile, RequestPermissionsArgs,
    RequestPermissionsResponse, ReviewDecision,
};

use std::sync::Mutex;

/// Session-scoped approval cache (prefix allow-list + strict review flag).
///
/// Only `ApprovedForSession` (`session_prefixes`) and execpolicy amendments
/// persist beyond a single exec call — a plain `Approved` decision is a
/// one-time grant for that command execution only (Codex `ReviewDecision`
/// semantics) and must never be cached here, or every subsequent identical
/// command would silently skip approval for the lifetime of this cache.
#[derive(Default)]
pub struct ApprovalCache {
    session_prefixes: Mutex<Vec<Vec<String>>>,
    strict_auto_review: Mutex<bool>,
    turn_grants: Mutex<Option<RequestPermissionProfile>>,
    session_grants: Mutex<Option<RequestPermissionProfile>>,
}

impl ApprovalCache {
    pub fn is_prefix_allowed(&self, cmd: &[String]) -> bool {
        let prefixes = self.session_prefixes.lock().unwrap();
        prefixes.iter().any(|p| cmd.starts_with(p.as_slice()))
    }

    pub fn allow_prefix_for_session(&self, prefix: Vec<String>) {
        if prefix.is_empty() {
            return;
        }
        let mut prefixes = self.session_prefixes.lock().unwrap();
        if !prefixes.iter().any(|p| p == &prefix) {
            prefixes.push(prefix);
        }
    }

    pub fn set_strict_auto_review(&self, on: bool) {
        *self.strict_auto_review.lock().unwrap() = on;
    }

    pub fn strict_auto_review(&self) -> bool {
        *self.strict_auto_review.lock().unwrap()
    }

    /// Record an approved `request_permissions` grant (turn vs session scope).
    pub fn record_permission_grant(&self, profile: RequestPermissionProfile, scope: &str) {
        let slot = if scope == "session" {
            &self.session_grants
        } else {
            &self.turn_grants
        };
        *slot.lock().unwrap() = Some(profile);
    }

    /// Merged session + turn grants (turn overlays session).
    pub fn effective_grants(&self) -> RequestPermissionProfile {
        let session = self.session_grants.lock().unwrap().clone();
        let turn = self.turn_grants.lock().unwrap().clone();
        match (session, turn) {
            (None, None) => RequestPermissionProfile::default(),
            (Some(s), None) => s,
            (None, Some(t)) => t,
            (Some(mut s), Some(t)) => {
                if t.network.is_some() {
                    s.network = t.network;
                }
                if let Some(t_fs) = t.file_system {
                    s.file_system = Some(merge_file_system_json(s.file_system.take(), t_fs));
                }
                s
            }
        }
    }

    pub fn network_granted(&self) -> bool {
        self.effective_grants().network_enabled()
    }

    /// Clears turn-scoped state. Session prefix allow-list is preserved.
    pub fn clear_turn_state(&self) {
        self.set_strict_auto_review(false);
        *self.turn_grants.lock().unwrap() = None;
    }
}

fn merge_file_system_json(
    base: Option<serde_json::Value>,
    overlay: serde_json::Value,
) -> serde_json::Value {
    let Some(mut base_obj) = base.and_then(|v| v.as_object().cloned()) else {
        return overlay;
    };
    let Some(overlay_obj) = overlay.as_object() else {
        return overlay;
    };
    for key in ["read", "write", "entries"] {
        if let Some(overlay_arr) = overlay_obj.get(key).and_then(|v| v.as_array()) {
            let entry = base_obj
                .entry(key.to_string())
                .or_insert(serde_json::json!([]));
            if let Some(base_arr) = entry.as_array_mut() {
                for item in overlay_arr {
                    if !base_arr.contains(item) {
                        base_arr.push(item.clone());
                    }
                }
            }
        }
    }
    serde_json::Value::Object(base_obj)
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn session_grant_survives_turn_clear() {
        let cache = ApprovalCache::default();
        cache.record_permission_grant(
            RequestPermissionProfile {
                network: Some(serde_json::json!({"enabled": true})),
                file_system: None,
            },
            "session",
        );
        cache.record_permission_grant(
            RequestPermissionProfile {
                network: Some(serde_json::json!({"enabled": false})),
                file_system: None,
            },
            "turn",
        );
        cache.clear_turn_state();
        assert!(!cache.strict_auto_review());
        assert!(cache.network_granted());
    }
}
