//! Lightweight execpolicy: prefix rules with Allow / Prompt / Forbidden.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum Decision {
    #[default]
    Allow = 0,
    Prompt = 1,
    Forbidden = 2,
}

impl Decision {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_lowercase().as_str() {
            "allow" => Some(Self::Allow),
            "prompt" => Some(Self::Prompt),
            "forbidden" | "deny" => Some(Self::Forbidden),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefixRule {
    pub pattern: Vec<String>,
    #[serde(default)]
    pub decision: Decision,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub justification: Option<String>,
}

impl PrefixRule {
    fn matches(&self, cmd: &[String]) -> bool {
        if cmd.len() < self.pattern.len() {
            return false;
        }
        cmd.iter().zip(self.pattern.iter()).all(|(a, b)| a == b)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExecPolicyFile {
    #[serde(default)]
    pub rules: Vec<PrefixRule>,
}

#[derive(Clone, Debug)]
pub struct ExecPolicy {
    rules: Vec<PrefixRule>,
}

#[derive(Clone, Debug)]
pub struct PolicyMatch {
    pub decision: Decision,
    pub matched_prefix: Vec<String>,
    pub justification: Option<String>,
}

impl ExecPolicy {
    pub fn new(rules: Vec<PrefixRule>) -> Self {
        Self { rules }
    }

    pub fn default_codex_like() -> Self {
        Self::new(vec![
            PrefixRule {
                pattern: vec!["git".into(), "reset".into(), "--hard".into()],
                decision: Decision::Forbidden,
                justification: Some("destructive git reset --hard".into()),
            },
            PrefixRule {
                pattern: vec!["rm".into(), "-rf".into()],
                decision: Decision::Prompt,
                justification: Some("recursive force delete".into()),
            },
            PrefixRule {
                pattern: vec!["curl".into()],
                decision: Decision::Prompt,
                justification: Some("network access via curl".into()),
            },
            PrefixRule {
                pattern: vec!["wget".into()],
                decision: Decision::Prompt,
                justification: Some("network access via wget".into()),
            },
            PrefixRule {
                pattern: vec!["sudo".into()],
                decision: Decision::Forbidden,
                justification: Some("elevated privileges".into()),
            },
        ])
    }

    pub fn from_json_str(raw: &str) -> Result<Self, String> {
        let file: ExecPolicyFile =
            serde_json::from_str(raw).map_err(|e| format!("invalid exec policy JSON: {e}"))?;
        Ok(Self::new(file.rules))
    }

    pub fn load_path(path: &Path) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| format!("read exec policy {}: {e}", path.display()))?;
        Self::from_json_str(&raw)
    }

    pub fn add_allow_prefix(&mut self, prefix: Vec<String>) {
        if prefix.is_empty() {
            return;
        }
        self.rules.retain(|r| r.pattern != prefix);
        self.rules.push(PrefixRule {
            pattern: prefix,
            decision: Decision::Allow,
            justification: None,
        });
    }

    pub fn evaluate(&self, cmd: &[String]) -> PolicyMatch {
        let mut best: Option<&PrefixRule> = None;
        for rule in &self.rules {
            if !rule.matches(cmd) {
                continue;
            }
            match best {
                None => best = Some(rule),
                Some(prev) if rule.pattern.len() > prev.pattern.len() => best = Some(rule),
                Some(prev)
                    if rule.pattern.len() == prev.pattern.len()
                        && rule.decision > prev.decision =>
                {
                    best = Some(rule)
                }
                _ => {}
            }
        }
        match best {
            Some(rule) => PolicyMatch {
                decision: rule.decision,
                matched_prefix: rule.pattern.clone(),
                justification: rule.justification.clone(),
            },
            None => PolicyMatch {
                decision: Decision::Prompt,
                matched_prefix: cmd.to_vec(),
                justification: Some("no matching execpolicy rule".into()),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forbidden_git_reset_hard() {
        let p = ExecPolicy::default_codex_like();
        let m = p.evaluate(&tokens(&["git", "reset", "--hard", "HEAD"]));
        assert_eq!(m.decision, Decision::Forbidden);
    }

    #[test]
    fn allow_amendment_prefix() {
        let mut p = ExecPolicy::default_codex_like();
        p.add_allow_prefix(tokens(&["echo", "ok"]));
        let m = p.evaluate(&tokens(&["echo", "ok"]));
        assert_eq!(m.decision, Decision::Allow);
    }

    fn tokens(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }
}
