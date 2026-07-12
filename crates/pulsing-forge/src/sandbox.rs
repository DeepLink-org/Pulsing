//! Bash sandbox helpers (MVP). Logic aligned with Craft/Codex `build_bash_exec`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SandboxPolicy {
    Off,
    Restricted,
    Bwrap,
}

pub fn normalize_policy(raw: &str) -> SandboxPolicy {
    match raw.trim().to_lowercase().as_str() {
        "restricted" => SandboxPolicy::Restricted,
        "bwrap" => SandboxPolicy::Bwrap,
        _ => SandboxPolicy::Off,
    }
}

#[derive(Clone, Debug)]
pub struct BashExecPlan {
    pub argv: Vec<String>,
    pub env: Option<HashMap<String, String>>,
    pub label: String,
}

/// Builds the argv/env for running `command` under the given sandbox policy.
///
/// `login` requests login-shell semantics (`bash -l`, sourcing profile files)
/// instead of a plain `-c` invocation. It must never itself disable the
/// sandbox wrapper — callers that need an unsandboxed login shell must go
/// through `dangerously_disable_sandbox` explicitly, which is gated by
/// approval upstream.
pub fn build_bash_exec(
    command: &str,
    cwd: Option<&Path>,
    policy: SandboxPolicy,
    dangerously_disable_sandbox: bool,
    login: bool,
) -> BashExecPlan {
    let wd = cwd
        .map(|p| p.canonicalize().unwrap_or_else(|_| p.to_path_buf()))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let pol = if dangerously_disable_sandbox {
        SandboxPolicy::Off
    } else {
        policy
    };

    if pol == SandboxPolicy::Off {
        let argv = if login {
            vec!["bash".into(), "-lc".into(), command.to_string()]
        } else {
            vec!["/bin/sh".into(), "-c".into(), command.to_string()]
        };
        return BashExecPlan {
            argv,
            env: None,
            label: "subprocess shell (sandbox=off)".into(),
        };
    }

    if pol == SandboxPolicy::Bwrap && which_bwrap() {
        let mut argv = vec![
            "bwrap".to_string(),
            "--die-with-parent".into(),
            "--unshare-pid".into(),
            "--tmpfs".into(),
            "/tmp".into(),
            "--proc".into(),
            "/proc".into(),
            "--dev".into(),
            "/dev".into(),
            "--ro-bind".into(),
            "/usr".into(),
            "/usr".into(),
            "--ro-bind".into(),
            "/bin".into(),
            "/bin".into(),
            "--ro-bind".into(),
            "/lib".into(),
            "/lib".into(),
        ];
        if Path::new("/lib64").is_dir() {
            argv.extend(["--ro-bind".into(), "/lib64".into(), "/lib64".into()]);
        }
        let wd_str = wd.to_string_lossy().to_string();
        let shell_flag = if login { "-lc" } else { "-c" };
        argv.extend([
            "--bind".into(),
            wd_str.clone(),
            "/work".into(),
            "--chdir".into(),
            "/work".into(),
            "bash".into(),
            shell_flag.into(),
            command.to_string(),
        ]);
        return BashExecPlan {
            argv,
            env: None,
            label: "bubblewrap (minimal profile; Linux)".into(),
        };
    }

    let mut env = HashMap::new();
    env.insert(
        "HOME".into(),
        std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()),
    );
    env.insert("PATH".into(), "/usr/bin:/bin:/usr/local/bin".into());
    env.insert(
        "LANG".into(),
        std::env::var("LANG").unwrap_or_else(|_| "C.UTF-8".into()),
    );
    env.insert(
        "USER".into(),
        std::env::var("USER").unwrap_or_else(|_| "user".into()),
    );
    let mut argv = vec!["env".into(), "-i".into()];
    for (k, v) in &env {
        argv.push(format!("{k}={v}"));
    }
    let label = if login {
        argv.extend(["bash".into(), "-lc".into(), command.to_string()]);
        "restricted env (env -i + bash -l)"
    } else {
        argv.extend([
            "bash".into(),
            "--norc".into(),
            "--noprofile".into(),
            "-c".into(),
            command.to_string(),
        ]);
        "restricted env (env -i + bash --norc)"
    };
    BashExecPlan {
        argv,
        env: Some(env),
        label: label.into(),
    }
}

fn which_bwrap() -> bool {
    std::process::Command::new("which")
        .arg("bwrap")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: `login=true` must still route through the restricted-env
    /// wrapper (`env -i` + no-inherit env) instead of falling back to a
    /// bare, unsandboxed `bash -lc` that inherits the full host environment.
    #[test]
    fn login_restricted_policy_still_clears_env() {
        let plan = build_bash_exec("echo hi", None, SandboxPolicy::Restricted, false, true);
        assert_eq!(plan.argv[0], "env");
        assert_eq!(plan.argv[1], "-i");
        assert!(
            plan.env.is_some(),
            "restricted plan must carry a scrubbed env map"
        );
        assert!(plan.argv.iter().any(|a| a == "-lc"));
    }

    #[test]
    fn login_off_policy_uses_login_shell_directly() {
        let plan = build_bash_exec("echo hi", None, SandboxPolicy::Off, false, true);
        assert_eq!(plan.argv, vec!["bash", "-lc", "echo hi"]);
    }

    #[test]
    fn non_login_off_policy_uses_plain_sh() {
        let plan = build_bash_exec("echo hi", None, SandboxPolicy::Off, false, false);
        assert_eq!(plan.argv, vec!["/bin/sh", "-c", "echo hi"]);
    }

    /// `dangerously_disable_sandbox` must win over any requested policy
    /// regardless of `login`, and is the only switch allowed to disable
    /// the sandbox wrapper.
    #[test]
    fn dangerously_disable_sandbox_overrides_policy_for_login_and_non_login() {
        for login in [true, false] {
            let plan = build_bash_exec("echo hi", None, SandboxPolicy::Restricted, true, login);
            assert!(plan.label.contains("sandbox=off"));
            assert!(plan.env.is_none());
        }
    }

    #[test]
    fn login_bwrap_policy_keeps_bwrap_wrapper_when_available() {
        if !which_bwrap() {
            return;
        }
        let plan = build_bash_exec("echo hi", None, SandboxPolicy::Bwrap, false, true);
        assert_eq!(plan.argv[0], "bwrap");
        assert!(plan.argv.iter().any(|a| a == "-lc"));
    }
}
