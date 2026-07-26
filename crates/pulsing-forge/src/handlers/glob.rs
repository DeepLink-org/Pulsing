use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{err, json_str, ok};
use crate::context::ToolCallContext;
use crate::error::ToolError;
use crate::executor::{ToolExecutor, ToolExecutorFuture};

/// Cap on returned matches, mirroring Grep's `GREP_MAX` — keeps tool output
/// bounded for the model without requiring pagination plumbing.
const GLOB_MAX_MATCHES: usize = 500;

pub struct GlobHandler;

impl ToolExecutor for GlobHandler {
    fn tool_name(&self) -> &str {
        "Glob"
    }

    fn spec(&self) -> crate::registry::ToolSpec {
        super::builtin_spec(self.tool_name())
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, arguments: Value) -> ToolExecutorFuture<'a> {
        let cwd = ctx.cwd.clone();
        Box::pin(async move { glob_impl(&cwd, &arguments) })
    }
}

fn resolve_path(cwd: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    }
}

fn glob_impl(cwd: &Path, args: &Value) -> Result<crate::result::ToolResult, ToolError> {
    let pattern = json_str(args, "pattern")?;
    if Path::new(pattern).is_absolute() {
        return err(
            "pattern must be relative to path/cwd; absolute glob patterns are not supported",
        );
    }
    let base = args
        .get("path")
        .and_then(|v| v.as_str())
        .map(|p| resolve_path(cwd, p))
        .unwrap_or_else(|| cwd.to_path_buf());
    if !base.exists() {
        return err(format!("path does not exist: {}", base.display()));
    }
    let pat_str = base.join(pattern).to_string_lossy().replace('\\', "/");
    let mut matches: Vec<String> = glob::glob(&pat_str)
        .map_err(|e| ToolError::respond(format!("invalid glob pattern {pattern:?}: {e}")))?
        .filter_map(|p| p.ok())
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    matches.sort();
    let total = matches.len();
    matches.truncate(GLOB_MAX_MATCHES);
    if matches.is_empty() {
        return ok("(no matches)");
    }
    if total > GLOB_MAX_MATCHES {
        matches.push(format!(
            "… truncated: showing {GLOB_MAX_MATCHES} of {total} matches …"
        ));
    }
    ok(matches.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(pattern: &str, path: Option<&str>) -> Value {
        let mut v = json!({"pattern": pattern});
        if let Some(p) = path {
            v["path"] = json!(p);
        }
        v
    }

    #[test]
    fn finds_matching_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "x").unwrap();
        std::fs::write(dir.path().join("b.rs"), "x").unwrap();
        let out = glob_impl(dir.path(), &args("*.txt", None)).unwrap();
        assert!(!out.is_error);
        assert!(out.content.ends_with("a.txt"), "{}", out.content);
    }

    #[test]
    fn reports_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        let out = glob_impl(dir.path(), &args("*.nope", None)).unwrap();
        assert!(!out.is_error);
        assert_eq!(out.content, "(no matches)");
    }

    #[test]
    fn rejects_missing_path() {
        let dir = tempfile::tempdir().unwrap();
        let out = glob_impl(dir.path(), &args("*", Some("does/not/exist"))).unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("does not exist"));
    }

    #[test]
    fn relative_path_resolves_against_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("a.txt"), "x").unwrap();
        let out = glob_impl(dir.path(), &args("*.txt", Some("sub"))).unwrap();
        assert!(!out.is_error);
        assert!(out.content.ends_with("a.txt"), "{}", out.content);
    }

    #[test]
    fn rejects_absolute_pattern() {
        // Without this check `base.join(pattern)` silently drops `base`
        // (PathBuf::join replaces the whole path when the RHS is absolute),
        // letting the pattern glob outside of `path`/cwd entirely.
        let dir = tempfile::tempdir().unwrap();
        let out = glob_impl(dir.path(), &args("/etc/*", None)).unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("absolute"), "{}", out.content);
    }

    #[test]
    fn truncates_at_glob_max_with_clear_message() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..GLOB_MAX_MATCHES + 10 {
            std::fs::write(dir.path().join(format!("f{i}.txt")), "x").unwrap();
        }
        let out = glob_impl(dir.path(), &args("*.txt", None)).unwrap();
        assert!(!out.is_error);
        assert!(
            out.content.contains(&format!(
                "truncated: showing {GLOB_MAX_MATCHES} of {} matches",
                GLOB_MAX_MATCHES + 10
            )),
            "{}",
            out.content
        );
    }
}
