use std::path::{Path, PathBuf};

use regex::Regex;
use serde_json::Value;

use super::{err, json_str, ok};
use crate::context::ToolCallContext;
use crate::error::ToolError;
use crate::executor::{ToolExecutor, ToolExecutorFuture};

const GREP_MAX: usize = 200;
const GREP_PATTERN_MAX: usize = 1000;

pub struct GrepHandler;

impl ToolExecutor for GrepHandler {
    fn tool_name(&self) -> &str {
        "Grep"
    }

    fn spec(&self) -> crate::registry::ToolSpec {
        super::builtin_spec(self.tool_name())
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, arguments: Value) -> ToolExecutorFuture<'a> {
        let cwd = ctx.cwd.clone();
        Box::pin(async move { grep_impl(&cwd, &arguments) })
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

/// When the search root lives under `cwd`, skip files that resolve outside `cwd`
/// (e.g. symlinks) while still allowing an explicit absolute `path` outside cwd.
fn search_boundary(cwd: &Path, root: &Path) -> Option<PathBuf> {
    let cwd_canon = cwd.canonicalize().ok()?;
    let root_canon = root.canonicalize().ok()?;
    root_canon.starts_with(&cwd_canon).then_some(cwd_canon)
}

fn grep_impl(cwd: &Path, args: &Value) -> Result<crate::result::ToolResult, ToolError> {
    let raw_pat = json_str(args, "pattern")?;
    if raw_pat.len() > GREP_PATTERN_MAX {
        return err(format!(
            "Pattern too long ({} > {GREP_PATTERN_MAX} chars); simplify the regex",
            raw_pat.len()
        ));
    }
    let root = args
        .get("path")
        .and_then(|v| v.as_str())
        .map(|p| resolve_path(cwd, p))
        .unwrap_or_else(|| cwd.to_path_buf());
    let glob_pat = args.get("glob").and_then(|v| v.as_str());
    let cre = Regex::new(raw_pat).map_err(|e| ToolError::respond(format!("Invalid regex: {e}")))?;
    if !root.exists() {
        return err("path not found");
    }
    let boundary = search_boundary(cwd, &root);
    let mut hits: Vec<String> = Vec::new();
    let mut total = 0usize;
    if root.is_file() {
        consider_file(
            &root,
            &cre,
            glob_pat,
            boundary.as_deref(),
            &mut hits,
            &mut total,
        );
    } else {
        walk(
            &root,
            &cre,
            glob_pat,
            boundary.as_deref(),
            &mut hits,
            &mut total,
        );
    }
    if hits.is_empty() {
        ok("(no matches)")
    } else {
        let extra = if total > GREP_MAX {
            format!("\n… truncated: showing {GREP_MAX} of {total} matches …")
        } else {
            String::new()
        };
        ok(format!("{}{}", hits.join("\n"), extra))
    }
}

fn walk(
    dir: &Path,
    cre: &Regex,
    glob_pat: Option<&str>,
    boundary: Option<&Path>,
    hits: &mut Vec<String>,
    total: &mut usize,
) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for ent in read.flatten() {
        let path = ent.path();
        if path.is_dir() {
            walk(&path, cre, glob_pat, boundary, hits, total);
        } else if path.is_file() {
            consider_file(&path, cre, glob_pat, boundary, hits, total);
        }
    }
}

fn consider_file(
    fp: &Path,
    cre: &Regex,
    glob_pat: Option<&str>,
    boundary: Option<&Path>,
    hits: &mut Vec<String>,
    total: &mut usize,
) {
    if let Some(b) = boundary {
        let Ok(fp_canon) = fp.canonicalize() else {
            return;
        };
        if !fp_canon.starts_with(b) {
            return;
        }
    }
    if let Some(g) = glob_pat {
        let name = fp.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !glob_match(g, name) {
            return;
        }
    }
    let Ok(text) = std::fs::read_to_string(fp) else {
        return;
    };
    for (i, line) in text.lines().enumerate() {
        if cre.is_match(line) {
            *total += 1;
            if hits.len() < GREP_MAX {
                hits.push(format!(
                    "{}:{}:{}",
                    fp.display(),
                    i + 1,
                    &line[..line.len().min(500)]
                ));
            }
        }
    }
}

fn glob_match(pat: &str, name: &str) -> bool {
    glob::Pattern::new(pat)
        .map(|p| p.matches(name))
        .unwrap_or(false)
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
    fn finds_matching_lines() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello\nworld\nhello again").unwrap();
        let out = grep_impl(dir.path(), &args("hello", None)).unwrap();
        assert!(!out.is_error);
        assert_eq!(out.content.lines().count(), 2);
        assert!(out.content.contains("a.txt:1:hello"));
    }

    #[test]
    fn reports_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello world").unwrap();
        let out = grep_impl(dir.path(), &args("nope", None)).unwrap();
        assert!(!out.is_error);
        assert_eq!(out.content, "(no matches)");
    }

    #[test]
    fn rejects_invalid_regex() {
        // Invalid input surfaces as `Err(ToolError::RespondToModel)`, which the
        // runtime turns into an error `ToolResult` (see `runtime.rs`).
        let dir = tempfile::tempdir().unwrap();
        let err = grep_impl(dir.path(), &args("(unclosed", None)).unwrap_err();
        assert!(err.to_string().contains("Invalid regex"), "{err}");
    }

    #[test]
    fn rejects_missing_path() {
        let dir = tempfile::tempdir().unwrap();
        let out = grep_impl(dir.path(), &args("x", Some("does/not/exist"))).unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("path not found"));
    }

    #[test]
    fn is_immune_to_catastrophic_backtracking_pattern() {
        // The `regex` crate guarantees linear-time matching, so a pattern that
        // would cause exponential backtracking in a backtracking engine (e.g.
        // Python's `re`) must still return promptly here.
        let dir = tempfile::tempdir().unwrap();
        let evil_line = "a".repeat(40) + "!";
        std::fs::write(dir.path().join("a.txt"), &evil_line).unwrap();
        let start = std::time::Instant::now();
        let out = grep_impl(dir.path(), &args("(a+)+b", None)).unwrap();
        assert!(start.elapsed() < std::time::Duration::from_secs(2));
        assert!(!out.is_error);
        assert_eq!(out.content, "(no matches)");
    }

    #[test]
    fn truncates_at_grep_max_with_clear_message() {
        let dir = tempfile::tempdir().unwrap();
        let many_lines = "hit\n".repeat(GREP_MAX + 10);
        std::fs::write(dir.path().join("a.txt"), many_lines).unwrap();
        let out = grep_impl(dir.path(), &args("hit", None)).unwrap();
        assert!(!out.is_error);
        assert!(
            out.content.contains(&format!(
                "truncated: showing {GREP_MAX} of {} matches",
                GREP_MAX + 10
            )),
            "{}",
            out.content
        );
    }

    #[test]
    fn resolves_relative_path_against_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("a.txt"), "needle\n").unwrap();
        let out = grep_impl(dir.path(), &args("needle", Some("sub"))).unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("needle"));
    }

    #[test]
    fn rejects_pattern_too_long() {
        let dir = tempfile::tempdir().unwrap();
        let out = grep_impl(dir.path(), &args(&"a".repeat(GREP_PATTERN_MAX + 1), None)).unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("Pattern too long"));
    }

    #[test]
    #[cfg(unix)]
    fn skips_symlink_targets_outside_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "outside_secret\n").unwrap();
        std::os::unix::fs::symlink(
            outside.path().join("secret.txt"),
            dir.path().join("link.txt"),
        )
        .unwrap();
        let out = grep_impl(dir.path(), &args("outside_secret", None)).unwrap();
        assert!(!out.is_error);
        assert_eq!(out.content, "(no matches)");
    }
}
