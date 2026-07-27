use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;

use super::write::{assert_canonical_within_cwd, resolve_within_cwd};
use super::{err, json_str, ok};
use crate::context::ToolCallContext;
use crate::error::ToolError;
use crate::executor::{ToolExecutor, ToolExecutorFuture};

pub struct EditHandler;

impl ToolExecutor for EditHandler {
    fn tool_name(&self) -> &str {
        "Edit"
    }

    fn spec(&self) -> crate::registry::ToolSpec {
        super::builtin_spec(self.tool_name())
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, arguments: Value) -> ToolExecutorFuture<'a> {
        let cwd = ctx.cwd.clone();
        Box::pin(async move { edit_impl(&cwd, &arguments) })
    }
}

fn edit_impl(cwd: &Path, args: &Value) -> Result<crate::result::ToolResult, ToolError> {
    let path = json_str(args, "file_path")?;
    let abs = resolve_within_cwd(cwd, path).map_err(ToolError::respond)?;
    let old = json_str(args, "old_string")?;
    let new = json_str(args, "new_string")?;

    if !abs.exists() {
        return err(format!("file not found: {}", abs.display()));
    }
    if !abs.is_file() {
        return err(format!("not a file: {}", abs.display()));
    }
    assert_canonical_within_cwd(cwd, &abs).map_err(ToolError::respond)?;
    let text = std::fs::read_to_string(&abs)
        .map_err(|e| ToolError::respond(format!("failed to read {}: {e}", abs.display())))?;

    let count = text.matches(old).count();
    if count == 0 {
        return err(format!("old_string not found in {}", abs.display()));
    }
    if count > 1 {
        return err(format!(
            "old_string is not unique in {} ({count} occurrences); refusing ambiguous edit",
            abs.display()
        ));
    }

    let updated = text.replacen(old, new, 1);
    write_atomic(&abs, &updated)
        .map_err(|e| ToolError::respond(format!("failed to write {}: {e}", abs.display())))?;
    ok("ok")
}

/// Writes `contents` to `path` via a sibling temp file + rename, so a failed
/// write (disk full, process killed, etc.) never leaves `path` truncated or
/// half-written. The rename is atomic on the same filesystem (POSIX and
/// Windows via `std::fs::rename`).
fn write_atomic(path: &Path, contents: &str) -> std::io::Result<()> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("edit");
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_path = dir.join(format!(".{file_name}.{}.{unique}.tmp", std::process::id()));

    let result = (|| -> std::io::Result<()> {
        let mut f = std::fs::File::create(&tmp_path)?;
        f.write_all(contents.as_bytes())?;
        f.sync_all()
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
        return result;
    }
    if let Ok(meta) = std::fs::metadata(path) {
        let _ = std::fs::set_permissions(&tmp_path, meta.permissions());
    }
    std::fs::rename(&tmp_path, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(file_path: &str, old: &str, new: &str) -> Value {
        json!({"file_path": file_path, "old_string": old, "new_string": new})
    }

    #[test]
    fn replaces_unique_occurrence() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("a.txt");
        std::fs::write(&f, "hello world").unwrap();
        let out = edit_impl(dir.path(), &args("a.txt", "world", "there")).unwrap();
        assert!(!out.is_error);
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "hello there");
    }

    #[test]
    fn rejects_missing_old_string() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("a.txt");
        std::fs::write(&f, "hello world").unwrap();
        let out = edit_impl(dir.path(), &args("a.txt", "nope", "x")).unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("not found"));
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "hello world");
    }

    #[test]
    fn rejects_ambiguous_old_string_with_count() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("a.txt");
        std::fs::write(&f, "a a a").unwrap();
        let out = edit_impl(dir.path(), &args("a.txt", "a", "b")).unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("3 occurrences"), "{}", out.content);
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "a a a");
    }

    #[test]
    fn rejects_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let out = edit_impl(dir.path(), &args("missing.txt", "a", "b")).unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("file not found"));
    }

    #[test]
    fn rejects_directory_path() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let out = edit_impl(dir.path(), &args("sub", "a", "b")).unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("not a file"));
    }

    #[test]
    fn rejects_relative_escape_outside_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("a.txt");
        std::fs::write(&f, "hello world").unwrap();
        let err = edit_impl(dir.path(), &args("../a.txt", "world", "there")).unwrap_err();
        assert!(
            matches!(err, ToolError::RespondToModel(ref msg) if msg.contains("outside working directory")),
            "{err:?}"
        );
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "hello world");
        assert!(!dir.path().parent().unwrap().join("a.txt").exists());
    }

    #[test]
    fn rejects_absolute_path_outside_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let outside =
            std::env::temp_dir().join(format!("pulsing-forge-edit-outside-{}", std::process::id()));
        std::fs::write(&outside, "secret").unwrap();
        let err = edit_impl(
            dir.path(),
            &args(outside.to_str().unwrap(), "secret", "leaked"),
        )
        .unwrap_err();
        assert!(
            matches!(err, ToolError::RespondToModel(ref msg) if msg.contains("outside working directory")),
            "{err:?}"
        );
        assert_eq!(std::fs::read_to_string(&outside).unwrap(), "secret");
        let _ = std::fs::remove_file(&outside);
    }

    #[test]
    fn allows_absolute_path_inside_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("inside.txt");
        std::fs::write(&f, "hello world").unwrap();
        let out = edit_impl(dir.path(), &args(f.to_str().unwrap(), "world", "there")).unwrap();
        assert!(!out.is_error);
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "hello there");
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape_outside_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let outside_file = outside.path().join("secret.txt");
        std::fs::write(&outside_file, "secret").unwrap();
        let link = dir.path().join("link.txt");
        std::os::unix::fs::symlink(&outside_file, &link).unwrap();
        let err = edit_impl(dir.path(), &args("link.txt", "secret", "leaked")).unwrap_err();
        assert!(
            matches!(err, ToolError::RespondToModel(ref msg) if msg.contains("outside working directory")),
            "{err:?}"
        );
        assert_eq!(std::fs::read_to_string(&outside_file).unwrap(), "secret");
        assert!(link.is_symlink());
    }

    #[test]
    fn no_leftover_temp_file_after_success() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("a.txt");
        std::fs::write(&f, "hello world").unwrap();
        edit_impl(dir.path(), &args("a.txt", "world", "there")).unwrap();
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }
}
