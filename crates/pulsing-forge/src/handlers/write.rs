use std::path::{Component, Path, PathBuf};

use serde_json::Value;

use super::{json_str, ok};
use crate::context::ToolCallContext;
use crate::error::ToolError;
use crate::executor::{ToolExecutor, ToolExecutorFuture};

pub struct WriteHandler;

impl ToolExecutor for WriteHandler {
    fn tool_name(&self) -> &str {
        "Write"
    }

    fn spec(&self) -> crate::registry::ToolSpec {
        super::builtin_spec(self.tool_name())
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, arguments: Value) -> ToolExecutorFuture<'a> {
        let cwd = ctx.cwd.clone();
        Box::pin(async move { write_impl(&cwd, &arguments) })
    }
}

fn write_impl(cwd: &Path, args: &Value) -> Result<crate::result::ToolResult, ToolError> {
    let path = json_str(args, "file_path")?;
    let content = json_str(args, "content")?;
    let target = resolve_within_cwd(cwd, path).map_err(ToolError::respond)?;

    let existed = target.exists();
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            ToolError::respond(format!(
                "failed to create parent directory {}: {e}",
                parent.display()
            ))
        })?;
    }
    atomic_write(&target, content)
        .map_err(|e| ToolError::respond(format!("failed to write {}: {e}", target.display())))?;
    ok(if existed { "overwritten" } else { "created" })
}

/// Resolve `path` against `cwd` and reject any target that would land outside
/// of it. Unlike `apply_patch`/`Bash`, `Write` runs with no OS-level sandbox,
/// so this is the only boundary check standing between the model and the rest
/// of the filesystem.
pub(crate) fn resolve_within_cwd(cwd: &Path, path: &str) -> Result<PathBuf, String> {
    let joined = if Path::new(path).is_absolute() {
        Path::new(path).to_path_buf()
    } else {
        cwd.join(path)
    };
    let target = normalize_lexically(&joined);
    let root = normalize_lexically(cwd);
    if !target.starts_with(&root) {
        return Err(format!(
            "refusing to write outside working directory: {} (cwd: {})",
            target.display(),
            root.display()
        ));
    }
    reject_symlink_escape(cwd, &target)?;
    Ok(target)
}

/// Follow symlinks on the longest existing prefix of `target` and reject if the
/// resolved location escapes `cwd`. Lexical `..` checks alone are not enough: a
/// symlink inside the workspace can still redirect writes to the rest of the
/// filesystem.
fn reject_symlink_escape(cwd: &Path, target: &Path) -> Result<(), String> {
    let cwd_canon = cwd
        .canonicalize()
        .map_err(|e| format!("invalid working directory {}: {e}", cwd.display()))?;
    let mut probe = target.to_path_buf();
    loop {
        if probe.exists() {
            let canon = probe
                .canonicalize()
                .map_err(|e| format!("cannot resolve {}: {e}", probe.display()))?;
            if !canon.starts_with(&cwd_canon) {
                return Err(format!(
                    "refusing to write outside working directory: {} (cwd: {})",
                    target.display(),
                    cwd.display()
                ));
            }
            return Ok(());
        }
        match probe.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => probe = parent.to_path_buf(),
            _ => return Ok(()),
        }
    }
}

/// Reject paths whose canonical location escapes `cwd` (e.g. symlinks to outside).
/// Call only after confirming `target` exists.
pub(crate) fn assert_canonical_within_cwd(cwd: &Path, target: &Path) -> Result<(), String> {
    let root = cwd
        .canonicalize()
        .map_err(|e| format!("failed to resolve working directory: {e}"))?;
    let canon = target
        .canonicalize()
        .map_err(|e| format!("failed to resolve {}: {e}", target.display()))?;
    if canon.starts_with(&root) {
        Ok(())
    } else {
        Err(format!(
            "refusing to write outside working directory: {} (cwd: {})",
            canon.display(),
            root.display()
        ))
    }
}

/// Collapse `.`/`..` components without touching the filesystem, since the
/// write target may not exist yet. Mirrors the common `path-clean` behavior:
/// a leading `..` beyond the root is kept as-is rather than panicking.
fn normalize_lexically(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => match out.components().next_back() {
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                _ => out.push(component),
            },
            other => out.push(other),
        }
    }
    out
}

/// Write via a sibling temp file + rename so a mid-write failure (disk full,
/// process kill) never leaves `target` truncated or half-written.
fn atomic_write(target: &Path, content: &str) -> std::io::Result<()> {
    let parent = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = target
        .file_name()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing file name"))?
        .to_string_lossy();
    let tmp_path = parent.join(format!(".{file_name}.{}.tmp", uuid::Uuid::new_v4()));

    let write_result = std::fs::write(&tmp_path, content);
    if write_result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
        return write_result;
    }
    std::fs::rename(&tmp_path, target).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp_path);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(
        cwd: &Path,
        file_path: &str,
        content: &str,
    ) -> Result<crate::result::ToolResult, ToolError> {
        write_impl(
            cwd,
            &serde_json::json!({"file_path": file_path, "content": content}),
        )
    }

    #[test]
    fn creates_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let out = call(dir.path(), "out.txt", "hello").unwrap();
        assert!(!out.is_error);
        assert_eq!(out.content, "created");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("out.txt")).unwrap(),
            "hello"
        );
    }

    #[test]
    fn overwrites_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("out.txt"), "old").unwrap();
        let out = call(dir.path(), "out.txt", "new").unwrap();
        assert!(!out.is_error);
        assert_eq!(out.content, "overwritten");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("out.txt")).unwrap(),
            "new"
        );
    }

    #[test]
    fn creates_deep_parent_dirs_inside_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let out = call(dir.path(), "a/b/c/out.txt", "deep").unwrap();
        assert!(!out.is_error);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a/b/c/out.txt")).unwrap(),
            "deep"
        );
    }

    #[test]
    fn rejects_relative_escape_outside_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let err = call(dir.path(), "../escape.txt", "x").unwrap_err();
        assert!(
            matches!(err, ToolError::RespondToModel(msg) if msg.contains("outside working directory"))
        );
        assert!(!dir.path().parent().unwrap().join("escape.txt").exists());
    }

    #[test]
    fn rejects_absolute_path_outside_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let err = call(
            dir.path(),
            "/etc/pulsing-forge-write-test-should-not-exist",
            "x",
        )
        .unwrap_err();
        assert!(
            matches!(err, ToolError::RespondToModel(msg) if msg.contains("outside working directory"))
        );
    }

    #[test]
    fn allows_absolute_path_inside_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let abs = dir.path().join("inside.txt");
        let out = call(dir.path(), abs.to_str().unwrap(), "ok").unwrap();
        assert!(!out.is_error);
        assert_eq!(std::fs::read_to_string(&abs).unwrap(), "ok");
    }

    #[test]
    fn rejects_symlink_escape_outside_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        let workspace = dir.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            symlink(&outside, workspace.join("link")).unwrap();
            let err = call(&workspace, "link/escape.txt", "x").unwrap_err();
            assert!(
                matches!(err, ToolError::RespondToModel(msg) if msg.contains("outside working directory"))
            );
            assert!(!outside.join("escape.txt").exists());
        }
    }
}
