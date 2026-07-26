use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{err, json_str, ok};
use crate::context::ToolCallContext;
use crate::error::ToolError;
use crate::executor::{ToolExecutor, ToolExecutorFuture};

/// Hard cap for a single read (or for the slice returned by a paginated read).
/// Larger files must be read with `offset`/`limit` instead of being rejected outright.
const READ_CAP: usize = 2 * 1024 * 1024;

pub struct ReadHandler;

impl ToolExecutor for ReadHandler {
    fn tool_name(&self) -> &str {
        "Read"
    }

    fn spec(&self) -> crate::registry::ToolSpec {
        super::builtin_spec(self.tool_name())
    }

    fn handle<'a>(&'a self, ctx: &'a ToolCallContext, arguments: Value) -> ToolExecutorFuture<'a> {
        let cwd = ctx.cwd.clone();
        Box::pin(async move { read_impl(&cwd, &arguments) })
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

fn read_impl(cwd: &Path, args: &Value) -> Result<crate::result::ToolResult, ToolError> {
    let path = json_str(args, "file_path")?;
    let abs = resolve_path(cwd, path);
    let offset = args.get("offset").and_then(Value::as_u64);
    let limit = args.get("limit").and_then(Value::as_u64);

    if abs.is_dir() {
        return err(format!(
            "Path is a directory, not a file: {}",
            abs.display()
        ));
    }

    if offset.is_some() || limit.is_some() {
        let start_line = offset.unwrap_or(1).max(1) as usize;
        return read_range(&abs, start_line, limit.map(|l| l as usize));
    }

    let data = std::fs::read(&abs).map_err(|e| read_error(&abs, &e))?;
    if data.len() > READ_CAP {
        return err(format!(
            "File too large for Read tool ({} bytes > {READ_CAP} cap); retry with offset/limit to page through it.",
            data.len()
        ));
    }
    let text = String::from_utf8(data)
        .map_err(|_| ToolError::respond(format!("Not valid UTF-8: {}", abs.display())))?;
    ok(text)
}

/// Streams the file line by line so a paginated read doesn't need the whole file in memory.
fn read_range(
    path: &Path,
    start_line: usize,
    limit: Option<usize>,
) -> Result<crate::result::ToolResult, ToolError> {
    let file = std::fs::File::open(path).map_err(|e| read_error(path, &e))?;
    let end_line = limit.map(|n| start_line.saturating_add(n));
    let mut out = String::new();

    for (idx, line) in io::BufReader::new(file).lines().enumerate() {
        let lineno = idx + 1;
        if lineno < start_line {
            continue;
        }
        if end_line.is_some_and(|end| lineno >= end) {
            break;
        }
        let line = line.map_err(|_| {
            ToolError::respond(format!(
                "Not valid UTF-8 at line {lineno}: {}",
                path.display()
            ))
        })?;
        out.push_str(&line);
        out.push('\n');
        if out.len() > READ_CAP {
            return err("Requested range too large for Read tool; use a smaller limit.");
        }
    }
    ok(out)
}

fn read_error(path: &Path, e: &io::Error) -> ToolError {
    let reason = match e.kind() {
        io::ErrorKind::NotFound => "No such file".to_string(),
        io::ErrorKind::PermissionDenied => "Permission denied".to_string(),
        _ => e.to_string(),
    };
    ToolError::respond(format!("{reason}: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors how `ToolRuntime::call_tool` turns a propagated `ToolError` into
    /// a model-visible error result (see `runtime.rs`), so tests can assert on
    /// `ToolResult` regardless of whether a handler used `err(..)` or `?`.
    fn run(cwd: &Path, args: Value) -> crate::result::ToolResult {
        match read_impl(cwd, &args) {
            Ok(r) => r,
            Err(e) => crate::result::ToolResult::err(e.to_string()),
        }
    }

    #[test]
    fn reads_whole_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        let out = run(dir.path(), serde_json::json!({"file_path": "a.txt"}));
        assert!(!out.is_error);
        assert_eq!(out.content, "hello");
    }

    #[test]
    fn rejects_oversized_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("big.txt"), vec![b'x'; READ_CAP + 1]).unwrap();
        let out = run(dir.path(), serde_json::json!({"file_path": "big.txt"}));
        assert!(out.is_error);
        assert!(out.content.contains("too large"));
    }

    #[test]
    fn rejects_non_utf8() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("bin.dat"), [0xff, 0xfe, 0x00, 0xff]).unwrap();
        let out = run(dir.path(), serde_json::json!({"file_path": "bin.dat"}));
        assert!(out.is_error);
        assert!(out.content.contains("Not valid UTF-8"));
    }

    #[test]
    fn missing_file_reports_path() {
        let dir = tempfile::tempdir().unwrap();
        let out = run(dir.path(), serde_json::json!({"file_path": "missing.txt"}));
        assert!(out.is_error);
        assert!(out.content.contains("missing.txt"));
        assert!(out.content.contains("No such file"));
    }

    #[test]
    fn rejects_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let out = run(dir.path(), serde_json::json!({"file_path": "sub"}));
        assert!(out.is_error);
        assert!(out.content.contains("directory"));
    }

    #[test]
    fn offset_and_limit_page_through_lines() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("lines.txt"), "l1\nl2\nl3\nl4\nl5\n").unwrap();
        let out = run(
            dir.path(),
            serde_json::json!({"file_path": "lines.txt", "offset": 2, "limit": 2}),
        );
        assert!(!out.is_error);
        assert_eq!(out.content, "l2\nl3\n");
    }

    #[test]
    fn offset_beyond_cap_still_streams_without_loading_whole_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut content = String::new();
        for i in 0..100_000 {
            content.push_str(&format!("line {i}\n"));
        }
        std::fs::write(dir.path().join("huge.txt"), &content).unwrap();
        let out = run(
            dir.path(),
            serde_json::json!({"file_path": "huge.txt", "offset": 99_999, "limit": 1}),
        );
        assert!(!out.is_error);
        assert_eq!(out.content, "line 99998\n");
    }
}
