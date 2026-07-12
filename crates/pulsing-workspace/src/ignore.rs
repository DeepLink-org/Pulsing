use std::path::Path;

/// Skip paths that should not appear in workspace checkpoints.
pub fn should_skip(rel: &Path) -> bool {
    let s = rel.to_string_lossy();
    if s.is_empty() {
        return true;
    }
    if s.starts_with(".pulsing/history/") || s == ".pulsing/history" {
        return true;
    }
    if s.starts_with(".git/") || s == ".git" {
        return true;
    }
    for part in rel.components() {
        let name = part.as_os_str().to_string_lossy();
        if matches!(
            name.as_ref(),
            "target"
                | "node_modules"
                | "__pycache__"
                | ".venv"
                | "venv"
                | ".mypy_cache"
                | ".pytest_cache"
                | ".ruff_cache"
                | "dist"
                | "build"
        ) {
            return true;
        }
    }
    false
}
