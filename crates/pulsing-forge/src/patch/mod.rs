//! Codex `apply_patch` format parser + local filesystem apply.
//! Parser adapted from `codex-apply-patch` (Apache-2.0).

mod apply;
mod heredoc;
mod invocation;
mod parser;
mod seek_sequence;
mod streaming_parser;

pub use apply::{apply_patch_to_fs, apply_patch_to_fs_bounded};
pub use invocation::{MaybeApplyPatch, apply_parsed_patch, maybe_parse_apply_patch};
pub use parser::Hunk;
pub use parser::ParseError;
pub use parser::UpdateFileChunk;
pub use parser::parse_patch;

use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct ApplyPatchArgs {
    pub hunks: Vec<Hunk>,
    pub patch: String,
    pub workdir: Option<String>,
    pub environment_id: Option<String>,
}

/// Join `path` against `base`, normalize `.`/`..`, and reject targets outside `root`.
pub(crate) fn resolve_patch_path(path: &Path, base: &Path, root: &Path) -> Result<PathBuf, String> {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    };
    let target = normalize_lexically(&joined);
    let boundary = normalize_lexically(root);
    if !target.starts_with(&boundary) {
        return Err(format!(
            "refusing to apply patch outside working directory: {} (cwd: {})",
            target.display(),
            boundary.display()
        ));
    }
    reject_symlink_escape(&target, &boundary)?;
    Ok(target)
}

/// Follow symlinks on the nearest existing ancestor and reject escapes outside `root`.
fn reject_symlink_escape(target: &Path, root: &Path) -> Result<(), String> {
    let root_canon = root.canonicalize().map_err(|e| e.to_string())?;
    let mut probe = target.to_path_buf();
    loop {
        match probe.canonicalize() {
            Ok(canon) => {
                if !canon.starts_with(&root_canon) {
                    return Err(format!(
                        "refusing to apply patch outside working directory: {} (cwd: {})",
                        target.display(),
                        root.display()
                    ));
                }
                return Ok(());
            }
            Err(_) if probe.pop() => continue,
            Err(_) => return Ok(()),
        }
    }
}

pub(crate) fn resolve_path_against_base(path: &Path, base: &Path) -> Result<PathBuf, String> {
    resolve_patch_path(path, base, base)
}

pub(crate) fn normalize_lexically(path: &Path) -> PathBuf {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_relative_escape_outside_root() {
        let root = PathBuf::from("/tmp/workspace");
        let err = resolve_patch_path(Path::new("../escape.txt"), &root, &root).unwrap_err();
        assert!(err.contains("outside working directory"));
    }

    #[test]
    fn allows_nested_path_within_root() {
        let root = PathBuf::from("/tmp/workspace");
        let got = resolve_patch_path(Path::new("a/b.txt"), &root, &root).unwrap();
        assert_eq!(got, PathBuf::from("/tmp/workspace/a/b.txt"));
    }

    #[test]
    fn allows_parent_hop_within_root_from_subdir() {
        let root = PathBuf::from("/tmp/workspace");
        let base = root.join("subdir");
        let got = resolve_patch_path(Path::new("../sibling.txt"), &base, &root).unwrap();
        assert_eq!(got, PathBuf::from("/tmp/workspace/sibling.txt"));
    }
}
