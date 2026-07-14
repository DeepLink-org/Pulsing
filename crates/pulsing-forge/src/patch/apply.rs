use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::parse_patch;
use super::parser::Hunk;
use super::parser::ParseError;
use super::parser::UpdateFileChunk;
use super::seek_sequence;

#[derive(Debug, thiserror::Error)]
pub enum ApplyPatchError {
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error("{0}")]
    Io(String),
    #[error("{0}")]
    Apply(String),
}

pub fn apply_patch_to_fs(patch: &str, cwd: &Path) -> Result<String, ApplyPatchError> {
    apply_patch_to_fs_bounded(patch, cwd, cwd)
}

pub fn apply_patch_to_fs_bounded(
    patch: &str,
    base: &Path,
    root: &Path,
) -> Result<String, ApplyPatchError> {
    let args = parse_patch(patch)?;
    apply_hunks(&args.hunks, base, root)
}

fn apply_hunks(hunks: &[Hunk], base: &Path, root: &Path) -> Result<String, ApplyPatchError> {
    if hunks.is_empty() {
        return Err(ApplyPatchError::Apply("No files were modified.".into()));
    }
    let mut added = Vec::new();
    let mut modified = Vec::new();
    let mut deleted = Vec::new();

    for hunk in hunks {
        match hunk {
            Hunk::AddFile { path, contents, .. } => {
                let path =
                    super::resolve_patch_path(path, base, root).map_err(ApplyPatchError::Apply)?;
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent).map_err(|e| ApplyPatchError::Io(e.to_string()))?;
                }
                fs::write(&path, contents).map_err(|e| ApplyPatchError::Io(e.to_string()))?;
                added.push(path);
            }
            Hunk::DeleteFile { path, .. } => {
                let path =
                    super::resolve_patch_path(path, base, root).map_err(ApplyPatchError::Apply)?;
                if path.is_dir() {
                    return Err(ApplyPatchError::Apply(format!(
                        "Refusing to delete directory {}",
                        path.display()
                    )));
                }
                fs::remove_file(&path).map_err(|e| ApplyPatchError::Io(e.to_string()))?;
                deleted.push(path);
            }
            Hunk::UpdateFile {
                path,
                move_path,
                chunks,
                ..
            } => {
                let path =
                    super::resolve_patch_path(path, base, root).map_err(ApplyPatchError::Apply)?;
                let new_contents = derive_new_contents(&path, chunks)?;
                if let Some(dest) = move_path {
                    let dest_path = super::resolve_patch_path(dest, base, root)
                        .map_err(ApplyPatchError::Apply)?;
                    if let Some(parent) = dest_path.parent() {
                        fs::create_dir_all(parent)
                            .map_err(|e| ApplyPatchError::Io(e.to_string()))?;
                    }
                    fs::write(&dest_path, &new_contents)
                        .map_err(|e| ApplyPatchError::Io(e.to_string()))?;
                    fs::remove_file(&path).map_err(|e| ApplyPatchError::Io(e.to_string()))?;
                    modified.push(dest_path);
                } else {
                    fs::write(&path, &new_contents)
                        .map_err(|e| ApplyPatchError::Io(e.to_string()))?;
                    modified.push(path);
                }
            }
        }
    }

    let mut out = Vec::new();
    print_summary(&added, &modified, &deleted, &mut out)
        .map_err(|e| ApplyPatchError::Io(e.to_string()))?;
    Ok(String::from_utf8_lossy(&out).into_owned())
}

fn derive_new_contents(path: &Path, chunks: &[UpdateFileChunk]) -> Result<String, ApplyPatchError> {
    let original_contents = fs::read_to_string(path)
        .map_err(|e| ApplyPatchError::Io(format!("Failed to read {}: {e}", path.display())))?;
    let mut original_lines: Vec<String> = original_contents.split('\n').map(String::from).collect();
    if original_lines.last().is_some_and(String::is_empty) {
        original_lines.pop();
    }
    let replacements = compute_replacements(&original_lines, path, chunks)?;
    let mut new_lines = apply_replacements(original_lines, &replacements);
    if !new_lines.last().is_some_and(String::is_empty) {
        new_lines.push(String::new());
    }
    Ok(new_lines.join("\n"))
}

fn compute_replacements(
    original_lines: &[String],
    path: &Path,
    chunks: &[UpdateFileChunk],
) -> Result<Vec<(usize, usize, Vec<String>)>, ApplyPatchError> {
    let mut replacements = Vec::new();
    let mut line_index = 0usize;
    for chunk in chunks {
        if let Some(ctx_line) = &chunk.change_context {
            if let Some(idx) = seek_sequence::seek_sequence(
                original_lines,
                std::slice::from_ref(ctx_line),
                line_index,
                false,
            ) {
                line_index = idx + 1;
            } else {
                return Err(ApplyPatchError::Apply(format!(
                    "Failed to find context '{ctx_line}' in {}",
                    path.display()
                )));
            }
        }
        if chunk.old_lines.is_empty() {
            let insertion_idx = if original_lines.last().is_some_and(String::is_empty) {
                original_lines.len() - 1
            } else {
                original_lines.len()
            };
            replacements.push((insertion_idx, 0, chunk.new_lines.clone()));
            continue;
        }
        let mut pattern: &[String] = &chunk.old_lines;
        let mut found =
            seek_sequence::seek_sequence(original_lines, pattern, line_index, chunk.is_end_of_file);
        let mut new_slice: &[String] = &chunk.new_lines;
        if found.is_none() && pattern.last().is_some_and(String::is_empty) {
            pattern = &pattern[..pattern.len() - 1];
            if new_slice.last().is_some_and(String::is_empty) {
                new_slice = &new_slice[..new_slice.len() - 1];
            }
            found = seek_sequence::seek_sequence(
                original_lines,
                pattern,
                line_index,
                chunk.is_end_of_file,
            );
        }
        if let Some(start_idx) = found {
            replacements.push((start_idx, pattern.len(), new_slice.to_vec()));
            line_index = start_idx + pattern.len();
        } else {
            return Err(ApplyPatchError::Apply(format!(
                "Failed to find expected lines in {}:\n{}",
                path.display(),
                chunk.old_lines.join("\n")
            )));
        }
    }
    replacements.sort_by_key(|(index, _, _)| *index);
    Ok(replacements)
}

fn apply_replacements(
    mut lines: Vec<String>,
    replacements: &[(usize, usize, Vec<String>)],
) -> Vec<String> {
    for (start_idx, old_len, new_segment) in replacements.iter().rev() {
        for _ in 0..*old_len {
            if *start_idx < lines.len() {
                lines.remove(*start_idx);
            }
        }
        for (offset, new_line) in new_segment.iter().enumerate() {
            lines.insert(start_idx + offset, new_line.clone());
        }
    }
    lines
}

fn print_summary(
    added: &[PathBuf],
    modified: &[PathBuf],
    deleted: &[PathBuf],
    out: &mut Vec<u8>,
) -> std::io::Result<()> {
    for p in added {
        writeln!(out, "A {}", p.display())?;
    }
    for p in modified {
        writeln!(out, "M {}", p.display())?;
    }
    for p in deleted {
        writeln!(out, "D {}", p.display())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_file_patch() {
        let dir = tempfile::tempdir().unwrap();
        let patch = concat!(
            "*** Begin Patch\n",
            "*** Add File: hello.txt\n",
            "+hello world\n",
            "*** End Patch\n"
        );
        let summary = apply_patch_to_fs(patch, dir.path()).unwrap();
        assert!(summary.contains("A "));
        assert_eq!(
            fs::read_to_string(dir.path().join("hello.txt")).unwrap(),
            "hello world\n"
        );
    }

    #[test]
    fn rejects_path_escape() {
        let dir = tempfile::tempdir().unwrap();
        let patch = concat!(
            "*** Begin Patch\n",
            "*** Add File: ../escape.txt\n",
            "+pwned\n",
            "*** End Patch\n"
        );
        let err = apply_patch_to_fs(patch, dir.path()).unwrap_err();
        assert!(err.to_string().contains("outside working directory"));
        assert!(!dir.path().parent().unwrap().join("escape.txt").exists());
    }

    #[test]
    #[cfg(unix)]
    fn rejects_symlink_escape() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        let outside = dir.path().join("outside");
        std::fs::create_dir(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, workspace.join("link")).unwrap();
        let patch = concat!(
            "*** Begin Patch\n",
            "*** Add File: link/pwned.txt\n",
            "+escaped\n",
            "*** End Patch\n"
        );
        let err = apply_patch_to_fs(patch, &workspace).unwrap_err();
        assert!(err.to_string().contains("outside working directory"));
        assert!(!outside.join("pwned.txt").exists());
    }
}
