//! Detect `apply_patch` in shell argv and verify hunks against the filesystem.

use std::path::{Path, PathBuf};

use crate::patch::heredoc::{HeredocError, extract_apply_patch_from_bash};
use crate::patch::{
    ApplyPatchArgs, Hunk, ParseError, apply_patch_to_fs_bounded, normalize_lexically, parse_patch,
    resolve_patch_path,
};

const APPLY_PATCH_COMMANDS: [&str; 2] = ["apply_patch", "applypatch"];

#[derive(Debug, Clone, PartialEq)]
pub enum MaybeApplyPatch {
    Body(ApplyPatchArgs),
    PatchParseError(ParseError),
    ShellParseError(String),
    ImplicitInvocation,
    NotApplyPatch,
}

/// If `argv` is an apply_patch invocation, return parsed args; else `NotApplyPatch`.
pub fn maybe_parse_apply_patch(argv: &[String]) -> MaybeApplyPatch {
    if let [body] = argv
        && parse_patch(body).is_ok()
    {
        return MaybeApplyPatch::ImplicitInvocation;
    }
    match argv {
        [cmd, body] if APPLY_PATCH_COMMANDS.contains(&cmd.as_str()) => match parse_patch(body) {
            Ok(mut source) => {
                source.patch = body.clone();
                MaybeApplyPatch::Body(source)
            }
            Err(e) => MaybeApplyPatch::PatchParseError(e),
        },
        [shell, flag, script] if is_shell_flag(shell, flag) => {
            if parse_patch(script).is_ok() {
                return MaybeApplyPatch::ImplicitInvocation;
            }
            match extract_apply_patch_from_bash(script) {
                Ok((body, workdir)) => match parse_patch(&body) {
                    Ok(mut source) => {
                        source.patch = body.clone();
                        source.workdir = workdir;
                        MaybeApplyPatch::Body(source)
                    }
                    Err(e) => MaybeApplyPatch::PatchParseError(e),
                },
                Err(HeredocError::CommandDidNotStartWithApplyPatch) => {
                    MaybeApplyPatch::NotApplyPatch
                }
                Err(e) => MaybeApplyPatch::ShellParseError(format!("{e:?}")),
            }
        }
        _ => MaybeApplyPatch::NotApplyPatch,
    }
}

pub fn apply_parsed_patch(args: &ApplyPatchArgs, cwd: &Path) -> Result<String, String> {
    let effective = resolve_effective_cwd(cwd, args.workdir.as_deref())?;
    verify_hunks_readable(&args.hunks, &effective, cwd)?;
    apply_patch_to_fs_bounded(&args.patch, &effective, cwd).map_err(|e| e.to_string())
}

fn resolve_effective_cwd(cwd: &Path, workdir: Option<&str>) -> Result<PathBuf, String> {
    match workdir {
        Some(w) => resolve_patch_path(Path::new(w), cwd, cwd),
        None => Ok(normalize_lexically(cwd)),
    }
}

fn verify_hunks_readable(hunks: &[Hunk], base: &Path, root: &Path) -> Result<(), String> {
    for hunk in hunks {
        match hunk {
            Hunk::AddFile { path, .. } => {
                let path = resolve_patch_path(path, base, root)?;
                if path.exists() {
                    return Err(format!(
                        "add file blocked: {} already exists",
                        path.display()
                    ));
                }
            }
            Hunk::DeleteFile { path } => {
                let path = resolve_patch_path(path, base, root)?;
                if !path.is_file() {
                    return Err(format!("file not found: {}", path.display()));
                }
            }
            Hunk::UpdateFile {
                path, move_path, ..
            } => {
                let path = resolve_patch_path(path, base, root)?;
                if !path.is_file() {
                    return Err(format!("file not found: {}", path.display()));
                }
                if let Some(dest) = move_path {
                    resolve_patch_path(dest, base, root)?;
                }
            }
        }
    }
    Ok(())
}

fn is_shell_flag(shell: &str, flag: &str) -> bool {
    let name = Path::new(shell)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(shell)
        .to_ascii_lowercase();
    matches!(name.as_str(), "sh" | "bash" | "zsh") && matches!(flag, "-c" | "-lc")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_direct_apply_patch_argv() {
        let argv = vec![
            "apply_patch".into(),
            "*** Begin Patch\n*** End Patch\n".into(),
        ];
        assert!(matches!(
            maybe_parse_apply_patch(&argv),
            MaybeApplyPatch::Body(_)
        ));
    }

    #[test]
    fn detects_bash_heredoc_via_tree_sitter() {
        let script = "apply_patch <<'PATCH'\n*** Begin Patch\n*** End Patch\nPATCH";
        let argv = vec!["bash".into(), "-lc".into(), script.into()];
        assert!(matches!(
            maybe_parse_apply_patch(&argv),
            MaybeApplyPatch::Body(_)
        ));
    }

    #[test]
    fn rejects_workdir_escape() {
        let dir = tempfile::tempdir().unwrap();
        let args = ApplyPatchArgs {
            hunks: vec![Hunk::AddFile {
                path: PathBuf::from("ok.txt"),
                contents: "x\n".into(),
            }],
            patch: String::new(),
            workdir: Some("../escape".into()),
            environment_id: None,
        };
        let err = apply_parsed_patch(&args, dir.path()).unwrap_err();
        assert!(err.contains("outside working directory"));
    }
}
