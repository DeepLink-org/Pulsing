//! End-to-end apply_patch scenarios (portable fixture layout from codex-apply-patch).
//!
//! Each scenario under `vendor/codex-rs/apply-patch/tests/fixtures/scenarios/` has:
//! `input/`, `patch.txt`, `expected/` — see that directory's README.

use pulsing_forge::patch::apply_patch_to_fs;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Scenarios not yet matching codex-apply-patch reference (tracked gaps).
const KNOWN_GAP_SCENARIOS: &[&str] = &[
    "011_add_overwrites_existing_file",
    "015_failure_after_partial_success_leaves_changes",
];

#[derive(Debug, Clone, PartialEq, Eq)]
enum Entry {
    File(Vec<u8>),
    Dir,
}

#[test]
fn codex_apply_patch_scenarios() {
    let scenarios_dir = scenarios_root();
    if !scenarios_dir.is_dir() {
        eprintln!(
            "skip codex_apply_patch_scenarios: missing {}",
            scenarios_dir.display()
        );
        return;
    }
    let mut passed = 0usize;
    let mut unexpected_failures: Vec<String> = Vec::new();
    for entry in fs::read_dir(&scenarios_dir).expect("read scenarios dir") {
        let entry = entry.expect("scenario entry");
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let ok = run_apply_patch_scenario(&path).is_ok();
        if ok {
            passed += 1;
        } else if !KNOWN_GAP_SCENARIOS.contains(&name.as_str()) {
            unexpected_failures.push(name);
        }
    }
    assert!(
        passed >= 10,
        "expected many passing scenarios, got {passed}"
    );
    assert!(
        unexpected_failures.is_empty(),
        "unexpected apply_patch scenario failures: {unexpected_failures:?}"
    );
}

fn scenarios_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../vendor/codex-rs/apply-patch/tests/fixtures/scenarios")
}

fn run_apply_patch_scenario(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;
    let input_dir = dir.join("input");
    if input_dir.is_dir() {
        copy_dir_recursive(&input_dir, tmp.path())?;
    }
    let patch = fs::read_to_string(dir.join("patch.txt"))?;
    let _ = apply_patch_to_fs(&patch, tmp.path());
    let expected_dir = dir.join("expected");
    let expected = snapshot_dir(&expected_dir)?;
    let actual = snapshot_dir(tmp.path())?;
    if actual != expected {
        return Err(format!(
            "filesystem mismatch for {}",
            dir.file_name().unwrap_or_default().to_string_lossy()
        )
        .into());
    }
    Ok(())
}

fn snapshot_dir(root: &Path) -> Result<BTreeMap<PathBuf, Entry>, Box<dyn std::error::Error>> {
    let mut entries = BTreeMap::new();
    if root.is_dir() {
        snapshot_dir_recursive(root, root, &mut entries)?;
    }
    Ok(entries)
}

fn snapshot_dir_recursive(
    base: &Path,
    dir: &Path,
    entries: &mut BTreeMap<PathBuf, Entry>,
) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.file_name().is_some_and(|n| n == ".gitattributes") {
            continue;
        }
        let rel = path.strip_prefix(base)?.to_path_buf();
        let metadata = fs::metadata(&path)?;
        if metadata.is_dir() {
            entries.insert(rel.clone(), Entry::Dir);
            snapshot_dir_recursive(base, &path, entries)?;
        } else if metadata.is_file() {
            entries.insert(rel, Entry::File(fs::read(&path)?));
        }
    }
    Ok(())
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let dest_path = dst.join(entry.file_name());
        let metadata = fs::metadata(&path)?;
        if metadata.is_dir() {
            fs::create_dir_all(&dest_path)?;
            copy_dir_recursive(&path, &dest_path)?;
        } else if metadata.is_file() {
            if let Some(parent) = dest_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&path, &dest_path)?;
        }
    }
    Ok(())
}
