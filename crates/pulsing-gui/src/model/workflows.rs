use std::path::PathBuf;

use pulsing_workspace::WorkspaceLayout;

pub fn list_workflow_scripts(layout: &WorkspaceLayout) -> Vec<PathBuf> {
    let dir = layout.root.join(".pulsing").join("workflows");
    if !dir.is_dir() {
        return Vec::new();
    }
    let mut scripts: Vec<PathBuf> = std::fs::read_dir(&dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "py"))
        .collect();
    scripts.sort();
    scripts
}
