use std::collections::HashSet;
use std::fs;
use std::path::Path;

use pulsing_workspace::WorkspaceLayout;

const DEFAULT_MAX_DEPTH: usize = 8;

#[derive(Clone, Debug)]
pub struct FileTreeNode {
    pub id: String,
    pub label: String,
    pub is_dir: bool,
    pub expanded: bool,
    pub children: Vec<FileTreeNode>,
}

impl FileTreeNode {
    pub fn new_file(id: String, label: String) -> Self {
        Self {
            id,
            label,
            is_dir: false,
            expanded: false,
            children: Vec::new(),
        }
    }

    pub fn new_dir(id: String, label: String, expanded: bool, children: Vec<FileTreeNode>) -> Self {
        Self {
            id,
            label,
            is_dir: true,
            expanded,
            children,
        }
    }
}

pub fn build_file_tree(layout: &WorkspaceLayout, previous: &[FileTreeNode]) -> Vec<FileTreeNode> {
    let expanded = collect_expanded_ids(previous);
    build_dir(layout, &layout.root, 0, DEFAULT_MAX_DEPTH, &expanded)
}

pub fn count_files(items: &[FileTreeNode]) -> usize {
    let mut n = 0;
    walk(items, &mut |item| {
        if !item.is_dir {
            n += 1;
        }
    });
    n
}

fn collect_expanded_ids(items: &[FileTreeNode]) -> HashSet<String> {
    let mut out = HashSet::new();
    walk(items, &mut |item| {
        if item.is_dir && item.expanded {
            out.insert(item.id.clone());
        }
    });
    out
}

fn walk(items: &[FileTreeNode], f: &mut dyn FnMut(&FileTreeNode)) {
    for item in items {
        f(item);
        walk(&item.children, f);
    }
}

fn build_dir(
    layout: &WorkspaceLayout,
    dir: &Path,
    depth: usize,
    max_depth: usize,
    expanded: &HashSet<String>,
) -> Vec<FileTreeNode> {
    if depth > max_depth {
        return Vec::new();
    }

    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let mut names: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    names.sort_by_key(|e| e.file_name());

    let mut dirs = Vec::new();
    let mut files = Vec::new();

    for entry in names {
        let path = entry.path();
        let Some(rel) = layout.rel_to_root(&path) else {
            continue;
        };
        if should_skip(&rel) {
            continue;
        }

        let id = rel.to_string_lossy().into_owned();
        let label = entry.file_name().to_string_lossy().into_owned();

        if path.is_dir() {
            let children = build_dir(layout, &path, depth + 1, max_depth, expanded);
            let is_expanded = expanded.contains(&id);
            dirs.push(FileTreeNode::new_dir(
                id,
                label,
                is_expanded || depth == 0,
                children,
            ));
        } else {
            files.push(FileTreeNode::new_file(id, label));
        }
    }

    dirs.extend(files);
    dirs
}

fn should_skip(rel: &Path) -> bool {
    let s = rel.to_string_lossy();
    if s.starts_with(".git") || s.contains("/.git/") {
        return true;
    }
    if s.starts_with("target") || s.contains("/target/") {
        return true;
    }
    if s.starts_with("node_modules") || s.contains("/node_modules/") {
        return true;
    }
    false
}
