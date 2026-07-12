use std::collections::HashSet;
use std::fs;
use std::path::Path;

use gpui::SharedString;
use gpui_component::tree::TreeItem;
use pulsing_workspace::WorkspaceLayout;

const DEFAULT_MAX_DEPTH: usize = 8;

pub fn build_file_tree(layout: &WorkspaceLayout, previous: &[TreeItem]) -> Vec<TreeItem> {
    let expanded = collect_expanded_ids(previous);
    build_dir(layout, &layout.root, 0, DEFAULT_MAX_DEPTH, &expanded)
}

pub fn count_files(items: &[TreeItem]) -> usize {
    let mut n = 0;
    walk(items, &mut |item| {
        if !item.is_folder() {
            n += 1;
        }
    });
    n
}

fn collect_expanded_ids(items: &[TreeItem]) -> HashSet<String> {
    let mut out = HashSet::new();
    walk(items, &mut |item| {
        if item.is_folder() && item.is_expanded() {
            out.insert(item.id.to_string());
        }
    });
    out
}

fn walk(items: &[TreeItem], f: &mut dyn FnMut(&TreeItem)) {
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
) -> Vec<TreeItem> {
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
        let label: SharedString = entry.file_name().to_string_lossy().into_owned().into();

        if path.is_dir() {
            let children = build_dir(layout, &path, depth + 1, max_depth, expanded);
            let is_expanded = expanded.contains(&id);
            dirs.push(
                TreeItem::new(id, label)
                    .expanded(is_expanded || depth == 0)
                    .children(children),
            );
        } else {
            files.push(TreeItem::new(id, label));
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
