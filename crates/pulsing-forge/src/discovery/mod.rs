//! Deferred tool catalog + BM25 search (Codex `tool_search` compatible).

mod bm25;
mod catalog;
mod codex_manifest;
mod codex_paths;
mod marketplace;
mod plugins;

pub use catalog::{
    DeferredToolEntry, TOOL_SEARCH_DEFAULT_LIMIT, TOOL_SEARCH_MAX_LIMIT, ToolCatalog,
    ToolCatalogSnapshot,
};
pub use plugins::{
    DiscoverablePlugin, PluginManifest, load_plugin_manifests, scan_codex_plugin_dirs,
};

use std::sync::{Arc, Mutex};

pub fn new_tool_catalog() -> Arc<Mutex<ToolCatalog>> {
    Arc::new(Mutex::new(ToolCatalog::default()))
}
