use std::collections::HashSet;
use std::path::PathBuf;

/// Tracks which file preview tabs are open in the center dock.
#[derive(Clone, Debug, Default)]
pub struct CenterTabState {
    pub open_files: HashSet<PathBuf>,
}

impl CenterTabState {
    pub fn register(&mut self, rel: PathBuf) {
        self.open_files.insert(rel);
    }

    pub fn unregister(&mut self, rel: &PathBuf) {
        self.open_files.remove(rel);
    }

    pub fn is_open(&self, rel: &PathBuf) -> bool {
        self.open_files.contains(rel)
    }
}
