use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub const PULSING_DIR: &str = ".pulsing";
pub const WORKSPACE_FILE: &str = "workspace.json";
pub const CLUSTER_FILE: &str = "cluster.json";
pub const HISTORY_DIR: &str = "history";
pub const REVISIONS_DIR: &str = "revisions";
pub const HEAD_FILE: &str = "HEAD";
pub const HOOKS_DIR: &str = "hooks";
pub const SCRIPTS_DIR: &str = "scripts";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceManifest {
    pub version: u32,
    pub template: String,
    pub name: String,
    pub cluster_id: String,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct WorkspaceLayout {
    pub root: PathBuf,
}

impl WorkspaceLayout {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            root: root.canonicalize().unwrap_or(root),
        }
    }

    pub fn pulsing_dir(&self) -> PathBuf {
        self.root.join(PULSING_DIR)
    }

    pub fn workspace_file(&self) -> PathBuf {
        self.pulsing_dir().join(WORKSPACE_FILE)
    }

    pub fn cluster_file(&self) -> PathBuf {
        self.pulsing_dir().join(CLUSTER_FILE)
    }

    pub fn hooks_dir(&self) -> PathBuf {
        self.pulsing_dir().join(HOOKS_DIR)
    }

    pub fn scripts_dir(&self) -> PathBuf {
        self.pulsing_dir().join(SCRIPTS_DIR)
    }

    pub fn history_dir(&self) -> PathBuf {
        self.pulsing_dir().join(HISTORY_DIR)
    }

    pub fn revisions_dir(&self) -> PathBuf {
        self.history_dir().join(REVISIONS_DIR)
    }

    pub fn head_file(&self) -> PathBuf {
        self.history_dir().join(HEAD_FILE)
    }

    pub fn revision_dir(&self, id: &str) -> PathBuf {
        self.revisions_dir().join(id)
    }

    pub fn is_initialized(&self) -> bool {
        self.cluster_file().is_file()
    }

    pub fn rel_to_root(&self, path: &Path) -> Option<PathBuf> {
        path.strip_prefix(&self.root).ok().map(|p| p.to_path_buf())
    }
}

pub fn cluster_id_for(root: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(
        root.canonicalize()
            .unwrap_or_else(|_| root.to_path_buf())
            .to_string_lossy()
            .as_bytes(),
    );
    format!("{:x}", hasher.finalize())[..12].to_string()
}
