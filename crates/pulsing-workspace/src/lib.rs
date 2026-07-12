//! Workspace bootstrap and revision journal for Pulsing CLI.

mod discover;
mod ignore;
mod init;
mod journal;
mod layout;

pub use discover::{find_workspace_root, require_workspace_root};
pub use init::{init_workspace, InitOptions, InitResult, Template};
pub use journal::{
    checkpoint, current_head, list_revisions, rollback, CheckpointOptions, RevisionInfo,
    RollbackOptions,
};
pub use layout::{WorkspaceLayout, WorkspaceManifest, PULSING_DIR};
