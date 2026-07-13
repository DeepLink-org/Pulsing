use std::path::PathBuf;

use crate::model::sessions::SessionId;

/// UI actions dispatched by `WorkspaceApp`.
#[derive(Clone, Debug)]
pub enum WorkspaceAction {
    OpenFile(PathBuf),
    CloseFile(PathBuf),
    RefreshExplorer,
    NewSession,
    FocusSession(SessionId),
    RefreshRevisions,
    RefreshWorkflows,
}
