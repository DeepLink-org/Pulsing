use std::path::PathBuf;

use crate::model::sessions::SessionId;

/// Cross-panel user intents — panels emit these; `WorkspaceShell` dispatches them.
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
