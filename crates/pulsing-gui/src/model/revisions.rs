use pulsing_workspace::{list_revisions, RevisionInfo, WorkspaceLayout};

#[derive(Clone, Debug, Default)]
pub struct RevisionSnapshot {
    pub head: Option<String>,
    pub revisions: Vec<RevisionInfo>,
}

pub fn load_revisions(layout: &WorkspaceLayout) -> RevisionSnapshot {
    let revisions = list_revisions(layout).unwrap_or_default();
    let head = pulsing_workspace::current_head(layout).ok().flatten();
    RevisionSnapshot { head, revisions }
}
