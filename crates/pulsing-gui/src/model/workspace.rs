use std::path::PathBuf;

use pulsing_workspace::WorkspaceLayout;

use super::revisions::RevisionSnapshot;
use super::workflows;

#[derive(Clone, Debug, Default)]
pub struct RuntimeSummary {
    pub local_busy: bool,
    pub session_title: String,
    pub file_count: usize,
    pub revision_count: usize,
    pub workflow_count: usize,
}

pub struct WorkspaceModel {
    pub layout: WorkspaceLayout,
    pub runtime: RuntimeSummary,
    pub selected_file: Option<PathBuf>,
    pub revisions: RevisionSnapshot,
    pub workflow_scripts: Vec<PathBuf>,
}

impl WorkspaceModel {
    pub fn new(cwd: PathBuf) -> Self {
        let layout = WorkspaceLayout::new(cwd);
        let revisions = super::revisions::load_revisions(&layout);
        let workflow_scripts = workflows::list_workflow_scripts(&layout);
        let revision_count = revisions.revisions.len();
        let workflow_count = workflow_scripts.len();
        Self {
            layout,
            runtime: RuntimeSummary {
                revision_count,
                workflow_count,
                ..RuntimeSummary::default()
            },
            selected_file: None,
            revisions,
            workflow_scripts,
        }
    }

    pub fn set_busy(&mut self, busy: bool) {
        self.runtime.local_busy = busy;
    }

    pub fn set_session_title(&mut self, title: String) {
        self.runtime.session_title = title;
    }

    pub fn set_selected_file(&mut self, path: Option<PathBuf>) {
        self.selected_file = path;
    }

    pub fn refresh_revisions(&mut self) {
        self.revisions = super::revisions::load_revisions(&self.layout);
        self.runtime.revision_count = self.revisions.revisions.len();
    }

    pub fn refresh_workflows(&mut self) {
        self.workflow_scripts = workflows::list_workflow_scripts(&self.layout);
        self.runtime.workflow_count = self.workflow_scripts.len();
    }
}
