use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use chrono::Utc;
use serde_json::json;

use crate::layout::{cluster_id_for, WorkspaceLayout, WorkspaceManifest};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    Minimal,
    Agent,
}

impl Template {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "minimal" => Ok(Self::Minimal),
            "agent" => Ok(Self::Agent),
            _ => bail!("unknown template {s:?} (expected minimal or agent)"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Agent => "agent",
        }
    }
}

#[derive(Debug, Clone)]
pub struct InitOptions {
    pub template: Template,
    pub name: Option<String>,
    pub force: bool,
    /// Natural-language goal; stored in workspace.json and used for LLM-guided bootstrap.
    pub guide: Option<String>,
}

#[derive(Debug, Clone)]
pub struct InitResult {
    pub root: std::path::PathBuf,
    pub created: bool,
    pub cluster_id: String,
}

pub fn init_workspace(root: &Path, opts: InitOptions) -> Result<InitResult> {
    let root = root.canonicalize().context("invalid workspace path")?;
    let layout = WorkspaceLayout::new(&root);

    if layout.is_initialized() && !opts.force {
        return Ok(InitResult {
            root: root.clone(),
            created: false,
            cluster_id: read_cluster_id(&layout)?,
        });
    }

    let cluster_id = cluster_id_for(&root);
    let name = opts.name.clone().unwrap_or_else(|| {
        root.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("workspace")
            .to_string()
    });

    fs::create_dir_all(layout.pulsing_dir())?;
    fs::create_dir_all(layout.hooks_dir())?;
    fs::create_dir_all(layout.scripts_dir())?;
    fs::create_dir_all(layout.workflows_dir())?;
    fs::create_dir_all(layout.revisions_dir())?;

    let now = Utc::now().to_rfc3339();
    let manifest = WorkspaceManifest {
        version: 1,
        template: opts.template.as_str().to_string(),
        name: name.clone(),
        cluster_id: cluster_id.clone(),
        created_at: now.clone(),
        init_guide: opts.guide.clone(),
    };
    fs::write(
        layout.workspace_file(),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;

    let cluster = match opts.template {
        Template::Minimal => minimal_cluster_json(&cluster_id, &name),
        Template::Agent => agent_cluster_json(&cluster_id, &name),
    };
    fs::write(
        layout.cluster_file(),
        serde_json::to_string_pretty(&cluster)? + "\n",
    )?;

    write_hook_stubs(&layout)?;
    write_scripts_readme(&layout)?;
    write_workflows_scaffold(&layout)?;

    // Initial checkpoint
    crate::journal::checkpoint(
        &layout,
        crate::journal::CheckpointOptions {
            message: Some("workspace init".into()),
            author: Some("pulsing".into()),
        },
    )?;

    Ok(InitResult {
        root,
        created: true,
        cluster_id,
    })
}

fn read_cluster_id(layout: &WorkspaceLayout) -> Result<String> {
    let data: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(layout.cluster_file())?)?;
    Ok(data
        .get("cluster_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string())
}

fn minimal_cluster_json(cluster_id: &str, name: &str) -> serde_json::Value {
    json!({
        "cluster_id": cluster_id,
        "name": name,
        "provider": "anthropic",
        "model": null,
        "auto_approve": false,
        "sandbox": "off",
        "default_agents": [],
        "shared_tool_worker": false,
        "puzzles": {}
    })
}

fn agent_cluster_json(cluster_id: &str, name: &str) -> serde_json::Value {
    json!({
        "cluster_id": cluster_id,
        "name": name,
        "provider": "anthropic",
        "model": null,
        "auto_approve": false,
        "sandbox": "off",
        "default_agents": ["guide"],
        "shared_tool_worker": false,
        "puzzles": {
            "unit-tests": {
                "title": "Unit test suite",
                "kind": "test",
                "path": "tests",
                "blurb": "Run pytest; keep green.",
                "status": "open",
                "assign_to": ""
            }
        }
    })
}

fn write_hook_stubs(layout: &WorkspaceLayout) -> Result<()> {
    let on_init = layout.hooks_dir().join("on_init.py");
    if !on_init.exists() {
        fs::write(
            &on_init,
            r#"""Pulsing workspace hook — called after `pulsing init`."""

from __future__ import annotations

from typing import Any


def on_init(ctx: dict[str, Any]) -> None:
    """Customize a new workspace. *ctx* has keys: root, cluster_id, template."""
    _ = ctx
"#,
        )?;
    }

    let on_checkpoint = layout.hooks_dir().join("on_checkpoint.py");
    if !on_checkpoint.exists() {
        fs::write(
            &on_checkpoint,
            r#"""Pulsing workspace hook — called before each checkpoint."""

from __future__ import annotations

from typing import Any


def before_checkpoint(ctx: dict[str, Any]) -> list[str] | None:
    """Return extra relative paths to include, or None for default scan."""
    _ = ctx
    return None


def after_checkpoint(ctx: dict[str, Any]) -> None:
    _ = ctx
"#,
        )?;
    }
    Ok(())
}

fn write_scripts_readme(layout: &WorkspaceLayout) -> Result<()> {
    let readme = layout.scripts_dir().join("README.md");
    if !readme.exists() {
        fs::write(
            &readme,
            "# Pulsing workspace scripts\n\n\
             Prefer workflows under `.pulsing/workflows/` (see `example.py`).\n\n\
             Legacy location for ad-hoc scripts:\n\n\
             ```bash\n\
             pulsing run scripts/your_flow.py\n\
             ```\n",
        )?;
    }
    Ok(())
}

const WORKFLOWS_README: &str = "# Pulsing workflows\n\n\
Code-as-workflow: plain Python, no graph DSL.\n\n\
```bash\n\
pulsing run                  # immersive session (default: example.py)\n\
pulsing run my_workflow.py   # explicit script\n\
```\n\n\
After success you stay in the CLI (`›` prompt). Type `exit` to return to the shell.\n\
On failure you return to the shell — then `pulsing rollback` or safe-mode fix.\n";

const WORKFLOW_EXAMPLE: &str = r#"""Example workflow — extension mode.

Run: pulsing run
"""

from __future__ import annotations

from pulsing.workflow import WorkflowContext, run


async def main(ctx: WorkflowContext) -> None:
    ctx.info(f"workspace: {ctx.root}")
    rev = ctx.checkpoint("example workflow")
    ctx.info(f"checkpoint {rev}")


if __name__ == "__main__":
    run(main)
"#;

fn write_workflows_scaffold(layout: &WorkspaceLayout) -> Result<()> {
    let readme = layout.workflows_dir().join("README.md");
    if !readme.exists() {
        fs::write(&readme, WORKFLOWS_README)?;
    }
    let example = layout.workflows_dir().join("example.py");
    if !example.exists() {
        fs::write(&example, WORKFLOW_EXAMPLE)?;
    }
    Ok(())
}
