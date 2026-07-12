//! Forge session REPL.

use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;

use crate::context::{StepStatus, ToolSession};
use crate::result::ToolResult;
use crate::runtime::{ToolRuntime, ToolRuntimeConfig};
use crate::unified_exec::UnifiedExecManager;
use anyhow::{Result, anyhow};
use comfy_table::{Table, presets::UTF8_FULL};
use reedline::{
    ColumnarMenu, DefaultPrompt, DefaultPromptSegment, EditCommand, Emacs, FileBackedHistory,
    KeyCode, KeyModifiers, MenuBuilder, Reedline, ReedlineEvent, ReedlineMenu, Signal,
    default_emacs_keybindings,
};
use serde_json::{Value, json};

use super::commands::{HELP, ParsedInput, SlashAction, parse_line, print_slash_menu};
use super::completer::{ForgeCompleter, ForgeHinter};
use super::parse::{parse_tool_args, parse_tool_invocation};
use super::session::ReplToolSession;
use super::trace::{self, TraceRecord};

pub struct ReplConfig {
    pub cwd: PathBuf,
    pub sandbox: String,
    pub dangerously_disable_sandbox: bool,
    pub approve_auto: bool,
    pub trace_path: Option<PathBuf>,
    pub record_path: Option<PathBuf>,
}

pub struct ForgeRepl {
    session: Arc<ReplToolSession>,
    runtime: ToolRuntime,
    exec: Arc<UnifiedExecManager>,
    trace: Vec<TraceRecord>,
    trace_path: Option<PathBuf>,
    replay_index: usize,
    record: Vec<TraceRecord>,
    record_path: Option<PathBuf>,
    next_seq: u64,
    cwd: PathBuf,
    last_result: Option<ToolResult>,
}

impl ForgeRepl {
    pub fn new(cfg: ReplConfig) -> Result<Self> {
        let session = ReplToolSession::new(cfg.approve_auto);
        let cwd = cfg.cwd.clone();
        let exec = Arc::new(UnifiedExecManager::new());
        let runtime = ToolRuntime::new(ToolRuntimeConfig {
            cwd: cfg.cwd,
            sandbox_policy: cfg.sandbox,
            dangerously_disable_sandbox: cfg.dangerously_disable_sandbox,
            session: session.clone(),
            exec: exec.clone(),
            ..Default::default()
        });
        let trace = cfg
            .trace_path
            .as_ref()
            .map(|p| trace::load_trace(p))
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            session,
            runtime,
            exec,
            trace,
            trace_path: cfg.trace_path,
            replay_index: 0,
            record: Vec::new(),
            record_path: cfg.record_path,
            next_seq: 1,
            cwd,
            last_result: None,
        })
    }

    fn prompt(&self) -> DefaultPrompt {
        let cwd = self.cwd.file_name().and_then(|s| s.to_str()).unwrap_or(".");
        let mode = if self.session.approve_auto() {
            "A"
        } else {
            "K"
        };
        let replay = if trace::tool_calls(&self.trace).is_empty() {
            String::new()
        } else {
            format!(
                " {}/{}",
                self.replay_index,
                trace::tool_calls(&self.trace).len()
            )
        };
        DefaultPrompt {
            left_prompt: DefaultPromptSegment::Basic(format!("forge ⟨{cwd}⟩ {mode}{replay}⟩ ")),
            ..Default::default()
        }
    }

    fn build_line_editor(&self) -> Result<Reedline> {
        let tool_names = self.runtime.tool_names();
        let completer = Box::new(ForgeCompleter::new(tool_names.clone(), self.cwd.clone()));
        let hinter = Box::new(ForgeHinter::new(tool_names, self.cwd.clone()));
        let completion_menu = Box::new(
            ColumnarMenu::default()
                .with_name("completion_menu")
                .with_columns(3),
        );
        let mut keybindings = default_emacs_keybindings();
        keybindings.add_binding(
            KeyModifiers::NONE,
            KeyCode::Tab,
            ReedlineEvent::UntilFound(vec![
                ReedlineEvent::Menu("completion_menu".to_string()),
                ReedlineEvent::MenuNext,
            ]),
        );
        keybindings.add_binding(
            KeyModifiers::ALT,
            KeyCode::Enter,
            ReedlineEvent::Edit(vec![EditCommand::InsertNewline]),
        );
        let history = Box::new(
            FileBackedHistory::with_file(1000, ".pforge_history".into())
                .map_err(|e| anyhow!("history: {e}"))?,
        );
        Ok(Reedline::create()
            .with_history(history)
            .with_completer(completer)
            .with_hinter(hinter)
            .with_menu(ReedlineMenu::EngineCompleter(completion_menu))
            .with_edit_mode(Box::new(Emacs::new(keybindings))))
    }

    pub fn run_interactive(&mut self) -> Result<()> {
        let mut line_editor = self.build_line_editor()?;
        println!("{HELP}");
        if !self.trace.is_empty() {
            println!(
                "loaded trace: {} tool calls (/replay or replay all)",
                trace::tool_calls(&self.trace).len()
            );
        }
        loop {
            match line_editor.read_line(&self.prompt()) {
                Ok(Signal::Success(line)) => match self.dispatch(&line) {
                    Ok(true) => {}
                    Ok(false) => break,
                    Err(e) => eprintln!("error: {e:#}"),
                },
                Ok(Signal::CtrlC) | Ok(Signal::CtrlD) => {
                    println!();
                    break;
                }
                Err(e) => return Err(anyhow!("readline: {e}")),
            }
        }
        Ok(())
    }

    pub async fn replay_all(&mut self, dry_run: bool, verify: bool) -> Result<Vec<String>> {
        let mut lines = Vec::new();
        loop {
            let msg = self.replay_step(dry_run, verify).await?;
            let done = msg.contains("complete") || msg.starts_with("no trace");
            lines.push(msg);
            if done {
                break;
            }
        }
        Ok(lines)
    }

    fn dispatch(&mut self, line: &str) -> Result<bool> {
        match parse_line(line) {
            ParsedInput::Empty => Ok(true),
            ParsedInput::SlashMenu => {
                print_slash_menu();
                Ok(true)
            }
            ParsedInput::Slash { action, args } => self.dispatch_slash(action, &args),
            ParsedInput::Mention(path) => {
                self.exec_call("Read", json!({ "file_path": path }))?;
                Ok(true)
            }
            ParsedInput::Meta(cmd) => self.dispatch_meta(&cmd),
            ParsedInput::Bare(line) => self.dispatch_bare(&line),
        }
    }

    fn dispatch_slash(&mut self, action: SlashAction, args: &str) -> Result<bool> {
        match action {
            SlashAction::Meta(cmd) => self.dispatch_meta(&format!("{cmd} {args}").trim()),
            SlashAction::Clear => {
                print!("\x1b[2J\x1b[H");
                Ok(true)
            }
            SlashAction::Diff | SlashAction::Review => {
                self.exec_call(
                    "shell_command",
                    json!({ "command": "git diff && git diff --cached && git status -sb" }),
                )?;
                Ok(true)
            }
            SlashAction::Ps => {
                let sessions = run_async(self.exec.list_sessions())?;
                if sessions.is_empty() {
                    println!("(no background exec sessions)");
                } else {
                    let mut table = Table::new();
                    table.load_preset(UTF8_FULL);
                    table.set_header(["id", "elapsed_s", "tty"]);
                    for s in sessions {
                        table.add_row([
                            s.id.to_string(),
                            format!("{:.1}", s.elapsed_secs),
                            s.tty.to_string(),
                        ]);
                    }
                    println!("{table}");
                }
                Ok(true)
            }
            SlashAction::Stop => {
                let n = run_async(self.exec.stop_all())?;
                println!("stopped {n} exec session(s)");
                Ok(true)
            }
            SlashAction::Fork => {
                let step: usize = args.trim().parse().map_err(|_| anyhow!("usage: /fork N"))?;
                self.fork_trace(step)?;
                println!(
                    "forked trace at step {step}; replay_index={}",
                    self.replay_index
                );
                Ok(true)
            }
            SlashAction::Compact => {
                self.exec_call("new_context", json!({}))?;
                Ok(true)
            }
            SlashAction::Mcp => {
                let mcp: Vec<_> = self
                    .runtime
                    .tool_names()
                    .into_iter()
                    .filter(|n| n.contains("mcp"))
                    .collect();
                if mcp.is_empty() {
                    println!("(no MCP tools registered)");
                } else {
                    for n in mcp {
                        println!("  {n}");
                    }
                }
                Ok(true)
            }
            SlashAction::Plugins => {
                self.exec_call("list_available_plugins_to_install", json!({}))?;
                Ok(true)
            }
            SlashAction::Mention => {
                let path = args.trim();
                if path.is_empty() {
                    return Err(anyhow!("usage: /mention PATH"));
                }
                self.exec_call("Read", json!({ "file_path": path }))?;
                Ok(true)
            }
            SlashAction::Rollout => {
                if let Some(p) = &self.trace_path {
                    println!("trace: {}", p.display());
                }
                if let Some(p) = &self.record_path {
                    println!("record: {} ({} lines)", p.display(), self.record.len());
                }
                if self.trace_path.is_none() && self.record_path.is_none() {
                    println!("(no --trace or --record path)");
                }
                Ok(true)
            }
            SlashAction::Copy => {
                match &self.last_result {
                    Some(r) => println!("{}", r.content),
                    None => println!("(no tool result yet)"),
                }
                Ok(true)
            }
        }
    }

    fn dispatch_meta(&mut self, line: &str) -> Result<bool> {
        let mut parts = line.split_whitespace();
        let cmd = parts.next().unwrap_or("help");
        match cmd {
            "help" | "?" => println!("{HELP}"),
            "quit" | "exit" => return Ok(false),
            "tools" => println!("{}", self.format_tools_table()),
            "session" | "status" => println!("{}", self.format_session_table()),
            "plan" => println!("{}", self.format_plan_table()),
            "events" => println!("{}", self.format_events_table()),
            "approve" | "permissions" => self.set_approval(parts.next()),
            "replay" => self.run_replay(&parts.collect::<Vec<_>>())?,
            "trace" => self.run_trace(&mut parts)?,
            "fork" => {
                let step: usize = parts
                    .next()
                    .ok_or_else(|| anyhow!("usage: fork N"))?
                    .parse()
                    .map_err(|_| anyhow!("fork step must be integer"))?;
                self.fork_trace(step)?;
                println!("forked at step {step}");
            }
            "call" => {
                let tool = parts
                    .next()
                    .ok_or_else(|| anyhow!("usage: call TOOL {{json}}"))?;
                let args = parse_tool_args(parts.collect::<Vec<_>>().join(" ").trim())?;
                self.exec_call(tool, args)?;
            }
            other => return Err(anyhow!("unknown command: {other} (try /help)")),
        }
        Ok(true)
    }

    fn dispatch_bare(&mut self, line: &str) -> Result<bool> {
        let lower = line.to_lowercase();
        if matches!(lower.as_str(), "help" | "?") {
            println!("{HELP}");
            return Ok(true);
        }
        if matches!(lower.as_str(), "quit" | "exit") {
            return Ok(false);
        }
        for prefix in [
            "tools",
            "session",
            "plan",
            "events",
            "replay",
            "approve",
            "permissions",
            "trace",
            "fork",
        ] {
            if lower == prefix || lower.starts_with(&format!("{prefix} ")) {
                return self.dispatch_meta(line);
            }
        }
        if let Some(rest) = line.strip_prefix("call ") {
            let (tool, args) = parse_tool_invocation(rest.trim())?;
            self.exec_call(&tool, args)?;
            return Ok(true);
        }
        if let Some((tool, rest)) = line.split_once(' ') {
            if self.runtime.tool_names().iter().any(|n| n == tool) {
                self.exec_call(tool, parse_tool_args(rest)?)?;
                return Ok(true);
            }
        }
        if self.runtime.tool_names().iter().any(|n| n == line) {
            self.exec_call(line, json!({}))?;
            return Ok(true);
        }
        Err(anyhow!("unknown: {line:?} (try /help)"))
    }

    fn set_approval(&self, mode: Option<&str>) {
        match mode {
            Some(m) => {
                let auto = m.eq_ignore_ascii_case("auto");
                self.session.set_approve_auto(auto);
                println!("approval → {}", if auto { "auto" } else { "ask" });
            }
            None => println!(
                "approval: {}",
                if self.session.approve_auto() {
                    "auto"
                } else {
                    "ask"
                }
            ),
        }
    }

    fn run_replay(&mut self, rest: &[&str]) -> Result<()> {
        let dry = rest.iter().any(|p| *p == "dry");
        let verify = rest.iter().any(|p| *p == "verify");
        if rest.iter().any(|p| *p == "all") {
            for msg in run_async(self.replay_all(dry, verify))?? {
                println!("{msg}");
            }
        } else {
            println!("{}", run_async(self.replay_step(dry, verify))??);
        }
        Ok(())
    }

    fn run_trace(&mut self, parts: &mut std::str::SplitWhitespace<'_>) -> Result<()> {
        match parts.next() {
            Some("show") => {
                if let Some(path) = &self.record_path {
                    println!(
                        "recording → {} ({} lines)",
                        path.display(),
                        self.record.len()
                    );
                } else {
                    println!("(not recording; use --record or trace save PATH)");
                }
            }
            Some("save") => {
                let path = PathBuf::from(
                    parts
                        .next()
                        .ok_or_else(|| anyhow!("usage: trace save PATH"))?,
                );
                trace::save_trace(&path, &self.record)?;
                self.record_path = Some(path.clone());
                println!("saved {} records → {}", self.record.len(), path.display());
            }
            _ => println!("usage: trace save PATH | trace show"),
        }
        Ok(())
    }

    fn fork_trace(&mut self, step: usize) -> Result<()> {
        if self.trace.is_empty() {
            return Err(anyhow!("no trace loaded"));
        }
        self.replay_index = 0;
        let records: Vec<_> = self.trace.clone();
        for rec in &records {
            if rec.seq > step as u64 {
                break;
            }
            if rec.kind == "session" {
                if let Some(snap) = &rec.session {
                    self.apply_session(snap);
                }
            } else if rec.kind == "tool_call" && rec.seq <= step as u64 {
                if let Some(tool) = &rec.tool {
                    if self.runtime.tool_names().iter().any(|n| n == tool) {
                        self.exec_call(tool, rec.arguments.clone().unwrap_or(json!({})))?;
                    }
                }
                self.replay_index += 1;
            }
        }
        Ok(())
    }

    fn apply_session(&self, snap: &Value) {
        use crate::context::{PlanItem, UpdatePlanArgs};
        if let Some(plan_raw) = snap.get("plan").and_then(|v| v.as_array()) {
            let items: Vec<PlanItem> = plan_raw
                .iter()
                .filter_map(|p| {
                    Some(PlanItem {
                        step: p.get("step")?.as_str()?.to_string(),
                        status: match p.get("status")?.as_str()? {
                            "in_progress" => StepStatus::InProgress,
                            "completed" => StepStatus::Completed,
                            _ => StepStatus::Pending,
                        },
                    })
                })
                .collect();
            if !items.is_empty() {
                let _ = self.session.update_plan(UpdatePlanArgs {
                    explanation: None,
                    plan: items,
                });
            }
        }
    }

    fn exec_call(&mut self, tool: &str, args: Value) -> Result<()> {
        let out = run_async(self.runtime.call_tool(tool, args.clone()))?;
        self.last_result = Some(out.clone());
        if let Some(path) = &self.record_path {
            self.record.push(TraceRecord {
                seq: self.next_seq,
                kind: "tool_call".into(),
                tool: Some(tool.into()),
                arguments: Some(args),
                result: Some(json!({
                    "content": out.content,
                    "is_error": out.is_error,
                    "structured": out.structured,
                })),
                event: None,
                session: None,
            });
            self.next_seq += 1;
            trace::save_trace(path, &self.record)?;
        }
        let payload = json!({
            "tool": tool,
            "is_error": out.is_error,
            "content": out.content.chars().take(2000).collect::<String>(),
            "structured": out.structured,
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        Ok(())
    }

    async fn replay_step(&mut self, dry_run: bool, verify: bool) -> Result<String> {
        let calls = trace::tool_calls(&self.trace);
        if calls.is_empty() {
            return Ok("no trace loaded".into());
        }
        if self.replay_index >= calls.len() {
            return Ok("replay complete".into());
        }
        let rec = calls[self.replay_index];
        self.replay_index += 1;
        let tool = rec.tool.as_deref().unwrap_or("?");
        let args = rec.arguments.clone().unwrap_or(json!({}));
        if dry_run {
            return Ok(format!(
                "dry-run #{} call {} {}",
                rec.seq,
                tool,
                serde_json::to_string(&args)?
            ));
        }
        let out = self.runtime.call_tool(tool, args).await;
        self.last_result = Some(out.clone());
        if verify {
            if let Some(exp) = &rec.result {
                let exp_err = exp
                    .get("is_error")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if out.is_error != exp_err {
                    return Ok(format!(
                        "verify FAIL #{} {tool}: is_error expected {exp_err} got {}",
                        rec.seq, out.is_error
                    ));
                }
            }
        }
        let flag = if out.is_error { "ERR" } else { "ok" };
        let preview: String = out.content.chars().take(400).collect();
        Ok(format!("replay #{} {tool} [{flag}] {preview}", rec.seq))
    }

    fn format_tools_table(&self) -> String {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_header(["tool"]);
        for name in self.runtime.tool_names() {
            table.add_row([name]);
        }
        table.to_string()
    }

    fn format_plan_table(&self) -> String {
        let plan = self.session.plan_snapshot();
        if plan.is_empty() {
            return "(empty plan)".into();
        }
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_header(["step", "status"]);
        for item in plan {
            let status = match item.status {
                StepStatus::Pending => "pending",
                StepStatus::InProgress => "in_progress",
                StepStatus::Completed => "completed",
            };
            table.add_row([item.step, status.to_string()]);
        }
        table.to_string()
    }

    fn format_events_table(&self) -> String {
        let deltas = self.session.exec_deltas();
        if deltas.is_empty() {
            return "(no exec output deltas yet)".into();
        }
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_header(["kind", "preview"]);
        for d in deltas.iter().rev().take(12).rev() {
            let preview: String = format!("{d:?}").chars().take(80).collect();
            table.add_row(["exec_delta".to_string(), preview]);
        }
        table.to_string()
    }

    fn format_session_table(&self) -> String {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_header(["field", "value"]);
        table.add_row(["cwd".to_string(), self.cwd.display().to_string()]);
        table.add_row([
            "approval".to_string(),
            if self.session.approve_auto() {
                "auto".to_string()
            } else {
                "ask".to_string()
            },
        ]);
        table.add_row([
            "plan".to_string(),
            self.session.plan_snapshot().len().to_string(),
        ]);
        table.add_row([
            "new_context".to_string(),
            self.session.new_context_requested().to_string(),
        ]);
        if let Some(tokens) = self.session.tokens_remaining() {
            table.add_row(["tokens_remaining".to_string(), tokens.to_string()]);
        }
        table.add_row([
            "replay".to_string(),
            format!(
                "{}/{}",
                self.replay_index,
                trace::tool_calls(&self.trace).len()
            ),
        ]);
        table.to_string()
    }
}

fn run_async<F, T>(fut: F) -> Result<T>
where
    F: Future<Output = T>,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => Ok(tokio::task::block_in_place(|| handle.block_on(fut))),
        Err(_) => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| anyhow!("runtime: {e}"))?;
            Ok(rt.block_on(fut))
        }
    }
}
