//! Tab completion + inline hints.

use std::path::{Path, PathBuf};

use reedline::{Completer, Hinter, History, Span, Suggestion};

use super::commands::{
    APPROVE_MODES, META_COMMANDS, REPLAY_FLAGS, TOOL_FLAGS, TRACE_SUBCMDS, slash_completion_names,
};

pub struct ForgeCompleter {
    tool_names: Vec<String>,
    cwd: PathBuf,
}

impl ForgeCompleter {
    pub fn new(tool_names: Vec<String>, cwd: PathBuf) -> Self {
        Self { tool_names, cwd }
    }

    fn word_bounds(line: &str, pos: usize) -> (usize, usize) {
        let before = &line[..pos.min(line.len())];
        let start = before
            .rfind(|c: char| c.is_whitespace())
            .map(|i| i + 1)
            .unwrap_or(0);
        (start, pos)
    }

    fn flag_suggestions(tool: &str, prefix: &str, span: Span) -> Vec<Suggestion> {
        TOOL_FLAGS
            .iter()
            .find(|(name, _)| *name == tool)
            .map(|(_, flags)| *flags)
            .unwrap_or(&[])
            .iter()
            .filter(|f| f.starts_with(prefix) || prefix.is_empty())
            .map(|f| Suggestion {
                value: format!("{f} "),
                description: Some("tool argument".into()),
                span,
                append_whitespace: false,
                ..Default::default()
            })
            .collect()
    }

    fn path_suggestions(prefix: &str, span: Span, cwd: &Path) -> Vec<Suggestion> {
        let clean = prefix.trim_start_matches("./");
        let (dir, file_prefix) = match clean.rfind('/') {
            Some(i) => (cwd.join(&clean[..i]), &clean[i + 1..]),
            None => (cwd.to_path_buf(), clean),
        };
        let read_dir = match std::fs::read_dir(dir) {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        let mut out = Vec::new();
        for entry in read_dir.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if file_prefix.is_empty() || name.starts_with(file_prefix) {
                let suffix = if entry.path().is_dir() { "/" } else { " " };
                out.push(Suggestion {
                    value: format!("{name}{suffix}"),
                    description: Some("path".into()),
                    span,
                    append_whitespace: false,
                    ..Default::default()
                });
            }
        }
        out.sort_by(|a, b| a.value.cmp(&b.value));
        out.truncate(32);
        out
    }
}

impl Completer for ForgeCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        let (start, end) = Self::word_bounds(line, pos);
        let span = Span::new(start, end);
        let word = &line[start..end];

        if let Some(needle) = line.strip_prefix('@') {
            let needle = needle.trim_start();
            return Self::path_suggestions(needle, Span::new(1, pos), &self.cwd)
                .into_iter()
                .map(|mut s| {
                    s.value = format!("@{}", s.value);
                    s.span = Span::new(0, pos);
                    s
                })
                .collect();
        }

        if let Some(needle) = line.strip_prefix('/') {
            let needle = needle.trim_start();
            return slash_completion_names()
                .iter()
                .filter(|(name, _)| name.starts_with(needle) || needle.is_empty())
                .map(|(name, desc)| Suggestion {
                    value: format!("/{name}"),
                    description: Some((*desc).into()),
                    span: Span::new(0, pos),
                    append_whitespace: true,
                    ..Default::default()
                })
                .collect();
        }

        let tokens: Vec<&str> = line[..pos].split_whitespace().collect();
        if tokens.is_empty() {
            let mut out: Vec<Suggestion> = META_COMMANDS
                .iter()
                .map(|(name, desc)| Suggestion {
                    value: (*name).into(),
                    description: Some((*desc).into()),
                    span,
                    append_whitespace: true,
                    ..Default::default()
                })
                .collect();
            for name in &self.tool_names {
                out.push(Suggestion {
                    value: name.clone(),
                    description: Some("tool".into()),
                    span,
                    append_whitespace: true,
                    ..Default::default()
                });
            }
            return out;
        }

        let head = tokens[0];
        if tokens.len() == 1 {
            let mut out: Vec<Suggestion> = META_COMMANDS
                .iter()
                .filter(|(n, _)| n.starts_with(head))
                .map(|(name, desc)| Suggestion {
                    value: (*name).into(),
                    description: Some((*desc).into()),
                    span,
                    append_whitespace: true,
                    ..Default::default()
                })
                .collect();
            for name in &self.tool_names {
                if name.starts_with(head) {
                    out.push(Suggestion {
                        value: name.clone(),
                        description: Some("tool".into()),
                        span,
                        append_whitespace: true,
                        ..Default::default()
                    });
                }
            }
            if !out.is_empty() {
                return out;
            }
        }

        if head == "approve" || head == "permissions" {
            return APPROVE_MODES
                .iter()
                .filter(|m| m.starts_with(word))
                .map(|m| Suggestion {
                    value: (*m).into(),
                    description: Some("approval mode".into()),
                    span,
                    append_whitespace: true,
                    ..Default::default()
                })
                .collect();
        }

        if head == "replay" {
            return REPLAY_FLAGS
                .iter()
                .filter(|f| f.starts_with(word))
                .map(|f| Suggestion {
                    value: (*f).into(),
                    description: Some("replay modifier".into()),
                    span,
                    append_whitespace: true,
                    ..Default::default()
                })
                .collect();
        }

        if head == "trace" && tokens.len() >= 2 {
            return TRACE_SUBCMDS
                .iter()
                .filter(|s| s.starts_with(word))
                .map(|s| Suggestion {
                    value: (*s).into(),
                    description: Some("trace subcommand".into()),
                    span,
                    append_whitespace: true,
                    ..Default::default()
                })
                .collect();
        }

        if word.starts_with("--") {
            return Self::flag_suggestions(tokens[0], word, span);
        }

        if self.tool_names.iter().any(|t| t == head) && tokens.len() >= 2 {
            let last_flag = tokens
                .iter()
                .rev()
                .find(|t| t.starts_with("--"))
                .copied()
                .unwrap_or("");
            if matches!(last_flag, "--file_path" | "--path" | "--pattern") {
                return Self::path_suggestions(word, span, &self.cwd);
            }
            if word.is_empty() || word.starts_with("--") {
                return Self::flag_suggestions(
                    head,
                    if word.starts_with("--") { word } else { "" },
                    span,
                );
            }
        }

        Vec::new()
    }
}

pub struct ForgeHinter {
    completer: ForgeCompleter,
    last_hint: String,
}

impl ForgeHinter {
    pub fn new(tool_names: Vec<String>, cwd: PathBuf) -> Self {
        Self {
            completer: ForgeCompleter::new(tool_names, cwd),
            last_hint: String::new(),
        }
    }
}

impl Hinter for ForgeHinter {
    fn handle(
        &mut self,
        line: &str,
        pos: usize,
        _history: &dyn History,
        use_ansi_coloring: bool,
        _cwd: &str,
    ) -> String {
        let hint = self
            .completer
            .complete(line, pos)
            .first()
            .map(|s| suffix_after(line, &s.value))
            .unwrap_or_default();
        self.last_hint = hint.clone();
        if hint.is_empty() {
            return String::new();
        }
        if use_ansi_coloring {
            format!("\x1b[2m{hint}\x1b[0m")
        } else {
            hint
        }
    }

    fn complete_hint(&self) -> String {
        self.last_hint.clone()
    }

    fn next_hint_token(&self) -> String {
        self.last_hint
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string()
    }
}

fn suffix_after(line: &str, completion: &str) -> String {
    completion.strip_prefix(line).unwrap_or("").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completes_slash_and_aliases() {
        let mut c = ForgeCompleter::new(vec![], PathBuf::from("."));
        assert!(c.complete("/hel", 4).iter().any(|s| s.value == "/help"));
        assert!(c.complete("/clean", 6).iter().any(|s| s.value == "/clean"));
    }

    #[test]
    fn completes_tool_flags() {
        let mut c = ForgeCompleter::new(vec!["Read".into()], PathBuf::from("."));
        let hits = c.complete("Read --file", 11);
        assert!(hits.iter().any(|s| s.value.contains("file_path")));
    }
}
