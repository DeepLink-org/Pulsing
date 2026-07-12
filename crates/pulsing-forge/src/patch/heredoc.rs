//! Tree-sitter bash heredoc extraction for `apply_patch` shell scripts.
//! Ported from codex-apply-patch `invocation.rs` (Apache-2.0).

use std::str::Utf8Error;
use std::sync::LazyLock;

use streaming_iterator::StreamingIterator;
use tree_sitter::{LanguageError, Parser, Query, QueryCursor};
use tree_sitter_bash::LANGUAGE as BASH;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeredocError {
    CommandDidNotStartWithApplyPatch,
    FailedToLoadBashGrammar(String),
    HeredocNotUtf8,
    FailedToParsePatchIntoAst,
}

impl From<Utf8Error> for HeredocError {
    fn from(_: Utf8Error) -> Self {
        Self::HeredocNotUtf8
    }
}

static APPLY_PATCH_QUERY: LazyLock<Query> = LazyLock::new(|| {
    let language: tree_sitter::Language = BASH.into();
    Query::new(
        &language,
        r#"
        (
          program
            . (redirected_statement
                body: (command
                        name: (command_name (word) @apply_name) .)
                (#any-of? @apply_name "apply_patch" "applypatch")
                redirect: (heredoc_redirect
                            . (heredoc_start)
                            . (heredoc_body) @heredoc
                            . (heredoc_end)
                            .))
            .)

        (
          program
            . (redirected_statement
                body: (list
                        . (command
                            name: (command_name (word) @cd_name) .
                            argument: [
                              (word) @cd_path
                              (string (string_content) @cd_path)
                              (raw_string) @cd_raw_string
                            ] .)
                        "&&"
                        . (command
                            name: (command_name (word) @apply_name))
                        .)
                (#eq? @cd_name "cd")
                (#any-of? @apply_name "apply_patch" "applypatch")
                redirect: (heredoc_redirect
                            . (heredoc_start)
                            . (heredoc_body) @heredoc
                            . (heredoc_end)
                            .))
            .)
        "#,
    )
    .expect("valid bash query")
});

/// Extract heredoc patch body from a `bash -lc` script.
pub fn extract_apply_patch_from_bash(src: &str) -> Result<(String, Option<String>), HeredocError> {
    let lang: tree_sitter::Language = BASH.into();
    let mut parser = Parser::new();
    parser
        .set_language(&lang)
        .map_err(|e: LanguageError| HeredocError::FailedToLoadBashGrammar(e.to_string()))?;
    let tree = parser
        .parse(src, None)
        .ok_or(HeredocError::FailedToParsePatchIntoAst)?;

    let bytes = src.as_bytes();
    let root = tree.root_node();
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&APPLY_PATCH_QUERY, root, bytes);
    while let Some(m) = matches.next() {
        let mut heredoc_text: Option<String> = None;
        let mut cd_path: Option<String> = None;

        for capture in m.captures.iter() {
            let name = APPLY_PATCH_QUERY.capture_names()[capture.index as usize];
            match name {
                "heredoc" => {
                    let text = capture
                        .node
                        .utf8_text(bytes)
                        .map_err(|_| HeredocError::HeredocNotUtf8)?
                        .trim_end_matches('\n')
                        .to_string();
                    heredoc_text = Some(text);
                }
                "cd_path" => {
                    cd_path = Some(
                        capture
                            .node
                            .utf8_text(bytes)
                            .map_err(|_| HeredocError::HeredocNotUtf8)?
                            .to_string(),
                    );
                }
                "cd_raw_string" => {
                    let raw = capture
                        .node
                        .utf8_text(bytes)
                        .map_err(|_| HeredocError::HeredocNotUtf8)?;
                    let trimmed = raw
                        .strip_prefix('\'')
                        .and_then(|s| s.strip_suffix('\''))
                        .unwrap_or(raw);
                    cd_path = Some(trimmed.to_string());
                }
                _ => {}
            }
        }

        if let Some(heredoc) = heredoc_text {
            return Ok((heredoc, cd_path));
        }
    }

    Err(HeredocError::CommandDidNotStartWithApplyPatch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_apply_patch_heredoc() {
        let script =
            "apply_patch <<'PATCH'\n*** Begin Patch\n*** Add File: foo\n+hi\n*** End Patch\nPATCH";
        let (body, wd) = extract_apply_patch_from_bash(script).unwrap();
        assert!(wd.is_none());
        assert!(body.contains("*** Add File: foo"));
    }

    #[test]
    fn extracts_cd_and_apply_patch_heredoc() {
        let script = "cd subdir && apply_patch <<'PATCH'\n*** Begin Patch\n*** End Patch\nPATCH";
        let (body, wd) = extract_apply_patch_from_bash(script).unwrap();
        assert_eq!(wd.as_deref(), Some("subdir"));
        assert!(body.contains("Begin Patch"));
    }

    #[test]
    fn rejects_unrelated_shell() {
        let script = "echo hello && apply_patch foo";
        assert_eq!(
            extract_apply_patch_from_bash(script),
            Err(HeredocError::CommandDidNotStartWithApplyPatch)
        );
    }
}
