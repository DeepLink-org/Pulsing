# SPDX-License-Identifier: Apache-2.0
"""Tests for Forge memories backend (Codex ext/memories alignment)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from pulsing.forge.context import ToolCallContext
from pulsing.forge.extension.memories.backend import (
    AD_HOC_NOTE_MAX_BYTES,
    MAX_LIST_RESULTS,
    MAX_SEARCH_CONTEXT_LINES,
    MAX_SEARCH_MATCH_WINDOW_LINES,
    MAX_SEARCH_QUERIES,
    MemoriesBackendError,
    SearchMatchMode,
    SearchMatchModeKind,
)
from pulsing.forge.extension.memories.local_backend import (
    LocalMemoriesStore,
    clamp_max_results,
    default_ad_hoc_filename,
)
from pulsing.forge.extension.memories.path_utils import (
    MAX_READ_FILE_BYTES,
    validate_read_path,
)
from pulsing.forge.handlers import dispatch_tool


def _store(tmp_path: Path) -> LocalMemoriesStore:
    root = tmp_path / "memories"
    root.mkdir()
    return LocalMemoriesStore(root)


def test_list_immediate_children_and_cursor(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.root / "a.md").write_text("a", encoding="utf-8")
    sub = store.root / "nested"
    sub.mkdir()
    (sub / "b.md").write_text("b", encoding="utf-8")

    page1 = store.list_memories(max_results=1)
    assert len(page1.entries) == 1
    assert page1.truncated
    assert page1.next_cursor == "1"

    page2 = store.list_memories(cursor=page1.next_cursor, max_results=1)
    assert len(page2.entries) == 1
    assert page2.entries[0].entry_type.value == "directory"


def test_list_nested_path(tmp_path: Path) -> None:
    store = _store(tmp_path)
    nested = store.root / "nested"
    nested.mkdir()
    (nested / "note.md").write_text("hello", encoding="utf-8")

    out = store.list_memories(path="nested")
    assert len(out.entries) == 1
    assert out.entries[0].path == "nested/note.md"


def test_list_skips_hidden_entries(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.root / ".hidden.md").write_text("x", encoding="utf-8")
    (store.root / "visible.md").write_text("y", encoding="utf-8")

    out = store.list_memories()
    paths = [e.path for e in out.entries]
    assert paths == ["visible.md"]


def test_list_rejects_path_traversal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(MemoriesBackendError, match="must stay within"):
        store.list_memories(path="../outside")


def test_list_rejects_hidden_path_component(tmp_path: Path) -> None:
    store = _store(tmp_path)
    hidden = store.root / ".secret"
    hidden.mkdir()
    (hidden / "note.md").write_text("x", encoding="utf-8")

    with pytest.raises(MemoriesBackendError, match="not found"):
        store.list_memories(path=".secret")


def test_list_rejects_symlink_path(tmp_path: Path) -> None:
    store = _store(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "note.md").write_text("x", encoding="utf-8")
    (store.root / "link").symlink_to(outside)

    with pytest.raises(MemoriesBackendError, match="symlink|must stay within"):
        store.list_memories(path="link")


def test_list_not_found(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(MemoriesBackendError, match="not found"):
        store.list_memories(path="missing-dir")


def test_list_single_file(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.root / "solo.md").write_text("solo", encoding="utf-8")

    out = store.list_memories(path="solo.md")
    assert len(out.entries) == 1
    assert out.entries[0].path == "solo.md"
    assert out.entries[0].entry_type.value == "file"
    assert out.next_cursor is None
    assert not out.truncated


def test_list_invalid_cursor(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.root / "a.md").write_text("a", encoding="utf-8")

    with pytest.raises(MemoriesBackendError, match="must be a non-negative integer"):
        store.list_memories(cursor="bad")

    with pytest.raises(MemoriesBackendError, match="exceeds result count"):
        store.list_memories(cursor="99")


def test_dispatch_list_path_traversal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FORGE_MEMORIES_ROOT", str(tmp_path / "memories"))
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    ctx = ToolCallContext(cwd=tmp_path)

    result = dispatch_tool("memories.list", {"path": "../outside.txt"}, ctx=ctx)
    assert result.is_error
    assert "must stay within" in result.content

    result = dispatch_tool("memories.list", {"path": str(outside)}, ctx=ctx)
    assert result.is_error
    assert "must stay within" in result.content


def test_dispatch_list_rejects_null_byte(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FORGE_MEMORIES_ROOT", str(tmp_path / "memories"))
    ctx = ToolCallContext(cwd=tmp_path)

    result = dispatch_tool("memories.list", {"path": "foo\x00bar"}, ctx=ctx)
    assert result.is_error
    assert "must stay within" in result.content


def test_list_max_results_capped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for i in range(5):
        (store.root / f"{i:02d}.md").write_text("x", encoding="utf-8")

    page = store.list_memories(max_results=2)
    assert len(page.entries) == 2
    assert page.truncated
    assert page.next_cursor == "2"

    assert (
        clamp_max_results(MAX_LIST_RESULTS + 500, 2000, MAX_LIST_RESULTS)
        == MAX_LIST_RESULTS
    )


def test_memories_root_falls_back_to_codex_home(tmp_path: Path, monkeypatch) -> None:
    codex = tmp_path / "codex"
    monkeypatch.delenv("FORGE_MEMORIES_ROOT", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(codex))

    store = LocalMemoriesStore()
    assert store.root == (codex / "memories").resolve()
    assert store.root.is_dir()


def test_read_one_indexed_and_max_lines(tmp_path: Path) -> None:
    store = _store(tmp_path)
    rel = "doc.md"
    (store.root / rel).write_text("line1\nline2\nline3\n", encoding="utf-8")

    out = store.read_memory(path=rel, line_offset=2, max_lines=1)
    assert out.start_line_number == 2
    assert out.content == "line2\n"
    assert out.truncated  # more lines remain in file

    with pytest.raises(MemoriesBackendError):
        store.read_memory(path=rel, line_offset=0)


def test_read_rejects_invalid_line_offset(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.root / "doc.md").write_text("line1\n", encoding="utf-8")

    for bad in [-1, 0]:
        with pytest.raises(MemoriesBackendError, match="1-indexed line number"):
            store.read_memory(path="doc.md", line_offset=bad)


def test_read_rejects_invalid_max_lines(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.root / "doc.md").write_text("line1\n", encoding="utf-8")

    for bad in [-1, 0]:
        with pytest.raises(MemoriesBackendError, match="positive integer"):
            store.read_memory(path="doc.md", max_lines=bad)


def test_read_not_found_and_not_file(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.root / "dir").mkdir()

    with pytest.raises(MemoriesBackendError, match="was not found"):
        store.read_memory(path="missing.md")

    with pytest.raises(MemoriesBackendError, match="is not a file"):
        store.read_memory(path="dir")


def test_read_rejects_hidden_path(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.root / ".secret.md").write_text("hidden", encoding="utf-8")

    with pytest.raises(MemoriesBackendError, match="not found"):
        store.read_memory(path=".secret.md")


def test_read_rejects_binary_file(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.root / "bin.md").write_bytes(b"\xff\xfe")

    with pytest.raises(MemoriesBackendError, match="not valid UTF-8 text"):
        store.read_memory(path="bin.md")


def test_read_line_offset_exceeds_length(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.root / "doc.md").write_text("only\n", encoding="utf-8")

    with pytest.raises(MemoriesBackendError, match="exceeds file length"):
        store.read_memory(path="doc.md", line_offset=99)


def test_validate_read_path_rejects_null_byte() -> None:
    with pytest.raises(MemoriesBackendError, match="must stay within"):
        validate_read_path("notes\x00.md")


def test_dispatch_read_empty_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FORGE_MEMORIES_ROOT", str(tmp_path / "memories"))
    ctx = ToolCallContext(cwd=tmp_path)

    result = dispatch_tool("memories.read", {"path": "   "}, ctx=ctx)
    assert result.is_error
    assert result.content == "path is required"


def test_dispatch_read_success(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "memories"
    root.mkdir()
    (root / "note.md").write_text("hello\nworld\n", encoding="utf-8")
    monkeypatch.setenv("FORGE_MEMORIES_ROOT", str(root))
    ctx = ToolCallContext(cwd=tmp_path)

    result = dispatch_tool(
        "memories.read",
        {"path": "note.md", "line_offset": 2, "max_lines": 1},
        ctx=ctx,
    )
    assert not result.is_error
    assert result.structured is not None
    assert result.structured["path"] == "note.md"
    assert result.structured["start_line_number"] == 2
    assert result.structured["content"] == "world\n"


def test_dispatch_read_binary_file(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "memories"
    root.mkdir()
    (root / "bin.md").write_bytes(b"\xff\xfe")
    monkeypatch.setenv("FORGE_MEMORIES_ROOT", str(root))
    ctx = ToolCallContext(cwd=tmp_path)

    result = dispatch_tool("memories.read", {"path": "bin.md"}, ctx=ctx)
    assert result.is_error
    assert "not valid UTF-8 text" in result.content


def test_read_rejects_path_traversal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (store.root / "ok.md").write_text("safe", encoding="utf-8")

    for bad in [
        "../outside.txt",
        "nested/../../outside.txt",
        str(outside),
        "/etc/passwd",
    ]:
        with pytest.raises(MemoriesBackendError, match="must stay within|not found"):
            store.read_memory(path=bad)


def test_read_rejects_symlink_escape(tmp_path: Path) -> None:
    store = _store(tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("TOP SECRET", encoding="utf-8")

    (store.root / "link.md").symlink_to(outside)
    with pytest.raises(MemoriesBackendError, match="symlink|must stay within"):
        store.read_memory(path="link.md")

    (store.root / "escape").symlink_to(tmp_path)
    with pytest.raises(MemoriesBackendError, match="symlink|must stay within"):
        store.read_memory(path="escape/secret.txt")


def test_read_rejects_oversized_file(tmp_path: Path) -> None:
    store = _store(tmp_path)
    big = store.root / "big.md"
    big.write_bytes(b"x" * (MAX_READ_FILE_BYTES + 1))

    with pytest.raises(MemoriesBackendError, match="byte read limit"):
        store.read_memory(path="big.md")


def test_dispatch_read_path_traversal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FORGE_MEMORIES_ROOT", str(tmp_path / "memories"))
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    ctx = ToolCallContext(cwd=tmp_path)

    result = dispatch_tool("memories.read", {"path": "../outside.txt"}, ctx=ctx)
    assert result.is_error
    assert "must stay within" in result.content or "not found" in result.content

    result = dispatch_tool("memories.read", {"path": str(outside)}, ctx=ctx)
    assert result.is_error
    assert "must stay within" in result.content


def test_search_any_and_all_on_same_line(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.root / "t.md").write_text("alpha beta\ngamma\n", encoding="utf-8")

    any_hit = store.search_memories(queries=["alpha", "gamma"])
    assert len(any_hit.matches) == 2

    same_line = store.search_memories(
        queries=["alpha", "beta"],
        match_mode=SearchMatchMode(kind=SearchMatchModeKind.ALL_ON_SAME_LINE),
    )
    assert len(same_line.matches) == 1
    assert same_line.matches[0].match_line_number == 1


def test_search_all_within_lines(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.root / "w.md").write_text("foo\nbar baz\n", encoding="utf-8")

    out = store.search_memories(
        queries=["foo", "baz"],
        match_mode=SearchMatchMode(
            kind=SearchMatchModeKind.ALL_WITHIN_LINES, line_count=2
        ),
    )
    assert len(out.matches) == 1
    assert "foo" in out.matches[0].content
    assert "baz" in out.matches[0].content


def test_search_case_insensitive_multi_query(tmp_path: Path) -> None:
    """Mirrors codex ext/memories search_tool_accepts_multiple_queries."""
    store = _store(tmp_path)
    (store.root / "MEMORY.md").write_text(
        "alpha only\nneedle only\nalpha needle\n",
        encoding="utf-8",
    )

    out = store.search_memories(queries=["alpha", "needle"], case_sensitive=False)
    assert [m.match_line_number for m in out.matches] == [1, 2, 3]
    assert out.matches[0].matched_queries == ["alpha"]
    assert out.matches[1].matched_queries == ["needle"]
    assert out.matches[2].matched_queries == ["alpha", "needle"]


def test_search_windowed_all_within_lines_codex_shape(tmp_path: Path) -> None:
    """Mirrors codex ext/memories search_tool_accepts_windowed_all_match_mode."""
    store = _store(tmp_path)
    (store.root / "MEMORY.md").write_text("alpha\nmiddle\nneedle\n", encoding="utf-8")

    out = store.search_memories(
        queries=["alpha", "needle"],
        match_mode=SearchMatchMode(
            kind=SearchMatchModeKind.ALL_WITHIN_LINES, line_count=3
        ),
    )
    assert len(out.matches) == 1
    assert out.matches[0].match_line_number == 1
    assert out.matches[0].content == "alpha\nmiddle\nneedle"
    assert out.matches[0].matched_queries == ["alpha", "needle"]


def test_search_no_results(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.root / "note.md").write_text("hello world\n", encoding="utf-8")

    out = store.search_memories(queries=["missing"])
    assert out.matches == []
    assert out.next_cursor is None
    assert not out.truncated


def test_search_empty_query_errors(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.root / "note.md").write_text("x\n", encoding="utf-8")

    with pytest.raises(MemoriesBackendError, match="at least one query is required"):
        store.search_memories(queries=[])

    with pytest.raises(
        MemoriesBackendError, match="queries must not contain empty strings"
    ):
        store.search_memories(queries=["ok", "  "])


def test_search_normalized_match(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.root / "note.md").write_text("foo-bar_baz\n", encoding="utf-8")

    out = store.search_memories(queries=["foobarbaz"], normalized=True)
    assert len(out.matches) == 1
    assert out.matches[0].match_line_number == 1


def test_search_normalized_empty_query_error(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(MemoriesBackendError, match="empty after normalization"):
        store.search_memories(queries=["!!!"], normalized=True)


def test_search_context_lines_and_pagination(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.root / "doc.md").write_text(
        "line0\nhit1\nline2\nhit3\nline4\n", encoding="utf-8"
    )

    with_ctx = store.search_memories(queries=["hit"], context_lines=1)
    assert with_ctx.matches[0].content == "line0\nhit1\nline2"
    assert with_ctx.matches[0].content_start_line_number == 1

    page1 = store.search_memories(queries=["hit"], max_results=1)
    assert len(page1.matches) == 1
    assert page1.truncated
    assert page1.next_cursor == "1"

    page2 = store.search_memories(
        queries=["hit"], cursor=page1.next_cursor, max_results=1
    )
    assert len(page2.matches) == 1
    assert page2.matches[0].match_line_number == 4


def test_search_path_scoped_and_skips_hidden(tmp_path: Path) -> None:
    store = _store(tmp_path)
    nested = store.root / "nested"
    nested.mkdir()
    (nested / "visible.md").write_text("secret keyword\n", encoding="utf-8")
    (nested / ".hidden.md").write_text("secret keyword\n", encoding="utf-8")
    (store.root / "other.md").write_text("secret keyword\n", encoding="utf-8")

    scoped = store.search_memories(queries=["secret"], path="nested")
    paths = {m.path for m in scoped.matches}
    assert paths == {"nested/visible.md"}


def test_search_rejects_path_traversal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("needle\n", encoding="utf-8")

    with pytest.raises(MemoriesBackendError, match="must stay within|not found"):
        store.search_memories(queries=["needle"], path="../outside.txt")


def test_search_query_limits_and_invalid_cursor(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (store.root / "a.md").write_text("x\n", encoding="utf-8")

    too_many = [f"q{i}" for i in range(MAX_SEARCH_QUERIES + 1)]
    with pytest.raises(MemoriesBackendError, match=f"at most {MAX_SEARCH_QUERIES}"):
        store.search_memories(queries=too_many)

    with pytest.raises(
        MemoriesBackendError,
        match="all_within_lines.line_count must be a positive integer",
    ):
        store.search_memories(
            queries=["x"],
            match_mode=SearchMatchMode(
                kind=SearchMatchModeKind.ALL_WITHIN_LINES, line_count=0
            ),
        )

    with pytest.raises(
        MemoriesBackendError,
        match=f"at most {MAX_SEARCH_MATCH_WINDOW_LINES}",
    ):
        store.search_memories(
            queries=["x"],
            match_mode=SearchMatchMode(
                kind=SearchMatchModeKind.ALL_WITHIN_LINES,
                line_count=MAX_SEARCH_MATCH_WINDOW_LINES + 1,
            ),
        )

    with pytest.raises(MemoriesBackendError, match="exceeds result count"):
        store.search_memories(queries=["x"], cursor="99")


def test_search_match_mode_from_wire_rejects_zero_line_count() -> None:
    mode = SearchMatchMode.from_wire({"type": "all_within_lines", "line_count": 0})
    assert mode.line_count == 0


def test_search_context_lines_clamped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lines = (
        "\n".join(f"line{i}" for i in range(10))
        + "\nneedle\n"
        + "\n".join(f"tail{i}" for i in range(10))
    )
    (store.root / "big.md").write_text(lines, encoding="utf-8")

    out = store.search_memories(
        queries=["needle"], context_lines=MAX_SEARCH_CONTEXT_LINES + 100
    )
    content_lines = out.matches[0].content.splitlines()
    assert len(content_lines) <= 1 + 2 * MAX_SEARCH_CONTEXT_LINES


def test_dispatch_search_legacy_query_and_errors(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FORGE_MEMORIES_ROOT", str(tmp_path / "memories"))
    root = tmp_path / "memories"
    root.mkdir()
    (root / "note.md").write_text("findme here\n", encoding="utf-8")
    ctx = ToolCallContext(cwd=tmp_path)

    legacy = dispatch_tool("memories.search", {"query": "findme"}, ctx=ctx)
    assert not legacy.is_error
    assert legacy.structured is not None
    assert legacy.structured["matches"]

    empty = dispatch_tool("memories.search", {"queries": []}, ctx=ctx)
    assert empty.is_error
    assert "at least one query is required" in empty.content


def test_ad_hoc_note_path_and_validation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    fname = "2026-05-23T10-11-12-test-note.md"
    store.add_ad_hoc_note(filename=fname, note="remember me")
    note_path = store.root / "extensions" / "ad_hoc" / "notes" / fname
    assert note_path.is_file()
    assert note_path.read_text(encoding="utf-8") == "remember me"
    assert oct(note_path.stat().st_mode & 0o777) == oct(0o600)

    with pytest.raises(MemoriesBackendError, match="must use YYYY-MM-DDTHH-MM-SS"):
        store.add_ad_hoc_note(filename="bad-name.md", note="x")

    with pytest.raises(MemoriesBackendError, match="must use YYYY-MM-DDTHH-MM-SS"):
        store.add_ad_hoc_note(
            filename="2026-05-23T10-11-12-evil/../escape.md", note="x"
        )


def test_ad_hoc_note_empty_content_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    fname = "2026-05-23T10-11-12-empty.md"

    with pytest.raises(MemoriesBackendError, match="must not be empty"):
        store.add_ad_hoc_note(filename=fname, note="")
    with pytest.raises(MemoriesBackendError, match="must not be empty"):
        store.add_ad_hoc_note(filename=fname, note="   \n\t")

    assert not (store.root / "extensions" / "ad_hoc" / "notes" / fname).exists()


def test_ad_hoc_note_rejects_symlink_in_notes_dir(tmp_path: Path) -> None:
    store = _store(tmp_path)
    escape = tmp_path / "escape"
    escape.mkdir()
    (escape / "ad_hoc").mkdir()
    (escape / "ad_hoc" / "notes").mkdir()
    link = store.root / "extensions"
    link.symlink_to(escape)

    with pytest.raises(MemoriesBackendError, match="symlink"):
        store.add_ad_hoc_note(filename="2026-05-23T10-11-12-link.md", note="x")


def test_ad_hoc_note_content_too_large_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    fname = "2026-05-23T10-11-13-too-big.md"
    oversized = "x" * (AD_HOC_NOTE_MAX_BYTES + 1)

    with pytest.raises(Exception, match="byte limit"):
        store.add_ad_hoc_note(filename=fname, note=oversized)

    assert not (store.root / "extensions" / "ad_hoc" / "notes" / fname).exists()


def test_ad_hoc_note_duplicate_filename_rejected_without_overwrite(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    fname = "2026-05-23T10-11-14-dup.md"
    store.add_ad_hoc_note(filename=fname, note="first")

    with pytest.raises(Exception, match="already exists"):
        store.add_ad_hoc_note(filename=fname, note="second")

    note_path = store.root / "extensions" / "ad_hoc" / "notes" / fname
    assert note_path.read_text(encoding="utf-8") == "first"


def test_default_ad_hoc_filename_unique_for_identical_slug() -> None:
    names = {default_ad_hoc_filename("same note") for _ in range(50)}
    assert len(names) == 50


def test_ad_hoc_note_concurrent_writes_do_not_conflict(tmp_path: Path) -> None:
    store = _store(tmp_path)
    errors: list[Exception] = []

    def _add(i: int) -> None:
        try:
            store.add_ad_hoc_note(
                filename=default_ad_hoc_filename("concurrent"), note=f"note {i}"
            )
        except Exception as exc:  # noqa: BLE001 - captured for assertion below
            errors.append(exc)

    threads = [threading.Thread(target=_add, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    notes_dir = store.root / "extensions" / "ad_hoc" / "notes"
    assert len(list(notes_dir.iterdir())) == 20


def test_dispatch_memories_wire(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FORGE_MEMORIES_ROOT", str(tmp_path / "memories"))
    ctx = ToolCallContext(cwd=tmp_path)
    fname = "2026-05-23T12-00-00-remember-detail.md"
    add = dispatch_tool(
        "memories.add_ad_hoc_note",
        {"filename": fname, "note": "user asked to remember"},
        ctx=ctx,
    )
    assert not add.is_error
    assert add.structured == {}

    listed = dispatch_tool(
        "memories.list", {"path": "extensions/ad_hoc/notes"}, ctx=ctx
    )
    assert not listed.is_error
    assert listed.structured is not None
    assert listed.structured["entries"]

    search = dispatch_tool(
        "memories.search",
        {"queries": ["remember"], "context_lines": 0},
        ctx=ctx,
    )
    assert not search.is_error
    assert search.structured is not None
    assert search.structured["matches"]


def test_dispatch_add_ad_hoc_note_codex_wire_aliases(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FORGE_MEMORIES_ROOT", str(tmp_path / "memories"))
    ctx = ToolCallContext(cwd=tmp_path)

    auto = dispatch_tool(
        "memories.add_ad_hoc_note", {"content": "remember via content"}, ctx=ctx
    )
    assert not auto.is_error
    notes_dir = tmp_path / "memories" / "extensions" / "ad_hoc" / "notes"
    created = list(notes_dir.iterdir())
    assert len(created) == 1
    assert created[0].read_text(encoding="utf-8") == "remember via content"

    fname = "2026-05-23T12-01-00-named.md"
    named = dispatch_tool(
        "memories.add_ad_hoc_note",
        {"content": "named note", "path": fname},
        ctx=ctx,
    )
    assert not named.is_error
    assert (notes_dir / fname).read_text(encoding="utf-8") == "named note"

    empty = dispatch_tool("memories.add_ad_hoc_note", {"content": "  "}, ctx=ctx)
    assert empty.is_error
    assert "must not be empty" in empty.content
