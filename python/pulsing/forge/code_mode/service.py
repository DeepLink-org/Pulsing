# SPDX-License-Identifier: Apache-2.0
"""Session-scoped Code Mode cell registry and exec/wait orchestration."""

from __future__ import annotations

from pulsing.forge.code_mode.cell import CellStatus, CodeCell
from pulsing.forge.code_mode.parse import parse_exec_source
from pulsing.forge.code_mode.protocol import CellId, RuntimeResponse, WaitArgs
from pulsing.forge.code_mode.tools_bridge import ToolsBridge


class CodeModeService:
    """In-process code cells (Actor control plane can wrap this later)."""

    def __init__(self) -> None:
        self._cells: dict[str, CodeCell] = {}

    def execute(self, source: str, tools: ToolsBridge) -> RuntimeResponse:
        parsed = parse_exec_source(source)
        cell = CodeCell.new(parsed)
        cell.run(tools)
        self._cells[str(cell.cell_id)] = cell
        return cell.to_response()

    def wait(self, args: WaitArgs) -> RuntimeResponse:
        """Resume a yielded cell or return its latest snapshot.

        ``yield_time_ms`` is accepted for wire compatibility with Codex; there
        is no OS-level blocking wait in the Python cell runtime.
        """
        cell = self._cells.get(args.cell_id)
        if cell is None:
            return RuntimeResponse(
                kind="result",
                cell_id=CellId(args.cell_id or "unknown"),
                error_text=f"unknown cell_id: {args.cell_id}",
            )

        if args.terminate:
            cell.mark_terminated()
        elif cell.status == CellStatus.YIELDED and cell._tools is not None:
            cell.run(cell._tools)

        return cell.to_response(max_tokens=args.max_tokens)

    def get(self, cell_id: str) -> CodeCell | None:
        return self._cells.get(cell_id)
