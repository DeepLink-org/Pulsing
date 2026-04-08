"""TransferBackend - indexed in-memory backend with incremental field merge semantics.

Samples are keyed by ``sample_idx``. Each ``put()`` merges new fields into the
existing sample dict, enabling incremental construction (for example prompt
first, response later).

Compared with the naive implementation, this version keeps per-query readiness
caches so repeated ``get_data(fields=...)`` calls avoid re-scanning all samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

logger = logging.getLogger(__name__)


QueryKey = tuple[str, ...]


@dataclass(slots=True)
class _QueryState:
    """Cached readiness state for one requested field-set."""

    required_fields: frozenset[str]
    ready_queue: list[int] = field(default_factory=list)
    ready_set: set[int] = field(default_factory=set)
    build_count: int = 0
    hit_count: int = 0


@dataclass(slots=True)
class _ConsumerCursor:
    """Per-task cursor for one cached query."""

    next_pos: int = 0


class TransferBackend:
    """Indexed in-memory backend for transfer queue.

    The backend optimizes repeated reads of the same field combinations by:
    - caching which samples satisfy a given field-set query
    - incrementally appending newly-ready samples to those caches on ``put()``
    - tracking per-task read cursors so consumption is mostly sequential

    Args:
        bucket_id: Bucket ID this backend serves
        batch_size: Default batch size for get_data
    """

    def __init__(self, bucket_id: int, batch_size: int = 10, **kwargs):
        self.bucket_id = bucket_id
        self.batch_size = batch_size

        # sample_idx -> merged field dict
        self._samples: dict[int, dict[str, Any]] = {}
        # sample_idx -> set of field names that have been written
        self._field_status: dict[int, set[str]] = {}

        # task_name -> set of sample_idxs already consumed
        self._consumed: dict[str, set[int]] = {}
        # task_name -> query_key -> cursor
        self._consumer_cursors: dict[str, dict[QueryKey, _ConsumerCursor]] = {}

        # query_key -> readiness cache for that field combination
        self._query_states: dict[QueryKey, _QueryState] = {}
        # field_name -> query_keys that depend on this field
        self._queries_by_field: dict[str, set[QueryKey]] = {}

    def _normalize_query_key(self, fields: list[str]) -> QueryKey:
        if not fields:
            raise ValueError("fields must not be empty")
        return tuple(sorted(set(fields)))

    def _build_query_state(self, query_key: QueryKey) -> _QueryState:
        required_fields = frozenset(query_key)
        ready_queue = [
            sample_idx
            for sample_idx in sorted(self._field_status)
            if required_fields.issubset(self._field_status[sample_idx])
        ]
        state = _QueryState(
            required_fields=required_fields,
            ready_queue=ready_queue,
            ready_set=set(ready_queue),
            build_count=1,
        )
        self._query_states[query_key] = state
        for field_name in required_fields:
            self._queries_by_field.setdefault(field_name, set()).add(query_key)
        return state

    def _get_query_state(self, query_key: QueryKey) -> _QueryState:
        state = self._query_states.get(query_key)
        if state is not None:
            state.hit_count += 1
            return state
        return self._build_query_state(query_key)

    def _append_ready_sample(self, query_state: _QueryState, sample_idx: int) -> None:
        if sample_idx in query_state.ready_set:
            return
        query_state.ready_queue.append(sample_idx)
        query_state.ready_set.add(sample_idx)

    def _update_query_caches(self, sample_idx: int, new_fields: set[str]) -> None:
        if not new_fields:
            return

        candidate_queries: set[QueryKey] = set()
        for field_name in new_fields:
            candidate_queries.update(self._queries_by_field.get(field_name, ()))

        if not candidate_queries:
            return

        sample_fields = self._field_status[sample_idx]
        for query_key in candidate_queries:
            query_state = self._query_states.get(query_key)
            if query_state is None:
                continue
            if query_state.required_fields.issubset(sample_fields):
                self._append_ready_sample(query_state, sample_idx)

    async def put(self, sample_idx: int, data: dict[str, Any]) -> dict[str, Any]:
        """Merge *data* into the sample identified by *sample_idx*.

        Returns:
            Meta dict with current field status for the sample.
        """
        if sample_idx not in self._samples:
            self._samples[sample_idx] = {}
            self._field_status[sample_idx] = set()

        current_fields = self._field_status[sample_idx]
        new_fields = set(data).difference(current_fields)

        self._samples[sample_idx].update(data)
        current_fields.update(data.keys())

        self._update_query_caches(sample_idx, new_fields)

        return {
            "sample_idx": sample_idx,
            "fields": sorted(current_fields),
            "status": "ok",
        }

    async def get_data(
        self,
        fields: list[str],
        batch_size: int,
        task_name: str = "default",
    ) -> list[dict[str, Any]]:
        """Return up to *batch_size* unread samples that contain all *fields*."""
        query_key = self._normalize_query_key(fields)
        query_state = self._get_query_state(query_key)

        consumed = self._consumed.setdefault(task_name, set())
        cursors = self._consumer_cursors.setdefault(task_name, {})
        cursor = cursors.setdefault(query_key, _ConsumerCursor())

        results: list[dict[str, Any]] = []
        queue = query_state.ready_queue
        pos = cursor.next_pos

        while pos < len(queue) and len(results) < batch_size:
            sample_idx = queue[pos]
            pos += 1
            if sample_idx in consumed:
                continue

            sample = self._samples.get(sample_idx)
            if sample is None:
                continue

            consumed.add(sample_idx)
            results.append({field_name: sample[field_name] for field_name in query_key})

        cursor.next_pos = pos
        return results

    async def clear(self) -> None:
        """Reset all state."""
        self._samples.clear()
        self._field_status.clear()
        self._consumed.clear()
        self._consumer_cursors.clear()
        self._query_states.clear()
        self._queries_by_field.clear()

    async def stats(self) -> dict[str, Any]:
        """Return diagnostic info."""
        return {
            "bucket_id": self.bucket_id,
            "sample_count": len(self._samples),
            "cached_queries": len(self._query_states),
            "cached_query_hits": {
                ",".join(query_key): state.hit_count
                for query_key, state in self._query_states.items()
            },
            "consumed_tasks": {k: len(v) for k, v in self._consumed.items()},
            "backend": "transfer_memory",
            "implementation": "indexed",
        }
