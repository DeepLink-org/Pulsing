"""TransferBackend - fixed-capacity per-bucket storage for transfer_queue."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class _Slot:
    sample_idx: int
    data: dict[str, Any]
    fields: set[str]


class TransferBackend:
    """Fixed-capacity in-memory backend for one transfer queue bucket.

    Each bucket stores at most ``bucket_capacity`` samples. New sample IDs are
    inserted into a ring buffer; when the bucket is full, the oldest inserted
    sample is overwritten. Repeated ``put()`` calls for the same ``sample_idx``
    merge fields into the existing sample and do not change FIFO order.
    """

    def __init__(self, bucket_id: int, bucket_capacity: int = 10, **kwargs):
        if bucket_capacity <= 0:
            raise ValueError("bucket_capacity must be greater than 0")

        self.bucket_id = bucket_id
        self.bucket_capacity = bucket_capacity

        self._slots: list[_Slot | None] = [None] * bucket_capacity
        self._sample_to_slot: dict[int, int] = {}
        self._next_slot = 0
        self._size = 0

    @staticmethod
    def _normalize_fields(fields: list[str]) -> tuple[str, ...]:
        if not fields:
            raise ValueError("data_fields must not be empty")
        return tuple(dict.fromkeys(fields))

    def _store_new_sample(self, sample_idx: int, data: dict[str, Any]) -> _Slot:
        slot_idx = self._next_slot
        existing = self._slots[slot_idx]
        if existing is not None:
            del self._sample_to_slot[existing.sample_idx]
        else:
            self._size += 1

        slot = _Slot(sample_idx=sample_idx, data=dict(data), fields=set(data))
        self._slots[slot_idx] = slot
        self._sample_to_slot[sample_idx] = slot_idx
        self._next_slot = (slot_idx + 1) % self.bucket_capacity
        return slot

    async def put(self, sample_idx: int, data: dict[str, Any]) -> dict[str, Any]:
        """Merge *data* into *sample_idx*, creating or overwriting a slot."""
        slot_idx = self._sample_to_slot.get(sample_idx)
        if slot_idx is None:
            slot = self._store_new_sample(sample_idx, data)
        else:
            slot = self._slots[slot_idx]
            if slot is None:
                slot = self._store_new_sample(sample_idx, data)
            else:
                slot.data.update(data)
                slot.fields.update(data.keys())

        return {
            "bucket_id": self.bucket_id,
            "sample_idx": sample_idx,
            "fields": sorted(slot.fields),
            "status": "ok",
        }

    async def get_data(
        self,
        fields: list[str],
        sample_idx: int,
    ) -> dict[str, Any] | None:
        """Return one sample if it still exists and contains all required fields."""
        normalized_fields = self._normalize_fields(fields)
        slot_idx = self._sample_to_slot.get(sample_idx)
        if slot_idx is None:
            return None

        slot = self._slots[slot_idx]
        if slot is None or slot.sample_idx != sample_idx:
            return None

        if not set(normalized_fields).issubset(slot.fields):
            return None

        return {field_name: slot.data[field_name] for field_name in normalized_fields}

    async def clear(self) -> None:
        """Reset all state in this bucket."""
        self._slots = [None] * self.bucket_capacity
        self._sample_to_slot.clear()
        self._next_slot = 0
        self._size = 0

    async def stats(self) -> dict[str, Any]:
        """Return diagnostic info."""
        return {
            "bucket_id": self.bucket_id,
            "bucket_capacity": self.bucket_capacity,
            "sample_count": self._size,
            "next_slot": self._next_slot,
            "stored_samples": sorted(self._sample_to_slot),
            "backend": "transfer_ring_buffer",
        }
