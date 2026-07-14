"""@pul.remote user instances must pickle (e.g. isolated ``new_process=True`` spawn)."""

from __future__ import annotations

import pickle

import pytest

from pulsing.core.remote import _WrappedActor, remote


@remote
class _PickleProbe:
    def __init__(self, n: int = 7) -> None:
        self.n = n

    def f(self) -> int:
        return self.n


def test_remote_user_instance_roundtrip_pickles_wrapped_actor() -> None:
    inst = _PickleProbe(42)
    w = _WrappedActor(inst)
    w2 = pickle.loads(pickle.dumps(w, protocol=pickle.HIGHEST_PROTOCOL))
    assert isinstance(w2, _WrappedActor)
    assert w2._instance.n == 42
    assert w2._instance.f() == 42


def test_remote_user_instance_roundtrip_without_wrapper() -> None:
    inst = _PickleProbe(99)
    inst2 = pickle.loads(pickle.dumps(inst, protocol=pickle.HIGHEST_PROTOCOL))
    assert inst2.n == 99
