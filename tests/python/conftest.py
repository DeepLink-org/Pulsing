"""
Conftest for actor system tests.

This file overrides the parent conftest.py to avoid requiring NATS/ETCD.
Actor system tests use standalone mode and don't need external services.
"""

import pytest


# Override the parent's autouse fixture to prevent NATS/ETCD startup
@pytest.fixture(scope="module", autouse=True)
def nats_and_etcd():
    """Override parent fixture - actor system tests don't need NATS/ETCD."""
    yield None
