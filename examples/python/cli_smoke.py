"""Smoke test for pulsing-cli (Path B): native ActorSystem + python/pulsing imports."""

import pulsing.core as core

assert core.is_initialized(), "ActorSystem should be attached by pulsing-cli bootstrap"
system = core.get_system()
assert hasattr(system.node_id, "uuid"), "node_id must be NodeId, not str"
print("node_id:", system.node_id.uuid())
print("addr:", system.addr)
print("cli_smoke ok")
