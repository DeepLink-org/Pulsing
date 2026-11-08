"""Validate the experiment contract without starting Codex or any model."""

import json
from collections import Counter
from pathlib import Path


def test_fermat_council_has_twenty_distinct_roles_and_hard_deadline():
    path = Path(__file__).parents[3] / "examples/python/fermat_council.json"
    plan = json.loads(path.read_text())
    roles = plan["roles"]
    assert plan["design_only"] is True
    assert (
        len(roles) == plan["participants"] == len({role["id"] for role in roles}) == 20
    )
    assert Counter(role["instance"] for role in roles) == {"A": 10, "B": 10}
    assert Counter(
        role["instance"] for role in roles if role["harness_role"] == "subagent"
    ) == {"A": 9, "B": 9}
    assert {role["role"] for role in roles if role["harness_role"] == "main"} == {
        "coordinator",
        "arbiter",
    }
    for group in ("G1", "G2", "G3", "G4"):
        assert {role["role"] for role in roles if role["group"] == group} == {
            "lead",
            "proposer",
            "critic",
            "checker",
        }
    end = 0
    for phase in plan["phases"]:
        assert phase["start"] == end < phase["end"]
        end = phase["end"]
    assert end == plan["deadline_seconds"] == 600
    assert plan["new_math_stop_seconds"] == 570
