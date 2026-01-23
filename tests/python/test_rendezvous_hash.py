"""Test Rendezvous Hashing Implementation

Verifies:
1. Same input always returns same result (deterministic)
2. Load distribution is uniform
3. Migration ratio is approximately 1/N when nodes are added/removed (minimal migration)
4. Only selects Alive nodes
"""

import hashlib
from collections import Counter


def _compute_owner(bucket_key: str, nodes: list[dict]) -> int | None:
    """Rendezvous Hashing implementation (copied from manager.py for standalone testing)"""
    if not nodes:
        return None

    # Only select nodes in Alive state
    alive_nodes = [n for n in nodes if n.get("state") == "Alive"]
    if not alive_nodes:
        alive_nodes = nodes

    best_score = -1
    best_node_id = None

    for node in alive_nodes:
        node_id = node.get("node_id")
        if node_id is None:
            continue
        node_id = int(node_id)
        combined = f"{bucket_key}:{node_id}"
        score = int(hashlib.md5(combined.encode()).hexdigest(), 16)
        if score > best_score:
            best_score = score
            best_node_id = node_id

    return best_node_id


def _compute_owner_old(bucket_key: str, nodes: list[dict]) -> int | None:
    """Old hash % N implementation (for comparison)"""
    if not nodes:
        return None
    sorted_nodes = sorted(nodes, key=lambda n: int(n.get("node_id", 0)))
    hash_value = int(hashlib.md5(bucket_key.encode()).hexdigest(), 16)
    index = hash_value % len(sorted_nodes)
    node_id = sorted_nodes[index].get("node_id")
    return int(node_id) if node_id is not None else None


class TestRendezvousHashing:
    """Test Rendezvous Hashing"""

    def test_deterministic(self):
        """Same input always returns same result"""
        nodes = [{"node_id": i, "state": "Alive"} for i in range(5)]

        for _ in range(100):
            owner1 = _compute_owner("test_key", nodes)
            owner2 = _compute_owner("test_key", nodes)
            assert owner1 == owner2

    def test_empty_nodes(self):
        """Empty node list returns None"""
        assert _compute_owner("test_key", []) is None

    def test_single_node(self):
        """Single node always returns that node"""
        nodes = [{"node_id": 42, "state": "Alive"}]
        assert _compute_owner("any_key", nodes) == 42

    def test_load_distribution(self):
        """Load distribution should be relatively uniform"""
        nodes = [{"node_id": i, "state": "Alive"} for i in range(5)]
        num_keys = 10000

        # Count number of keys assigned to each node
        distribution = Counter()
        for i in range(num_keys):
            key = f"bucket_{i}"
            owner = _compute_owner(key, nodes)
            distribution[owner] += 1

        # Ideally each node should be assigned 2000 keys
        expected = num_keys / len(nodes)

        # Allow 20% deviation
        for node_id, count in distribution.items():
            ratio = count / expected
            assert 0.8 <= ratio <= 1.2, (
                f"Node {node_id} has {count} keys, expected ~{expected}"
            )

    def test_minimal_migration_on_add_node(self):
        """Migration ratio should be approximately 1/(N+1) when adding a node"""
        num_keys = 10000

        # Initially 5 nodes
        nodes_before = [{"node_id": i, "state": "Alive"} for i in range(5)]
        # Add 1 node to make 6 nodes
        nodes_after = [{"node_id": i, "state": "Alive"} for i in range(6)]

        # Record owner changes for each key
        migrated = 0
        for i in range(num_keys):
            key = f"bucket_{i}"
            owner_before = _compute_owner(key, nodes_before)
            owner_after = _compute_owner(key, nodes_after)
            if owner_before != owner_after:
                migrated += 1

        migration_ratio = migrated / num_keys
        expected_ratio = 1 / 6  # Approximately 16.7%

        # Rendezvous hashing: migration ratio should be close to 1/(N+1)
        # Allow 50% error margin
        assert migration_ratio < expected_ratio * 1.5, (
            f"Migration ratio {migration_ratio:.2%} too high, expected ~{expected_ratio:.2%}"
        )
        print(
            f"[Rendezvous] Add node: migration ratio = {migration_ratio:.2%} (expected ~{expected_ratio:.2%})"
        )

    def test_minimal_migration_on_remove_node(self):
        """Migration ratio should be approximately 1/N when removing a node"""
        num_keys = 10000

        # Initially 6 nodes
        nodes_before = [{"node_id": i, "state": "Alive"} for i in range(6)]
        # Remove 1 node to make 5 nodes (remove node_id=5)
        nodes_after = [{"node_id": i, "state": "Alive"} for i in range(5)]

        migrated = 0
        for i in range(num_keys):
            key = f"bucket_{i}"
            owner_before = _compute_owner(key, nodes_before)
            owner_after = _compute_owner(key, nodes_after)
            if owner_before != owner_after:
                migrated += 1

        migration_ratio = migrated / num_keys
        expected_ratio = 1 / 6  # Approximately 16.7%

        assert migration_ratio < expected_ratio * 1.5, (
            f"Migration ratio {migration_ratio:.2%} too high, expected ~{expected_ratio:.2%}"
        )
        print(
            f"[Rendezvous] Remove node: migration ratio = {migration_ratio:.2%} (expected ~{expected_ratio:.2%})"
        )

    def test_compare_with_old_algorithm(self):
        """Compare migration ratios of old and new algorithms"""
        num_keys = 10000

        nodes_before = [{"node_id": i, "state": "Alive"} for i in range(5)]
        nodes_after = [{"node_id": i, "state": "Alive"} for i in range(6)]

        # Old algorithm (hash % N)
        old_migrated = 0
        for i in range(num_keys):
            key = f"bucket_{i}"
            old_before = _compute_owner_old(key, nodes_before)
            old_after = _compute_owner_old(key, nodes_after)
            if old_before != old_after:
                old_migrated += 1

        # New algorithm (Rendezvous)
        new_migrated = 0
        for i in range(num_keys):
            key = f"bucket_{i}"
            new_before = _compute_owner(key, nodes_before)
            new_after = _compute_owner(key, nodes_after)
            if new_before != new_after:
                new_migrated += 1

        old_ratio = old_migrated / num_keys
        new_ratio = new_migrated / num_keys

        print(
            f"[Comparison] Old (hash%N): {old_ratio:.2%}, New (Rendezvous): {new_ratio:.2%}"
        )

        # New algorithm should be significantly better than old
        assert new_ratio < old_ratio * 0.5, (
            f"New algorithm ({new_ratio:.2%}) should be much better than old ({old_ratio:.2%})"
        )

    def test_only_alive_nodes(self):
        """Only select nodes in Alive state"""
        nodes = [
            {"node_id": 0, "state": "Alive"},
            {"node_id": 1, "state": "Dead"},
            {"node_id": 2, "state": "Alive"},
            {"node_id": 3, "state": "Suspect"},
            {"node_id": 4, "state": "Alive"},
        ]

        # Check multiple keys to ensure non-Alive nodes are not selected
        alive_ids = {0, 2, 4}
        for i in range(1000):
            key = f"test_key_{i}"
            owner = _compute_owner(key, nodes)
            assert owner in alive_ids, f"Key {key} assigned to non-alive node {owner}"

    def test_fallback_when_no_alive(self):
        """Fall back to all nodes when no Alive nodes exist"""
        nodes = [
            {"node_id": 0, "state": "Dead"},
            {"node_id": 1, "state": "Suspect"},
        ]

        owner = _compute_owner("test_key", nodes)
        assert owner in {0, 1}

    def test_node_order_independence(self):
        """Result should not depend on node list order"""
        import random

        nodes = [{"node_id": i, "state": "Alive"} for i in range(10)]

        key = "test_key"
        expected_owner = _compute_owner(key, nodes)

        # Test multiple times with shuffled order
        for _ in range(100):
            shuffled = nodes.copy()
            random.shuffle(shuffled)
            owner = _compute_owner(key, shuffled)
            assert owner == expected_owner


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-s"])
