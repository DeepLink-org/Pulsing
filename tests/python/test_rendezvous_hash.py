"""测试 Rendezvous Hashing 实现

验证：
1. 相同输入总是返回相同结果（确定性）
2. 负载分布均匀
3. 节点增减时迁移比例约为 1/N（最小迁移）
4. 只选择 Alive 节点
"""

import hashlib
from collections import Counter


def _compute_owner(bucket_key: str, nodes: list[dict]) -> int | None:
    """Rendezvous Hashing 实现（从 manager.py 复制用于独立测试）"""
    if not nodes:
        return None

    # 只选择 Alive 状态的节点
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
    """旧的 hash % N 实现（用于对比）"""
    if not nodes:
        return None
    sorted_nodes = sorted(nodes, key=lambda n: int(n.get("node_id", 0)))
    hash_value = int(hashlib.md5(bucket_key.encode()).hexdigest(), 16)
    index = hash_value % len(sorted_nodes)
    node_id = sorted_nodes[index].get("node_id")
    return int(node_id) if node_id is not None else None


class TestRendezvousHashing:
    """测试 Rendezvous Hashing"""

    def test_deterministic(self):
        """相同输入总是返回相同结果"""
        nodes = [{"node_id": i, "state": "Alive"} for i in range(5)]

        for _ in range(100):
            owner1 = _compute_owner("test_key", nodes)
            owner2 = _compute_owner("test_key", nodes)
            assert owner1 == owner2

    def test_empty_nodes(self):
        """空节点列表返回 None"""
        assert _compute_owner("test_key", []) is None

    def test_single_node(self):
        """单节点总是返回该节点"""
        nodes = [{"node_id": 42, "state": "Alive"}]
        assert _compute_owner("any_key", nodes) == 42

    def test_load_distribution(self):
        """负载分布应该相对均匀"""
        nodes = [{"node_id": i, "state": "Alive"} for i in range(5)]
        num_keys = 10000

        # 统计每个节点分配到的 key 数量
        distribution = Counter()
        for i in range(num_keys):
            key = f"bucket_{i}"
            owner = _compute_owner(key, nodes)
            distribution[owner] += 1

        # 理想情况每个节点应该分配 2000 个 key
        expected = num_keys / len(nodes)

        # 允许 20% 的偏差
        for node_id, count in distribution.items():
            ratio = count / expected
            assert (
                0.8 <= ratio <= 1.2
            ), f"Node {node_id} has {count} keys, expected ~{expected}"

    def test_minimal_migration_on_add_node(self):
        """添加节点时迁移比例应约为 1/(N+1)"""
        num_keys = 10000

        # 初始 5 个节点
        nodes_before = [{"node_id": i, "state": "Alive"} for i in range(5)]
        # 添加 1 个节点变成 6 个
        nodes_after = [{"node_id": i, "state": "Alive"} for i in range(6)]

        # 记录每个 key 的 owner 变化
        migrated = 0
        for i in range(num_keys):
            key = f"bucket_{i}"
            owner_before = _compute_owner(key, nodes_before)
            owner_after = _compute_owner(key, nodes_after)
            if owner_before != owner_after:
                migrated += 1

        migration_ratio = migrated / num_keys
        expected_ratio = 1 / 6  # 约 16.7%

        # Rendezvous hashing: 迁移比例应接近 1/(N+1)
        # 允许 50% 的误差范围
        assert (
            migration_ratio < expected_ratio * 1.5
        ), f"Migration ratio {migration_ratio:.2%} too high, expected ~{expected_ratio:.2%}"
        print(
            f"[Rendezvous] Add node: migration ratio = {migration_ratio:.2%} (expected ~{expected_ratio:.2%})"
        )

    def test_minimal_migration_on_remove_node(self):
        """移除节点时迁移比例应约为 1/N"""
        num_keys = 10000

        # 初始 6 个节点
        nodes_before = [{"node_id": i, "state": "Alive"} for i in range(6)]
        # 移除 1 个节点变成 5 个（移除 node_id=5）
        nodes_after = [{"node_id": i, "state": "Alive"} for i in range(5)]

        migrated = 0
        for i in range(num_keys):
            key = f"bucket_{i}"
            owner_before = _compute_owner(key, nodes_before)
            owner_after = _compute_owner(key, nodes_after)
            if owner_before != owner_after:
                migrated += 1

        migration_ratio = migrated / num_keys
        expected_ratio = 1 / 6  # 约 16.7%

        assert (
            migration_ratio < expected_ratio * 1.5
        ), f"Migration ratio {migration_ratio:.2%} too high, expected ~{expected_ratio:.2%}"
        print(
            f"[Rendezvous] Remove node: migration ratio = {migration_ratio:.2%} (expected ~{expected_ratio:.2%})"
        )

    def test_compare_with_old_algorithm(self):
        """对比新旧算法的迁移比例"""
        num_keys = 10000

        nodes_before = [{"node_id": i, "state": "Alive"} for i in range(5)]
        nodes_after = [{"node_id": i, "state": "Alive"} for i in range(6)]

        # 旧算法 (hash % N)
        old_migrated = 0
        for i in range(num_keys):
            key = f"bucket_{i}"
            old_before = _compute_owner_old(key, nodes_before)
            old_after = _compute_owner_old(key, nodes_after)
            if old_before != old_after:
                old_migrated += 1

        # 新算法 (Rendezvous)
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

        # 新算法应该显著优于旧算法
        assert (
            new_ratio < old_ratio * 0.5
        ), f"New algorithm ({new_ratio:.2%}) should be much better than old ({old_ratio:.2%})"

    def test_only_alive_nodes(self):
        """只选择 Alive 状态的节点"""
        nodes = [
            {"node_id": 0, "state": "Alive"},
            {"node_id": 1, "state": "Dead"},
            {"node_id": 2, "state": "Alive"},
            {"node_id": 3, "state": "Suspect"},
            {"node_id": 4, "state": "Alive"},
        ]

        # 检查多个 key，确保不会选择非 Alive 节点
        alive_ids = {0, 2, 4}
        for i in range(1000):
            key = f"test_key_{i}"
            owner = _compute_owner(key, nodes)
            assert owner in alive_ids, f"Key {key} assigned to non-alive node {owner}"

    def test_fallback_when_no_alive(self):
        """没有 Alive 节点时退回到所有节点"""
        nodes = [
            {"node_id": 0, "state": "Dead"},
            {"node_id": 1, "state": "Suspect"},
        ]

        owner = _compute_owner("test_key", nodes)
        assert owner in {0, 1}

    def test_node_order_independence(self):
        """结果不应依赖节点列表顺序"""
        import random

        nodes = [{"node_id": i, "state": "Alive"} for i in range(10)]

        key = "test_key"
        expected_owner = _compute_owner(key, nodes)

        # 打乱顺序多次测试
        for _ in range(100):
            shuffled = nodes.copy()
            random.shuffle(shuffled)
            owner = _compute_owner(key, shuffled)
            assert owner == expected_owner


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-s"])
