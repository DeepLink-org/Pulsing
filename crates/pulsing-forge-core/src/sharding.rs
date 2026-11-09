//! 一致性哈希分片 — 多 shard 的店→节点分配(10万级门槛的地基)。
//! 性质: 增删节点只迁移 ~1/N 的店; 同名店稳定落点(账本连续性)。

/// 环: 节点按哈希环分布, 店顺时针找最近节点。
#[derive(Debug, Clone)]
pub struct Ring {
    /// (hash, node) 升序
    points: Vec<(u64, String)>,
    /// 每节点虚节点数(默认 160, 小集群均匀性够用)
    vnodes: usize,
}

impl Ring {
    pub fn new(nodes: &[String], vnodes: usize) -> Self {
        let mut points = Vec::with_capacity(nodes.len() * vnodes);
        for n in nodes {
            for v in 0..vnodes {
                points.push((Ring::hash(&format!("{n}#{v}")), n.clone()));
            }
        }
        points.sort();
        Ring { points, vnodes }
    }

    /// FNV-1a 64bit: 稳定、无依赖、跨平台一致(账本连续性要求确定性)。
    pub fn hash(s: &str) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in s.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// 店 → 负责节点(顺时针首个 ≥ hash(point))。
    pub fn node_for(&self, shop: &str) -> &str {
        if self.points.is_empty() {
            return "";
        }
        let h = Ring::hash(shop);
        let idx = self.points.partition_point(|(p, _)| *p < h);
        &self.points[idx % self.points.len()].1
    }

    /// 扩容后需迁移的店(占全量比例应 ≈ 1/N — 增量迁移性质)。
    pub fn migrating<'a>(&self, old: &'a Ring, shops: &[String]) -> Vec<String> {
        shops
            .iter()
            .filter(|s| self.node_for(s) != old.node_for(s))
            .cloned()
            .collect()
    }

    pub fn nodes(&self) -> Vec<String> {
        let mut ns: Vec<String> = self.points.iter().map(|(_, n)| n.clone()).collect();
        ns.sort();
        ns.dedup();
        ns
    }
    pub fn vnode_count(&self) -> usize {
        self.vnodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shops(n: usize) -> Vec<String> {
        (1..=n).map(|i| format!("agent_{i}")).collect()
    }

    /// 基本性质: 分配全覆盖且确定性(同名店同节点)
    #[test]
    fn deterministic_assignment() {
        let ring = Ring::new(&["n1".into(), "n2".into(), "n3".into()], 160);
        let s = shops(100);
        assert!(s.iter().all(|x| !ring.node_for(x).is_empty()));
        assert_eq!(ring.node_for("agent_1"), ring.node_for("agent_1"));
    }

    /// 核心性质: 3→4 节点扩容, 迁移比例 ≈ 1/4 (容忍区间 15%~40%)
    #[test]
    fn incremental_rebalance() {
        let old = Ring::new(&["n1".into(), "n2".into(), "n3".into()], 160);
        let new = Ring::new(&["n1".into(), "n2".into(), "n3".into(), "n4".into()], 160);
        let s = shops(1000);
        let mig = new.migrating(&old, &s);
        let ratio = mig.len() as f64 / s.len() as f64;
        assert!(ratio > 0.15 && ratio < 0.40, "迁移比例异常: {ratio:.2}");
    }

    /// 负载均衡: 1000店/3节点, 每节点份额 20%~47%
    #[test]
    fn load_balance() {
        let ring = Ring::new(&["n1".into(), "n2".into(), "n3".into()], 160);
        let s = shops(1000);
        let mut count = std::collections::HashMap::new();
        for x in &s {
            *count.entry(ring.node_for(x).to_string()).or_insert(0usize) += 1;
        }
        for (_, c) in count {
            let share = c as f64 / 1000.0;
            assert!(share > 0.18 && share < 0.50, "节点份额失衡: {share:.2}");
        }
    }
}
