//! forge-core — 闸门原语: 把压测验证过的看门狗逻辑收编为库。
//! 元原则: 机器管协议(收齐/裁决/投递/收尾/预算), 模型只管语义。

use std::collections::HashMap;

/// 一封信的最小表示(来自账本回放)。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Letter {
    pub ts_ms: u64,
    /// 账本序号(journal 追加时分配; 内存构造默认0, recall 按它取全文)
    pub seq: u64,
    pub from: String,
    pub to: String,
    pub body: String,
}

/// 审查结论。按**路由**分类(审查店→judge), 不信前缀——前缀纪律漂移是实测
/// 八成会发生的行为(failure-taxonomy #14)。
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Clean,
    NeedsFix,
    Unknown,
}

/// 关键词判定表(实测校准): 命中任一即不通过。覆盖 CLEAN/NEEDS_FIX/通过/不通过/
/// BLOCK/failed/暂不/有问题 等全部实测变体。
pub fn classify(body: &str) -> Verdict {
    // 前缀优先: 标准前缀直接定(职责文本可能撞关键词, 如"失败返回路径检查"撞"失败")
    if body.starts_with("CLEAN") {
        return Verdict::Clean;
    }
    if body.starts_with("NEEDS_FIX") {
        return Verdict::NeedsFix;
    }
    let head = &body[..body.len().min(160)];
    let bad = [
        "不通过",
        "NEEDS_FIX",
        "暂不",
        "有问题",
        "无法",
        "BLOCK",
        "blocked",
        "failed",
        "失败",
        "REJECT",
        "缺",
    ];
    let good = ["CLEAN", "通过"];
    let has_bad = bad.iter().any(|k| head.contains(k));
    let has_good = good.iter().any(|k| head.contains(k));
    if has_bad {
        Verdict::NeedsFix
    } else if has_good {
        Verdict::Clean
    } else {
        Verdict::Unknown
    }
}

/// 收齐闸门: 波次锚定的回信收集。
/// 旧波次的回信不计入(混波免疫, failure-taxonomy #12 的机器侧)。
#[derive(Debug, Default)]
pub struct CollectGate {
    /// 审查店名(前缀匹配, 如 "review_")
    pub reviewers_prefix: String,
    pub arbitrator: String,
    pub min_replies: usize,
}

impl CollectGate {
    /// 返回 (已答店→结论, 是否收齐)。只统计 ts >= wave_ts 的审查店→仲裁者回信。
    pub fn collect(&self, letters: &[Letter], wave_ts: u64) -> (HashMap<String, Verdict>, bool) {
        let mut answered: HashMap<String, Verdict> = HashMap::new();
        for l in letters {
            if l.ts_ms < wave_ts
                || !l.from.starts_with(&self.reviewers_prefix)
                || l.to != self.arbitrator
                || l.body.starts_with("KICK=")
            {
                continue;
            }
            answered
                .entry(l.from.clone())
                .and_modify(|v| *v = classify(&l.body)) // 每店取该波次最后一封
                .or_insert_with(|| classify(&l.body));
        }
        let done = answered.len() >= self.min_replies;
        (answered, done)
    }
}

/// 机器裁决表: 从账本生成逐店结论 + 指定动作, 仲裁者照表执行。
/// (实测: judge 从"数分钟自创格式"变为"6秒逐字照表执行")
pub struct VerdictTable<'a> {
    pub gate: &'a CollectGate,
}

impl<'a> VerdictTable<'a> {
    pub fn render(&self, letters: &[Letter], wave_ts: u64) -> (String, usize) {
        let (answered, _) = self.gate.collect(letters, wave_ts);
        // 理由摘录: 该店本波最后一封回信的前48字 — 打回信靠它点名缺失检查,
        // 没有摘录时修正方在REJECT里找不到要补什么(确定性场景实测卡死3波)
        let arb = self.gate.arbitrator.clone();
        let last_body = |shop: &str| -> String {
            letters
                .iter()
                .rev()
                .find(|l| l.ts_ms >= wave_ts && l.from == shop && l.to == arb)
                // 取"||"之后的有效内容(跳过前缀), 64字 — 摘录截断在关键词中间
                // 会让修正方 contains() 匹配失败(实测卡死3波)
                .map(|l| {
                    let b = &l.body;
                    let core = b.split_once("||").map(|(_, r)| r).unwrap_or(b);
                    core.chars().take(64).collect::<String>()
                })
                .unwrap_or_default()
        };
        let mut nfix = 0;
        let mut lines = Vec::new();
        for (shop, v) in &answered {
            let mark = match v {
                Verdict::NeedsFix => {
                    nfix += 1;
                    "NEEDS_FIX"
                }
                Verdict::Clean => "CLEAN",
                Verdict::Unknown => "CLEAN?", // 未知按乐观处理, 完整回路校验兜底
            };
            lines.push(format!("{shop}: {mark} ({})", last_body(shop)));
        }
        lines.sort();
        (lines.join("\n"), nfix)
    }
}

/// 完整回路校验: REPORT 必须伴随 ≥min_fanout 家群发 + ≥min_replies 家回信,
/// 否则是假REPORT(进度备忘冒充收尾, failure-taxonomy #8)。
pub fn report_is_final(
    letters: &[Letter],
    judge: &str,
    reviewers_prefix: &str,
    min_fanout: usize,
    min_replies: usize,
) -> bool {
    let fan: std::collections::HashSet<_> = letters
        .iter()
        .filter(|l| l.from == judge && l.to.starts_with(reviewers_prefix))
        .map(|l| l.to.clone())
        .collect();
    let rep: std::collections::HashSet<_> = letters
        .iter()
        .filter(|l| {
            l.from.starts_with(reviewers_prefix) && l.to == judge && !l.body.starts_with("KICK=")
        })
        .map(|l| l.from.clone())
        .collect();
    fan.len() >= min_fanout && rep.len() >= min_replies
}

/// 登记处代投: 无论 judge 把 REJECT 写给了谁(实测写给过自己), 从账本取
/// 最新裁决内容代投给 writer。
pub fn extract_reject_for_relay(letters: &[Letter], judge: &str, after_ts: u64) -> Option<String> {
    letters
        .iter()
        .rev()
        .find(|l| {
            l.from == judge
                && l.ts_ms > after_ts
                && !l.body.starts_with("REPORT=")
                && (l.body.contains("REJECT") || l.body.contains("不通过"))
        })
        .map(|l| l.body.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l(ts: u64, from: &str, to: &str, body: &str) -> Letter {
        Letter {
            ts_ms: ts,
            seq: 0,
            from: from.into(),
            to: to.into(),
            body: body.into(),
        }
    }

    /// 回归测试 #14: 前缀漂移 — 自由格式结论按路由正确分类
    #[test]
    fn freeform_prefixes_classify_by_route() {
        let cases = [
            ("CLEAN=r1||ok", Verdict::Clean),
            ("REVIEW_XXX_RESULT 结论: 通过", Verdict::Clean),
            ("STATUS=BLOCK 底稿获取失败", Verdict::NeedsFix),
            ("review_plen 审查结果: 不通过", Verdict::NeedsFix),
            ("REVIEW=群审||请指示", Verdict::Unknown),
        ];
        for (body, want) in cases {
            assert_eq!(classify(body), want, "body={body}");
        }
    }

    /// 回归测试 #14b: 职责文本撞关键词 — 前缀优先, CLEAN 不因"失败"字样误判
    #[test]
    fn duty_text_keyword_collision() {
        assert_eq!(
            classify("CLEAN=review_return||失败返回路径存在: 已覆盖"),
            Verdict::Clean
        );
        assert_eq!(
            classify("NEEDS_FIX=review_x||成功路径检查: 缺失(try)"),
            Verdict::NeedsFix
        );
    }

    /// 回归测试 #12: 混波免疫 — 旧波次回信不计入收齐
    #[test]
    fn old_wave_replies_ignored() {
        let gate = CollectGate {
            reviewers_prefix: "r".into(),
            arbitrator: "judge".into(),
            min_replies: 2,
        };
        let letters = vec![
            l(100, "r1", "judge", "CLEAN"),       // 旧波
            l(100, "r2", "judge", "NEEDS_FIX=x"), // 旧波
            l(500, "r1", "judge", "CLEAN"),       // 新波
        ];
        let (ans, done) = gate.collect(&letters, 400);
        assert!(!done, "新波只有1家, 不应判收齐");
        assert_eq!(ans.len(), 1);
        assert_eq!(ans["r1"], Verdict::Clean);
    }

    /// 回归测试 #8: 假REPORT — 进度备忘不能当终局
    #[test]
    fn fake_report_rejected() {
        let letters = vec![
            l(1, "judge", "writer", "TASK=x"),
            l(2, "judge", "judge", "REPORT=已完成派单, 等待交稿"), // 假REPORT
        ];
        assert!(!report_is_final(&letters, "judge", "r", 8, 8));
    }

    /// 回归测试 #8 正向: 完整回路后的 REPORT 是终局
    #[test]
    fn real_report_accepted() {
        let mut letters = vec![l(1, "judge", "writer", "TASK=x")];
        for i in 1..=8 {
            letters.push(l(10, "judge", &format!("r{i}"), "REVIEW=群审"));
            letters.push(l(20, &format!("r{i}"), "judge", "CLEAN"));
        }
        letters.push(l(30, "judge", "judge", "REPORT=全部通过"));
        assert!(report_is_final(&letters, "judge", "r", 8, 8));
    }

    /// 回归测试 #10: 打回代投 — judge 把 REJECT 写给自己也能取出内容
    #[test]
    fn stray_reject_extracted() {
        let letters = vec![
            l(
                10,
                "judge",
                "judge",
                "JUDGEMENT=REJECT: 数据泄漏, 先split再fit",
            ),
            l(20, "judge", "review_x", "已转达"),
        ];
        let rej = extract_reject_for_relay(&letters, "judge", 5).unwrap();
        assert!(rej.contains("数据泄漏"));
    }

    /// 回归测试 #12: 裁决表 — nfix 计数与逐店行
    #[test]
    fn verdict_table_renders() {
        let gate = CollectGate {
            reviewers_prefix: "r".into(),
            arbitrator: "judge".into(),
            min_replies: 3,
        };
        let letters = vec![
            l(10, "r_a", "judge", "CLEAN"),
            l(11, "r_b", "judge", "不通过: 泄漏"),
            l(12, "r_c", "judge", "STATUS=BLOCK"),
        ];
        let vt = VerdictTable { gate: &gate };
        let (table, nfix) = vt.render(&letters, 5);
        assert_eq!(nfix, 2);
        assert!(table.contains("r_a: CLEAN"));
        assert!(table.contains("r_b: NEEDS_FIX"));
        assert!(table.contains("r_c: NEEDS_FIX"));
    }
}
