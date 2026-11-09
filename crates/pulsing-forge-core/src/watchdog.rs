//! 进度看门狗(Super-agent 吸收): 外部 Control Loop 的确定性 Fast Path。
//! 进程活着 ≠ 任务在推进 —— 看门狗从账本识别"无进展"模式并注入诊断性踢信。
//! 元原则: 阈值检测纯代码(Fast Path), 语义判断留给模型(Slow Path, 暂未实现)。

use crate::Letter;

/// 协议合法前缀: 这些信件是工作流本体, 不算杂信
const PROTOCOL_PREFIXES: &[&str] = &[
    "TASK=",
    "RESULT=",
    "REPORT=",
    "WORK=",
    "SENT=",
    "V=",
    "ARG=",
    "POEM=",
    "ECHO=",
    "ASK=",
    "CHECK=",
    "REVIEW=",
    "CLEAN=",
    "NEEDS_FIX=",
    "DRAFT=",
    "REVISED=",
    "REJECT=",
    "PART=",
    "VERIFY=",
    "RELAY=",
    "ARBIT=",
    "KICK=",
    "REMINDER",
    "ACK=",
];

#[derive(Debug, Clone, PartialEq)]
pub enum WatchdogAlert {
    /// 同构信件循环: 同一 (from,to) 上近乎相同的信重复 ≥ loop_threshold 次
    Loop {
        shop: String,
        to: String,
        count: usize,
        sample: String,
    },
    /// 杂信风暴: 单店在观察窗内发出 ≥ chatter_threshold 封非协议信(客套/确认/自言自语)
    Chatter {
        shop: String,
        count: usize,
        sample: String,
    },
}

#[derive(Debug, Clone)]
pub struct WatchdogConfig {
    /// 观察窗: 最近多少封信
    pub window: usize,
    /// 同构循环阈值
    pub loop_threshold: usize,
    /// 杂信阈值
    pub chatter_threshold: usize,
    /// 同构判定的正文前缀长度
    pub fingerprint_chars: usize,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        WatchdogConfig {
            window: 40,
            loop_threshold: 3,
            chatter_threshold: 6,
            fingerprint_chars: 48,
        }
    }
}

/// 扫描账本增量, 返回应注入的告警(每店每类至多一条, 自带幂等)
pub fn inspect(letters: &[Letter], cfg: &WatchdogConfig) -> Vec<WatchdogAlert> {
    let mut alerts = Vec::new();
    let tail: &[Letter] = if letters.len() > cfg.window {
        &letters[letters.len() - cfg.window..]
    } else {
        letters
    };

    // 只盯模型行为: 机器流量(编排器/看门狗自身/KICK信)豁免 —
    // 首版上线实锤: 把机器群审分发误判为循环, 又把自己的踢信当循环自踢12次
    let is_machine = |l: &Letter| {
        l.from == "orchestrator" || l.from == "watchdog" || l.body.starts_with("KICK=")
    };

    // 1) 同构循环: 按 (from, to, 正文指纹) 计数
    use std::collections::HashMap;
    let mut fp_count: HashMap<(String, String, String), usize> = HashMap::new();
    let mut fp_sample: HashMap<(String, String, String), String> = HashMap::new();
    for l in tail {
        if l.from == l.to || is_machine(l) {
            continue;
        } // 自回声已由工具层熔断
        let fp: String = l.body.chars().take(cfg.fingerprint_chars).collect();
        let key = (l.from.clone(), l.to.clone(), fp);
        *fp_count.entry(key.clone()).or_insert(0) += 1;
        fp_sample
            .entry(key)
            .or_insert_with(|| l.body.chars().take(60).collect());
    }
    let mut loop_shops: Vec<&(String, String, String)> = fp_count
        .iter()
        .filter(|(_, &c)| c >= cfg.loop_threshold)
        .map(|(k, _)| k)
        .collect();
    loop_shops.sort(); // 确定性输出
    for key in loop_shops {
        alerts.push(WatchdogAlert::Loop {
            shop: key.0.clone(),
            to: key.1.clone(),
            count: fp_count[key],
            sample: fp_sample[key].clone(),
        });
    }

    // 2) 杂信风暴: 单店非协议前缀信件计数
    let mut chatter: HashMap<String, (usize, String)> = HashMap::new();
    for l in tail {
        if l.from == l.to || is_machine(l) {
            continue;
        }
        if PROTOCOL_PREFIXES.iter().any(|p| l.body.starts_with(p)) {
            continue;
        }
        let e = chatter.entry(l.from.clone()).or_insert((0, String::new()));
        e.0 += 1;
        if e.1.is_empty() {
            e.1 = l.body.chars().take(60).collect();
        }
    }
    let mut ch_shops: Vec<_> = chatter
        .iter()
        .filter(|(_, (c, _))| *c >= cfg.chatter_threshold)
        .collect();
    ch_shops.sort();
    for (shop, (c, sample)) in ch_shops {
        alerts.push(WatchdogAlert::Chatter {
            shop: shop.clone(),
            count: *c,
            sample: sample.clone(),
        });
    }
    alerts
}

/// 告警 → 诊断性踢信正文(教学式: 告诉模型错在哪、该干什么)
pub fn kick_body(a: &WatchdogAlert) -> String {
    match a {
        WatchdogAlert::Loop { shop, to, count, sample } => format!(
            "KICK=看门狗: 检测到同构信件循环(你向{to}发送了{count}封几乎相同的信: \"{sample}…\")。\
             立即停止重复发送。回顾你的规程信, 执行其中尚未完成的唯一动作; 若任务已无从推进, \
             调用 pulsing_send(to='{}', silent=true, message='REPORT=STALLED||卡点说明') 如实上报后结束。禁止再发同构信。",
            shop
        ),
        WatchdogAlert::Chatter { shop, count, sample } => format!(
            "KICK=看门狗: 检测到杂信风暴(你发出了{count}封非协议信件, 如 \"{sample}…\")。\
             协议只认 TASK/RESULT/REPORT/WORK/DRAFT/CLEAN/VERIFY 等前缀信。\
             立即停止客套/确认/解释类信件; 回到任务本身, 按规程信的指示调用工具; \
             无事可做则直接结束回合不发信。"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn l(seq: u64, from: &str, to: &str, body: &str) -> Letter {
        Letter {
            ts_ms: seq,
            seq,
            from: from.into(),
            to: to.into(),
            body: body.into(),
        }
    }

    #[test]
    fn detects_ack_ping_pong() {
        // 实测形态: judge/writer 互发"已收到触发信"(taxonomy #4 ACK乒乓)
        let mut ls = vec![l(1, "orchestrator", "judge", "TASK=开始")];
        for i in 0..8 {
            ls.push(l(
                2 + i * 2,
                "judge",
                "writer",
                "judge 已收到触发信。信件未指定动作。",
            ));
            ls.push(l(
                3 + i * 2,
                "writer",
                "judge",
                "writer 已收到触发信。信件未指定动作。",
            ));
        }
        let alerts = inspect(&ls, &WatchdogConfig::default());
        assert!(
            alerts
                .iter()
                .any(|a| matches!(a, WatchdogAlert::Loop { shop, .. } if shop == "judge")),
            "{alerts:?}"
        );
        assert!(
            alerts
                .iter()
                .any(|a| matches!(a, WatchdogAlert::Loop { shop, .. } if shop == "writer")),
            "{alerts:?}"
        );
    }

    #[test]
    fn healthy_journal_no_alert() {
        let ls = vec![
            l(1, "orchestrator", "judge", "TASK=开始"),
            l(2, "judge", "writer", "TASK=写clamp"),
            l(3, "writer", "judge", "DRAFT=```python def clamp```"),
            l(4, "judge", "review_r1", "REVIEW=群审"),
            l(5, "review_r1", "judge", "CLEAN=r1||ok"),
        ];
        assert!(inspect(&ls, &WatchdogConfig::default()).is_empty());
    }

    #[test]
    fn machine_traffic_never_alerts() {
        // 上线实锤回归: 机器群审分发(3波×3审查店, 指纹相近) + 看门狗自己的踢信
        // 首版把它们误判为循环并自踢 — 机器流量必须全程豁免
        let mut ls = vec![l(1, "orchestrator", "judge", "TASK=开始")];
        for wave in 0..3 {
            for r in ["review_r1", "review_r2", "review_r3"] {
                ls.push(l(
                    10 + wave * 3 + r.len() as u64,
                    "orchestrator",
                    r,
                    "REVIEW=群审||你只查||底稿如下||DRAFT=clamp全文",
                ));
            }
        }
        for i in 0..5 {
            ls.push(l(
                100 + i,
                "watchdog",
                "writer",
                "KICK=看门狗: 检测到同构信件循环",
            ));
        }
        assert!(
            inspect(&ls, &WatchdogConfig::default()).is_empty(),
            "机器分发与踢信不得触发告警"
        );
    }

    #[test]
    fn kick_body_is_educational() {
        let a = WatchdogAlert::Chatter {
            shop: "writer".into(),
            count: 7,
            sample: "已收到".into(),
        };
        let b = kick_body(&a);
        assert!(b.starts_with("KICK=看门狗:"));
        assert!(b.contains("停止"));
    }
}
