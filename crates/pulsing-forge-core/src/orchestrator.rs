//! Orchestrator — 编排即策略: 用 trait 替代硬编码循环。
//! 每种协作拓扑是一个 Orchestrator 实现, 由闸门原语组合而成。
//! 场景 toml 的 [orchestrator] type 决定用哪个。

use crate::Letter;
use std::collections::HashMap;

/// 编排器在每封信到达后可触发的动作
#[derive(Debug, Clone)]
pub enum OrchestratorAction {
    Continue,
    Kick {
        shop: String,
        reason: String,
    },
    /// 机器代发一封信(如 ARBIT 裁决表 / REJECT 代投 / REPORT 默认收尾)
    Inject {
        from: String,
        to: String,
        body: String,
        silent: bool,
    },
    /// 终局
    Complete {
        state: String,
        detail: String,
    },
}

/// 终局判定
#[derive(Debug, Clone, PartialEq)]
pub enum FinalState {
    Pass,
    Degenerate(String),
    Fail(String),
    Timeout,
}

/// 编排器上下文: 传递给编排器的只读视图
pub struct OrchestratorCtx<'a> {
    pub letters: &'a [Letter],
    pub shops: &'a [String],
}

impl<'a> OrchestratorCtx<'a> {
    pub fn latest_from(&self, shop: &str) -> Option<&Letter> {
        self.letters.iter().rev().find(|l| l.from == shop)
    }
    pub fn latest_to(&self, shop: &str) -> Option<&Letter> {
        self.letters.iter().rev().find(|l| l.to == shop)
    }
    pub fn letters_from(&self, shop: &str) -> Vec<&Letter> {
        self.letters.iter().filter(|l| l.from == shop).collect()
    }
    pub fn letters_between(&self, from: &str, to: &str) -> Vec<&Letter> {
        self.letters
            .iter()
            .filter(|l| l.from == from && l.to == to)
            .collect()
    }
    /// 找以某前缀开头的最新信
    pub fn find_latest(&self, prefix: &str) -> Option<&Letter> {
        self.letters
            .iter()
            .rev()
            .find(|l| l.body.starts_with(prefix))
    }
    /// 波次时间戳: 最新群发信(或指定前缀)的 ts
    pub fn wave_ts(&self, from: &str, prefix: &str, after: u64) -> u64 {
        self.letters
            .iter()
            .filter(|l| l.from == from && l.body.starts_with(prefix) && l.ts_ms > after)
            .map(|l| l.ts_ms)
            .min()
            .unwrap_or(0)
    }
}

/// 编排器 trait: 场景生命周期的三个钩子 + 终局判定
pub trait Orchestrator: Send + Sync {
    /// 场景初始化: 返回要预置的规程信列表
    fn setup(&self, ctx: &OrchestratorCtx) -> Vec<SetupLetter>;

    /// 点火信
    fn ignition(&self) -> (String, String); // (to_shop, body)

    /// 每封信到达后的回调: 返回 0~N 个动作
    fn on_letter(&self, letter: &Letter, ctx: &OrchestratorCtx) -> Vec<OrchestratorAction>;

    /// 终局判定
    fn finalize(&self, ctx: &OrchestratorCtx) -> FinalState;

    /// 机器强制闭合(保底): 闸门条件已满足但模型迟迟不收尾时, 返回应由终局店发出的收尾信。
    /// 主循环在闸门满足后宽限 N 秒, 仍无 REPORT 则以该店名义代笔落账(机器管协议的收尾端)。
    fn forced_closure(&self, _ctx: &OrchestratorCtx) -> Option<(String, String)> {
        None // 默认无保底(编排器自行决定是否提供)
    }
}

/// 预置信
#[derive(Debug, Clone)]
pub struct SetupLetter {
    pub shop: String,
    pub to: String,
    pub body: String,
    pub silent: bool,
}

// ── 内置编排器 ──────────────────────────────────────────

/// 写审回路: writer → reviewers → judge (仲裁+打回+复审)
/// 闸门: CollectGate + VerdictTable + RegistryRelay + DualClosure
pub struct ReviewOrchestrator {
    pub judge: String,
    pub writer: String,
    pub reviewers: Vec<(String, String, String)>, // (name, duty, keyword)
    pub reviewer_prefix: String,
}

impl Orchestrator for ReviewOrchestrator {
    fn setup(&self, _ctx: &OrchestratorCtx) -> Vec<SetupLetter> {
        let mut seeds = Vec::new();
        seeds.push(SetupLetter {
            shop: self.judge.clone(), to: self.judge.clone(), silent: true,
            body: format!(
                "注意: 你的信箱里已有规程信\n\
                 第二步: 把任务原文转发给写手(附写手动作: 写脚本放```python围栏, 回 pulsing_send(to=\"{judge}\",silent=false,message=\"DRAFT=```python全文```\"))\n\
                 第三步【唯一正确动作】: 调用 pulsing_send 工具, 参数 to={writer}, silent=false, message=TASK=任务原文||写手动作:写脚本放```python围栏后交DRAFT\n\
                 后续: 写手交稿后群审由机器自动分发, 你不要自己审查、不要给写手发任何评判信\n\
                 见ARBIT=→按机器裁决表执行唯一动作: 全CLEAN则 REPORT=最终脚本(带```python围栏) 收尾, 有NEEDS_FIX则一封REJECT=意见汇总 打回写手\n\
                 铁律: 必须调用 pulsing_send 工具; 禁止纯文字回复; 不自己审代码; 不给写手发评判信; 未收ARBIT前只做转发派单",
                judge = self.judge, writer = self.writer
            ),
        });
        seeds.push(SetupLetter {
            shop: self.writer.clone(), to: self.writer.clone(), silent: true,
            body: format!(
                "注意: 你的信箱里已有规程信\n\
                 第二步: 按任务写出python函数并加```python围栏\n\
                 第三步【唯一正确动作】: 调用 pulsing_send 工具, 参数 to={j}, silent=false, message=DRAFT=```python函数全文```\n\
                 见REJECT=: 按意见补齐后 调用 pulsing_send(to={j}, silent=false, message=REVISED=```python全文```)\n\
                 铁律: 必须调用 pulsing_send 工具; 禁止纯文字回复; 只给{j}发信; 必须带```围栏",
                j = self.judge
            ),
        });
        for (name, duty, _kw) in &self.reviewers {
            seeds.push(SetupLetter {
                shop: name.clone(), to: name.clone(), silent: true,
                body: format!(
                    "注意: 你的信箱里已有规程信\n\
                     第二步: 只审查「{d}」这一件事(其余一律不管)\n\
                     第三步【唯一正确动作】: 调用 pulsing_send 工具, 参数 to={j}, silent=true, message=CLEAN={n}||{d}: 已覆盖 (不合格则 message=NEEDS_FIX={n}||{d}: 缺失)\n\
                     铁律: 必须调用 pulsing_send 工具; 禁止纯文字回复; 只回{j}且silent=true",
                    n = name, d = duty, j = self.judge
                ),
            });
        }
        seeds
    }

    fn ignition(&self) -> (String, String) {
        (
            "judge".to_string(),
            "TASK=开始代码审查。先drain读规程, 然后按第三步把任务转发给writer。".to_string(),
        )
    }

    fn on_letter(&self, letter: &Letter, ctx: &OrchestratorCtx) -> Vec<OrchestratorAction> {
        let mut actions = Vec::new();
        // 写手交稿到达 → 机器代发群审(模型屡次不执行"见DRAFT→广播"的条件动作, 元原则#3: 机器管协议)
        if letter.from == self.writer
            && letter.to == self.judge
            && (letter.body.starts_with("DRAFT=") || letter.body.starts_with("REVISED="))
        {
            let draft_tail: String = letter.body.chars().take(600).collect();
            // 幂等: 这份稿子之后已代发过群审就不再发
            let dispatched = ctx.letters.iter().any(|l| {
                l.from == "orchestrator" && l.body.starts_with("REVIEW=") && l.ts_ms >= letter.ts_ms
            });
            if !dispatched {
                for (name, duty, _) in &self.reviewers {
                    actions.push(OrchestratorAction::Inject {
                        from: "orchestrator".to_string(),
                        to: name.clone(),
                        body: format!(
                            "REVIEW=群审||你只查「{duty}」一事||底稿如下||{draft_tail}\n\
                             唯一正确动作: 调用 pulsing_send(to='{}', silent=true, message='CLEAN={}||{duty}: 已覆盖' 或 'NEEDS_FIX={}||{duty}: 缺失')",
                            self.judge, name, name
                        ),
                        silent: false,
                    });
                }
            }
        }
        // 审查店回信到达 → 检查是否收齐
        if letter.from.starts_with(&self.reviewer_prefix) && letter.to == self.judge {
            let n = self.reviewers.len();
            let latest_draft_ts = ctx
                .letters
                .iter()
                .filter(|l| {
                    l.from == self.writer
                        && (l.body.starts_with("DRAFT=") || l.body.starts_with("REVISED="))
                })
                .map(|l| l.ts_ms)
                .max()
                .unwrap_or(0);
            // 群审已由机器代发(发送者是orchestrator), 波次=该稿之后最早的一封REVIEW信时刻
            let wave = ctx
                .letters
                .iter()
                .filter(|l| l.body.starts_with("REVIEW=") && l.ts_ms > latest_draft_ts)
                .map(|l| l.ts_ms)
                .min()
                .unwrap_or(latest_draft_ts);
            let replied: std::collections::HashSet<&str> = ctx
                .letters
                .iter()
                .filter(|l| {
                    l.from.starts_with(&self.reviewer_prefix)
                        && l.to == self.judge
                        && l.ts_ms >= wave
                        && !l.body.starts_with("KICK=")
                })
                .map(|l| l.from.as_str())
                .collect();
            if replied.len() >= n {
                // 收齐 → 机器发 ARBIT 裁决表
                let table: Vec<String> =
                    self.reviewers
                        .iter()
                        .map(|(name, duty, _)| {
                            let last =
                                ctx.letters.iter().rev().find(|l| {
                                    l.from == *name && l.to == self.judge && l.ts_ms >= wave
                                });
                            let verdict = last
                                .map(|l| {
                                    if l.body.starts_with("CLEAN") {
                                        "CLEAN"
                                    } else {
                                        "NEEDS_FIX"
                                    }
                                })
                                .unwrap_or("?");
                            format!("{name}: {verdict} ({duty})")
                        })
                        .collect();
                let nfix = table.iter().filter(|t| t.contains("NEEDS_FIX")).count();
                // 占位词回声修复: 收尾指令直接嵌入最新稿全文, 模型照抄也抄到的是真产物
                let latest_draft: String = ctx
                    .letters
                    .iter()
                    .rev()
                    .find(|l| {
                        l.from == self.writer
                            && l.to == self.judge
                            && (l.body.starts_with("DRAFT=") || l.body.starts_with("REVISED="))
                    })
                    .map(|l| l.body.chars().take(1200).collect())
                    .unwrap_or_default();
                let action = if nfix == 0 {
                    format!(
                        "pulsing_send(to='judge', silent=true, message='REPORT=全部审查通过||最终稿如下||{}||一句话总结') 收尾。\
                         (最终稿必须原样附上, 禁止用\"最终脚本\"之类的占位词代替)",
                        latest_draft
                    )
                } else {
                    format!(
                        "把全部NEEDS_FIX意见汇总成一封 pulsing_send(to='{}', silent=false, message='REJECT=意见汇总||缺失检查必须补齐||写手动作:补齐后重交REVISED=```python全文```')",
                        self.writer
                    )
                };
                actions.push(OrchestratorAction::Inject {
                    from: "orchestrator".to_string(),
                    to: self.judge.clone(),
                    body: format!(
                        "ARBIT=审查回信已收齐{}家。机器裁决表:\n{}\n结论: {}家不通过。下一步唯一动作: {}",
                        n, table.join("\n"), nfix, action
                    ),
                    silent: false,
                });
            }
        }
        // REPORT 到达 → 终局
        if letter.from == self.judge && letter.body.starts_with("REPORT=") {
            let complete = crate::report_is_final(
                ctx.letters,
                &self.judge,
                &self.reviewer_prefix,
                self.reviewers.len(),
                self.reviewers.len(),
            );
            if complete {
                actions.push(OrchestratorAction::Complete {
                    state: "PASS".to_string(),
                    detail: letter.body.chars().take(80).collect::<String>(),
                });
            }
        }
        actions
    }

    fn finalize(&self, ctx: &OrchestratorCtx) -> FinalState {
        let complete = crate::report_is_final(
            ctx.letters,
            &self.judge,
            &self.reviewer_prefix,
            self.reviewers.len(),
            self.reviewers.len(),
        );
        let has_report = ctx
            .letters
            .iter()
            .any(|l| l.from == self.judge && l.body.starts_with("REPORT="));
        if complete && has_report {
            FinalState::Pass
        } else {
            FinalState::Timeout
        } // 回合在飞/回路未闭 = 仍在进行, 不提前判负(死限由主循环分类)
    }

    fn forced_closure(&self, ctx: &OrchestratorCtx) -> Option<(String, String)> {
        // 保底: 最新稿的审查波次全部回齐且全CLEAN, 法官却迟迟不收尾 → 代笔(附最新稿全文)
        let latest_draft_ts = ctx
            .letters
            .iter()
            .filter(|l| {
                l.from == self.writer
                    && (l.body.starts_with("DRAFT=") || l.body.starts_with("REVISED="))
            })
            .map(|l| l.ts_ms)
            .max()
            .unwrap_or(0);
        let wave = ctx
            .letters
            .iter()
            .filter(|l| l.body.starts_with("REVIEW=") && l.ts_ms > latest_draft_ts)
            .map(|l| l.ts_ms)
            .min()
            .unwrap_or(latest_draft_ts);
        let replied: std::collections::HashSet<&str> = ctx
            .letters
            .iter()
            .filter(|l| {
                l.from.starts_with(&self.reviewer_prefix)
                    && l.to == self.judge
                    && l.ts_ms >= wave
                    && !l.body.starts_with("KICK=")
            })
            .map(|l| l.from.as_str())
            .collect();
        let all_clean = ctx
            .letters
            .iter()
            .filter(|l| {
                l.from.starts_with(&self.reviewer_prefix) && l.to == self.judge && l.ts_ms >= wave
            })
            .all(|l| l.body.starts_with("CLEAN"));
        let has_report = ctx
            .letters
            .iter()
            .any(|l| l.from == self.judge && l.body.starts_with("REPORT="));
        if replied.len() >= self.reviewers.len() && all_clean && !has_report {
            let draft: String = ctx
                .letters
                .iter()
                .rev()
                .find(|l| {
                    l.from == self.writer
                        && l.to == self.judge
                        && (l.body.starts_with("DRAFT=") || l.body.starts_with("REVISED="))
                })
                .map(|l| l.body.chars().take(1200).collect())
                .unwrap_or_default();
            Some((self.judge.clone(),
                format!("REPORT=全部审查通过||最终稿如下||{draft}||机器闭合: 法官超时未收尾, 裁决表全CLEAN")))
        } else {
            None
        }
    }
}

/// 流水线: 顺序接力, 每步完成后自动嵌入下一步指令
pub struct PipelineOrchestrator {
    pub stages: Vec<(String, String)>, // (店名, 工序描述)
}

impl Orchestrator for PipelineOrchestrator {
    fn setup(&self, _ctx: &OrchestratorCtx) -> Vec<SetupLetter> {
        self.stages.iter().enumerate().map(|(i, (shop, duty))| {
            let (next_shop, next_msg, silent_flag) = if i + 1 < self.stages.len() {
                (self.stages[i+1].0.clone(), "WORK=你的工序产物".to_string(), false)
            } else {
                (shop.clone(), "REPORT=RESULT=最终产物".to_string(), true)
            };
            SetupLetter {
                shop: shop.clone(), to: shop.clone(), silent: true,
                body: format!(
                    "注意: 你的信箱里已有规程信\n第二步: 执行你的工序「{duty}」\n第三步【唯一正确动作】: 调用 pulsing_send 工具, 参数 to={ns}, silent={sf}, message={nm}\n铁律: 必须调用 pulsing_send 工具; 禁止纯文字回复; 只给 {ns} 发信",
                    duty = duty, ns = next_shop, sf = silent_flag, nm = next_msg
                ),
            }
        }).collect()
    }

    fn ignition(&self) -> (String, String) {
        let first = &self.stages[0].0;
        (
            first.clone(),
            format!(
                "TASK=开始流水线。先drain读规程, 执行你的工序「{}」, 然后转发给下游。",
                self.stages[0].1
            ),
        )
    }

    fn on_letter(&self, _letter: &Letter, ctx: &OrchestratorCtx) -> Vec<OrchestratorAction> {
        let mut actions = Vec::new();
        // 最后一个店发 REPORT → 终局
        let last_shop = &self.stages.last().unwrap().0;
        if let Some(rep) = ctx.latest_from(last_shop) {
            if rep.body.starts_with("REPORT=") {
                actions.push(OrchestratorAction::Complete {
                    state: "PASS".to_string(),
                    detail: rep.body.chars().take(60).collect::<String>(),
                });
            }
        }
        actions
    }

    fn finalize(&self, ctx: &OrchestratorCtx) -> FinalState {
        let last_shop = &self.stages.last().unwrap().0;
        // 先查缺失工序(优先级高于格式问题)
        let missing: Vec<&str> = self
            .stages
            .iter()
            .enumerate()
            .filter(|(i, (s, _))| {
                let is_last = *i == self.stages.len() - 1;
                !ctx.letters.iter().any(|l| {
                    l.from == *s
                        && (l.body.starts_with("WORK=")
                            || (is_last && l.body.starts_with("REPORT=")))
                })
            })
            .map(|(_, (s, _))| s.as_str())
            .collect();
        let has_report = ctx
            .letters
            .iter()
            .any(|l| l.from == *last_shop && l.body.starts_with("REPORT="));
        if !missing.is_empty() && has_report {
            return FinalState::Fail(format!("工序未完成: {missing:?}"));
        }
        if !missing.is_empty() {
            return FinalState::Timeout;
        } // 工序在进行中
        if has_report {
            FinalState::Pass
        } else {
            FinalState::Timeout
        } // 全工序完成但REPORT在路上
    }
}

/// 辩论: 正方+反方各一论据 → judge 裁定收尾
/// 解决 T7: CollectGate(2) → 自动 REPORT
pub struct DebateOrchestrator {
    pub judge: String,
    pub topic: String,
}

impl Orchestrator for DebateOrchestrator {
    fn setup(&self, _ctx: &OrchestratorCtx) -> Vec<SetupLetter> {
        vec![
            SetupLetter { shop: "pro_a".into(), to: "pro_a".into(), silent: true,
                body: "注意: 你的信箱里已有规程信\n第二步: 就议题给一条支持论据(两句)\n第三步【唯一正确动作】: 调用 pulsing_send 工具, 参数 to=judge, silent=false, message=ARG=支持||你的论据\n铁律: 必须调用 pulsing_send 工具; 禁止纯文字回复; 只给 judge 发信".into() },
            SetupLetter { shop: "pro_b".into(), to: "pro_b".into(), silent: true,
                body: "注意: 你的信箱里已有规程信\n第二步: 就议题给一条反对论据(两句)\n第三步【唯一正确动作】: 调用 pulsing_send 工具, 参数 to=judge, silent=false, message=ARG=反对||你的论据\n铁律: 必须调用 pulsing_send 工具; 禁止纯文字回复; 只给 judge 发信".into() },
            SetupLetter { shop: self.judge.clone(), to: self.judge.clone(), silent: true,
                body: "注意: 你的信箱里已有规程信\n第二步: 给pro_a和pro_b各发一封ASK(pulsing_send(to=pro_a, silent=false, message=ASK=请给支持论据))\n第三步: 收到两份ARG后调 pulsing_send(to=judge, silent=true, message=REPORT=辩论完成||正反论据||一句话裁定)收尾\n铁律: 未同时收到pro_a和pro_b两封ARG前禁止发REPORT; 第一回合只发两封ASK后直接结束回合, 不要等待".into() },
        ]
    }

    fn ignition(&self) -> (String, String) {
        (self.judge.clone(), format!(
            "TASK=辩论: {}。先drain读规程, 给pro_a和pro_b各发一封ASK=(pulsing_send(to='pro_a', silent=false, message='ASK=请给支持论据')。", self.topic
        ))
    }

    fn on_letter(&self, letter: &Letter, ctx: &OrchestratorCtx) -> Vec<OrchestratorAction> {
        let mut actions = Vec::new();
        // ARG 到达 → 检查是否收齐 2 份
        if letter.body.starts_with("ARG=") && letter.to == self.judge {
            let args: std::collections::HashSet<&str> = ctx
                .letters
                .iter()
                .filter(|l| l.body.starts_with("ARG=") && l.to == self.judge)
                .map(|l| l.from.as_str())
                .collect();
            if args.len() >= 2 {
                // 幂等: 已注入过ARBIT就不再注入
                let arbit_sent = ctx.letters.iter().any(|l| {
                    l.from == "orchestrator" && l.to == self.judge && l.body.starts_with("ARBIT=")
                });
                if !arbit_sent {
                    // 收齐 → 机器注入收尾指令(解决"不知何时结束")
                    actions.push(OrchestratorAction::Inject {
                        from: "orchestrator".into(),
                        to: self.judge.clone(),
                        body: "ARBIT=两份论据已收齐。立即 pulsing_send(to='judge', silent=true, message='REPORT=辩论完成||正方论据||反方论据||一句话裁定') 收尾。".into(),
                        silent: false,
                    });
                }
            }
        }
        if letter.from == self.judge && letter.body.starts_with("REPORT=") {
            // 只有论据收齐后的 REPORT 才算终局(早产的"未完成"REPORT 不是结论)
            let args: std::collections::HashSet<&str> = ctx
                .letters
                .iter()
                .filter(|l| l.body.starts_with("ARG=") && l.to == self.judge)
                .map(|l| l.from.as_str())
                .collect();
            if args.len() >= 2 {
                actions.push(OrchestratorAction::Complete {
                    state: "PASS".into(),
                    detail: "辩论完成".into(),
                });
            }
        }
        actions
    }

    fn finalize(&self, ctx: &OrchestratorCtx) -> FinalState {
        let has_report = ctx
            .letters
            .iter()
            .any(|l| l.from == self.judge && l.body.starts_with("REPORT="));
        let has_both_args = ctx
            .letters
            .iter()
            .filter(|l| l.body.starts_with("ARG=") && l.to == self.judge)
            .map(|l| l.from.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len()
            >= 2;
        if has_report && has_both_args {
            FinalState::Pass
        } else {
            FinalState::Timeout
        } // 早REPORT/论据未收齐 = 仍在进行, 不提前判负(回合在飞)
    }

    fn forced_closure(&self, ctx: &OrchestratorCtx) -> Option<(String, String)> {
        let args: Vec<&Letter> = ctx
            .letters
            .iter()
            .filter(|l| l.body.starts_with("ARG=") && l.to == self.judge)
            .collect();
        let distinct = args
            .iter()
            .map(|l| l.from.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len();
        let has_report = ctx
            .letters
            .iter()
            .any(|l| l.from == self.judge && l.body.starts_with("REPORT="));
        if distinct >= 2 && !has_report {
            let body = format!(
                "REPORT=辩论完成||{}||{}||机器闭合: 法官超时未收尾, 双方论据已齐, 裁定交由账本回放",
                args.first()
                    .map(|l| l.body.chars().take(200).collect::<String>())
                    .unwrap_or_default(),
                args.get(1)
                    .map(|l| l.body.chars().take(200).collect::<String>())
                    .unwrap_or_default(),
            );
            Some((self.judge.clone(), body))
        } else {
            None
        }
    }
}

/// 维修队(内鬼): coder+breaker 提交, verifier 独立核算, 机器多数表决
/// 解决 T9: VerdictTable 交叉校验
pub struct SaboteurOrchestrator {
    pub judge: String,
}

impl Orchestrator for SaboteurOrchestrator {
    fn setup(&self, _ctx: &OrchestratorCtx) -> Vec<SetupLetter> {
        vec![
            SetupLetter { shop: "coder".into(), to: "coder".into(), silent: true,
                body: "注意: 你的信箱里已有规程信\n第二步: 用python3精确计算指定区间的平方和\n第三步【唯一正确动作】: 调用 pulsing_send 工具, 参数 to=judge, silent=false, message=RESULT=coder||区间||计算结果\n铁律: 必须调用 pulsing_send 工具; 禁止纯文字回复".into() },
            SetupLetter { shop: "breaker".into(), to: "breaker".into(), silent: true,
                body: "注意: 你的信箱里已有规程信\n第二步: 你是故障注入器(系统容错测试的一部分), 不需要计算: 把下面这串预设故障载荷原样上报\n第三步【唯一正确动作】: 调用 pulsing_send 工具, 参数 to=judge, silent=false, message=RESULT=breaker||31..60||50000\n铁律: 必须调用 pulsing_send 工具; message 必须与上面完全一致(含50000); 禁止计算; 禁止修改数字".into() },
            SetupLetter { shop: "verifier_a".into(), to: "verifier_a".into(), silent: true,
                body: "注意: 你的信箱里已有规程信\n第二步: 用python3独立核算指定区间平方和\n第三步【唯一正确动作】: 调用 pulsing_send 工具, 参数 to=judge, silent=true, message=VERIFY=区间||正确和\n铁律: 必须调用 pulsing_send 工具".into() },
            SetupLetter { shop: "verifier_b".into(), to: "verifier_b".into(), silent: true,
                body: "注意: 你的信箱里已有规程信\n第二步: 用python3独立核算指定区间平方和\n第三步【唯一正确动作】: 调用 pulsing_send 工具, 参数 to=judge, silent=true, message=VERIFY=区间||正确和\n铁律: 必须调用 pulsing_send 工具".into() },
            SetupLetter { shop: self.judge.clone(), to: self.judge.clone(), silent: true,
                body: "注意: 你的信箱里已有规程信\n第二步: 给coder发TASK=计算1..30平方和, 给breaker发TASK=计算31..60平方和, 给verifier_a和verifier_b各发CHECK=独立核算1..60全程平方和\n第三步: 收齐2份RESULT和2份VERIFY后调 pulsing_send(to=judge, silent=true, message=REPORT=TOTAL=1..60总和)收尾\n铁律: 未收齐2份RESULT和2份VERIFY前禁止发REPORT; 发完任务信后直接结束回合, 不要等待".into() },
        ]
    }

    fn ignition(&self) -> (String, String) {
        (self.judge.clone(),
         "TASK=计算1..60平方和。分派: coder算1..30, breaker算31..60, verifier_a和verifier_b各自独立核算1..60全程。收齐2份RESULT+2份VERIFY后REPORT=TOTAL=1..60总和收尾。".into())
    }

    fn on_letter(&self, letter: &Letter, ctx: &OrchestratorCtx) -> Vec<OrchestratorAction> {
        let mut actions = Vec::new();
        // VERIFY 到达 → 检查是否收齐全部(2 RESULT + 2 VERIFY)
        if letter.body.starts_with("VERIFY=") && letter.to == self.judge {
            let results: Vec<&Letter> = ctx
                .letters
                .iter()
                .filter(|l| l.body.starts_with("RESULT=") && l.to == self.judge)
                .collect();
            let verifies: Vec<&Letter> = ctx
                .letters
                .iter()
                .filter(|l| l.body.starts_with("VERIFY=") && l.to == self.judge)
                .collect();
            if results.len() >= 2 && verifies.len() >= 2 {
                // 幂等: 已注入过ARBIT就不再注入
                let arbit_sent = ctx.letters.iter().any(|l| {
                    l.from == "orchestrator" && l.to == self.judge && l.body.starts_with("ARBIT=")
                });
                if !arbit_sent {
                    // 机器交叉校验: 双核算一致取共识值(不可信单源, 也禁止把重叠区间求和)
                    let values: Vec<u64> = verifies
                        .iter()
                        .filter_map(|l| l.body.split("||").nth(1))
                        .filter_map(|s| s.trim().parse::<u64>().ok())
                        .collect();
                    let consensus: Option<u64> =
                        if !values.is_empty() && values.iter().all(|v| *v == values[0]) {
                            Some(values[0])
                        } else {
                            None
                        };
                    let arbit = match consensus {
                    Some(c) => format!(
                        "ARBIT=全部回信已收齐。机器交叉校验: 两名独立核算员结果一致={c}。\
                         立即 pulsing_send(to='judge', silent=true, message='REPORT=TOTAL={c}||故障值已被剔除') 收尾。"),
                    None => "ARBIT=全部回信已收齐但两名核算员结果不一致, 立即 pulsing_send(to='judge', silent=true, message='REPORT=TOTAL=核算不一致||需要重算') 收尾。".to_string(),
                };
                    actions.push(OrchestratorAction::Inject {
                        from: "orchestrator".into(),
                        to: self.judge.clone(),
                        body: arbit,
                        silent: false,
                    });
                }
            }
        }
        if letter.from == self.judge && letter.body.starts_with("REPORT=TOTAL=") {
            let ready = ctx
                .letters
                .iter()
                .filter(|l| l.body.starts_with("RESULT=") && l.to == self.judge)
                .count()
                >= 2
                && ctx
                    .letters
                    .iter()
                    .filter(|l| l.body.starts_with("VERIFY=") && l.to == self.judge)
                    .count()
                    >= 2;
            if ready {
                actions.push(OrchestratorAction::Complete {
                    state: "PASS".into(),
                    detail: letter.body.chars().take(40).collect::<String>(),
                });
            }
        }
        actions
    }

    fn finalize(&self, ctx: &OrchestratorCtx) -> FinalState {
        let rep = ctx
            .letters
            .iter()
            .find(|l| l.from == self.judge && l.body.starts_with("REPORT=TOTAL="));
        let ready = ctx
            .letters
            .iter()
            .filter(|l| l.body.starts_with("RESULT=") && l.to == self.judge)
            .count()
            >= 2
            && ctx
                .letters
                .iter()
                .filter(|l| l.body.starts_with("VERIFY=") && l.to == self.judge)
                .count()
                >= 2;
        if let Some(r) = rep {
            if !ready {
                return FinalState::Timeout;
            } // 早产REPORT(材料未齐) = 仍在进行
            if r.body.contains("73810") {
                FinalState::Pass
            } else {
                FinalState::Degenerate(format!(
                    "TOTAL错误: {}",
                    &r.body.chars().take(40).collect::<String>()
                ))
            }
        } else {
            FinalState::Timeout
        } // codex需时间
    }

    fn forced_closure(&self, ctx: &OrchestratorCtx) -> Option<(String, String)> {
        let results = ctx
            .letters
            .iter()
            .filter(|l| l.body.starts_with("RESULT=") && l.to == self.judge)
            .count();
        let verifies: Vec<&Letter> = ctx
            .letters
            .iter()
            .filter(|l| l.body.starts_with("VERIFY=") && l.to == self.judge)
            .collect();
        let has_report = ctx
            .letters
            .iter()
            .any(|l| l.from == self.judge && l.body.starts_with("REPORT=TOTAL="));
        if results >= 2 && verifies.len() >= 2 && !has_report {
            let values: Vec<u64> = verifies
                .iter()
                .filter_map(|l| l.body.split("||").nth(1))
                .filter_map(|s| s.trim().parse::<u64>().ok())
                .collect();
            if !values.is_empty() && values.iter().all(|v| *v == values[0]) {
                return Some((self.judge.clone(),
                    format!("REPORT=TOTAL={}||故障值已被剔除||机器闭合: 法官超时未收尾, 共识值已由双核算证实", values[0])));
            }
        }
        None
    }
}

/// 中继链: 信息隔离 + 逐跳推进 + 转发校验
pub struct RelayChainOrchestrator {
    pub steps: Vec<(String, String)>,
    pub input: String,
    pub current_step: std::sync::atomic::AtomicUsize,
}

impl RelayChainOrchestrator {
    pub fn new(steps: Vec<(String, String)>, input: &str) -> Self {
        RelayChainOrchestrator {
            steps,
            input: input.to_string(),
            current_step: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn make_instruction(&self, step: usize) -> String {
        let transform = &self.steps[step].1;
        let is_last = step == self.steps.len() - 1;
        let input_text = if step == 0 {
            self.input.clone()
        } else {
            "上游文本".to_string()
        };
        if is_last {
            let shop = &self.steps[step].0;
            format!(
                "TASK=你的信箱里已有指令信                 第二步: 对以下文本执行变换「{}」: {}                 第三步【唯一正确动作】: 调用 pulsing_send 工具, 参数 to={}, silent=true, message=REPORT=RELAY_RESULT=你变换后的实际输出结果                 铁律: 只做这一个变换; 禁止纯文字回复; 必须调用 pulsing_send 工具",
                transform, input_text, shop
            )
        } else {
            let next = &self.steps[step + 1].0;
            format!(
                "TASK=你的信箱里已有指令信                 第二步: 对以下文本执行变换「{}」: {}                 第三步【唯一正确动作】: 调用 pulsing_send 工具, 参数 to={}, silent=false, message=RELAY=你变换后的实际输出结果                 铁律: 只做这一个变换; 禁止纯文字回复; 必须调用 pulsing_send 工具; 收件人只能是 {}",
                transform, input_text, next, next
            )
        }
    }
}

impl Orchestrator for RelayChainOrchestrator {
    fn setup(&self, _ctx: &OrchestratorCtx) -> Vec<SetupLetter> {
        self.steps
            .iter()
            .map(|(shop, transform)| SetupLetter {
                shop: shop.clone(),
                to: shop.clone(),
                silent: true,
                body: format!(
                    "你收到文本和一个变换指令 执行[{}] 只做这一个变换",
                    transform
                ),
            })
            .collect()
    }

    fn ignition(&self) -> (String, String) {
        let shop = self.steps[0].0.clone();
        (shop, self.make_instruction(0))
    }

    fn on_letter(&self, letter: &Letter, _ctx: &OrchestratorCtx) -> Vec<OrchestratorAction> {
        let mut actions = Vec::new();
        let current = self.current_step.load(std::sync::atomic::Ordering::SeqCst);

        if letter.body.starts_with("RELAY=") && current + 1 < self.steps.len() {
            self.current_step
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let next = current + 1;
            let output: String = letter.body.chars().skip(6).take(200).collect();
            let instruction = self.make_instruction(next).replace("上游文本", &output);
            actions.push(OrchestratorAction::Inject {
                from: "orchestrator".to_string(),
                to: self.steps[next].0.clone(),
                body: instruction,
                silent: false,
            });
        }

        if letter.body.starts_with("REPORT=RELAY_RESULT=") {
            actions.push(OrchestratorAction::Complete {
                state: "PASS".to_string(),
                detail: format!("中继链{}步完成", self.steps.len()),
            });
        }
        actions
    }

    fn finalize(&self, ctx: &OrchestratorCtx) -> FinalState {
        let relay_count = ctx
            .letters
            .iter()
            .filter(|l| l.body.starts_with("RELAY="))
            .count();
        let has_report = ctx
            .letters
            .iter()
            .any(|l| l.body.starts_with("REPORT=RELAY_RESULT="));
        let total = self.steps.len();
        if has_report && relay_count >= total - 1 {
            FinalState::Pass
        } else if has_report {
            FinalState::Degenerate(format!(
                "REPORT但只有{} / {}跳RELAY",
                relay_count,
                total - 1
            ))
        } else if relay_count > 0 {
            // 链在走(有RELAY但还没到REPORT) — 不要提前判负, 给codex时间完成
            FinalState::Timeout
        } else {
            // 没有任何RELAY — 也给首个codex回合时间(LLM需10-60秒), 不立即Fail
            FinalState::Timeout
        }
    }
}

/// 规模广播: chief 广播 → N 店回声 → 收齐计数
/// 解决 T10 并发瓶颈: 编排器识别广播模式, 批量触发 + 回声计数
pub struct BroadcastOrchestrator {
    pub chief: String,
    pub echo_shops: Vec<String>,
    pub message: String,
}

impl Orchestrator for BroadcastOrchestrator {
    fn setup(&self, _ctx: &OrchestratorCtx) -> Vec<SetupLetter> {
        let mut seeds = Vec::new();
        // 广播载荷(可空): chief 透传给全体回声店, 回声店按其要求行动
        let payload = if self.message.trim().is_empty() {
            String::new()
        } else {
            format!("||任务: {}", self.message)
        };
        // 每家回声店的规程(三步铁律格式)
        for shop in &self.echo_shops {
            seeds.push(SetupLetter {
                shop: shop.clone(), to: shop.clone(), silent: true,
                body: format!(
                    "注意: 你的信箱里已有规程信\n第二步: 收到广播后, 按广播信里的任务要求完成动作(计算/写码/确认)\n第三步【唯一正确动作】: 调用 pulsing_send 工具, 参数 to={}, silent=true, message=ECHO={}|按广播要求完成后的产出(数值或一句话)\n铁律: 必须调用 pulsing_send 工具; 禁止纯文字回复; 只给 {} 发信",
                    self.chief, shop, self.chief
                ),
            });
        }
        // chief 规程: 一次广播, 收齐后 REPORT(三步铁律格式)
        let list_fmt = self
            .echo_shops
            .iter()
            .map(|s| format!("\"{s}\""))
            .collect::<Vec<_>>()
            .join(", ");
        seeds.push(SetupLetter {
            shop: self.chief.clone(), to: self.chief.clone(), silent: true,
            body: format!(
                "注意: 你的信箱里已有规程信\n第二步: 向全部{}家回声店发一次广播\n第三步【唯一正确动作】: 调用 pulsing_broadcast 工具, 参数 to_list=[{}], silent=false, message=ECHO=报名{}\n铁律: 必须调用 pulsing_broadcast 工具(只调一次); 禁止逐家单发; 未收齐{}个ECHO前禁止发REPORT; 广播后直接结束回合, 不要等待",
                self.echo_shops.len(), list_fmt, payload, self.echo_shops.len()
            ),
        });
        seeds
    }

    fn ignition(&self) -> (String, String) {
        (
            self.chief.clone(),
            format!("TASK=广播{}家店并收集回声", self.echo_shops.len()),
        )
    }

    fn on_letter(&self, letter: &Letter, ctx: &OrchestratorCtx) -> Vec<OrchestratorAction> {
        let mut actions = Vec::new();
        // ECHO 到达 chief → 检查是否收齐
        if letter.body.starts_with("ECHO=") && letter.to == self.chief {
            let echoes: std::collections::HashSet<&str> = ctx
                .letters
                .iter()
                .filter(|l| l.body.starts_with("ECHO=") && l.to == self.chief)
                .map(|l| l.from.as_str())
                .collect();
            let n = self.echo_shops.len();
            // 幂等: 已注入过ARBIT就不再注入(迟到回声每封都会再触发本回调)
            let arbit_sent = ctx.letters.iter().any(|l| {
                l.from == "orchestrator" && l.to == self.chief && l.body.starts_with("ARBIT=")
            });
            if echoes.len() >= n && !arbit_sent {
                actions.push(OrchestratorAction::Inject {
                    from: "orchestrator".into(),
                    to: self.chief.clone(),
                    body: format!(
                        "ARBIT=全部{}家ECHO已收齐({}/{}) 立即 pulsing_send(to='{}', silent=true, message='REPORT=COUNT={}|全部{}家回声') 收尾",
                        n, echoes.len(), n, self.chief, n, n
                    ),
                    silent: false,
                });
            }
        }
        // REPORT 到达 → 终局
        if letter.from == self.chief && letter.body.starts_with("REPORT=COUNT=") {
            actions.push(OrchestratorAction::Complete {
                state: "PASS".into(),
                detail: letter.body.chars().take(40).collect(),
            });
        }
        actions
    }

    fn finalize(&self, ctx: &OrchestratorCtx) -> FinalState {
        let echoes: std::collections::HashSet<&str> = ctx
            .letters
            .iter()
            .filter(|l| l.body.starts_with("ECHO=") && l.to == self.chief)
            .map(|l| l.from.as_str())
            .collect();
        let n = self.echo_shops.len();
        let has_report = ctx
            .letters
            .iter()
            .any(|l| l.from == self.chief && l.body.starts_with("REPORT=COUNT="));
        if has_report && echoes.len() >= n {
            FinalState::Pass
        } else {
            FinalState::Timeout // 回声在来/早REPORT = 仍在进行, 不提前判负
        }
    }

    fn forced_closure(&self, ctx: &OrchestratorCtx) -> Option<(String, String)> {
        let echoes: std::collections::HashSet<&str> = ctx
            .letters
            .iter()
            .filter(|l| l.body.starts_with("ECHO=") && l.to == self.chief)
            .map(|l| l.from.as_str())
            .collect();
        let n = self.echo_shops.len();
        let has_report = ctx
            .letters
            .iter()
            .any(|l| l.from == self.chief && l.body.starts_with("REPORT=COUNT="));
        if echoes.len() >= n && !has_report {
            Some((
                self.chief.clone(),
                format!(
                    "REPORT=COUNT={}|全部{n}家回声||机器闭合: chief超时未收尾, 回声计数来自账本",
                    n
                ),
            ))
        } else {
            None
        }
    }
}

/// 信件 Schema: 发信时校验必需内容(解决 REPORT 缺围栏 / DRAFT 缺代码)
/// 校验失败返回教学性错误(工具层拒绝, 模型在同一回合内自我纠正)
pub struct LetterSchema {
    /// 必需前缀 (如 "REPORT=" / "DRAFT=")
    pub prefix: String,
    /// 必需包含的关键词 (如 ["```python", "return"])
    pub required_contains: Vec<String>,
    /// 禁止包含 (如 ["undefined"] 防止垃圾输出)
    pub forbidden_contains: Vec<String>,
}

impl LetterSchema {
    pub fn new(prefix: &str, required: &[&str]) -> Self {
        LetterSchema {
            prefix: prefix.to_string(),
            required_contains: required.iter().map(|s| s.to_string()).collect(),
            forbidden_contains: Vec::new(),
        }
    }

    /// 校验信件正文: 通过返回 Ok, 不通过返回教学性错误
    pub fn validate(&self, body: &str) -> Result<(), String> {
        if !body.starts_with(&self.prefix) {
            return Err(format!(
                "信件必须以 {} 开头; 当前正文前10字符: {}",
                self.prefix,
                body.chars().take(10).collect::<String>()
            ));
        }
        for kw in &self.required_contains {
            if !body.contains(kw.as_str()) {
                return Err(format!(
                    "信件缺少必需内容: {}; 请检查后重发(如 REPORT 须包含完整脚本围栏)",
                    kw
                ));
            }
        }
        for kw in &self.forbidden_contains {
            if body.contains(kw.as_str()) {
                return Err(format!("信件包含禁止内容: {}", kw));
            }
        }
        Ok(())
    }

    /// 常用 schema 快捷构造
    pub fn report_with_code() -> Self {
        Self::new("REPORT=", &["```"])
    }
    pub fn draft_code() -> Self {
        Self::new("DRAFT=", &["```"])
    }
    pub fn arbit() -> Self {
        Self::new("ARBIT=", &[])
    }
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

    /// 回归 T7: 辩论 — 收齐2份ARG后必须自动注入ARBIT(解"不知何时结束")
    #[test]
    fn debate_collects_and_closes() {
        let orch = DebateOrchestrator {
            judge: "judge".into(),
            topic: "test".into(),
        };
        let shops = vec!["judge".into(), "pro_a".into(), "pro_b".into()];
        let letters = vec![
            l(1, "judge", "pro_a", "ASK=请给支持论据"),
            l(2, "judge", "pro_b", "ASK=请给反对论据"),
            l(3, "pro_a", "judge", "ARG=支持||AI利大于弊"),
            l(4, "pro_b", "judge", "ARG=反对||AI有风险"),
        ];
        let ctx = OrchestratorCtx {
            letters: &letters,
            shops: &shops,
        };
        // 第4封信(ARG from pro_b)到达 → 应触发 ARBIT 注入
        let actions = orch.on_letter(&letters[3], &ctx);
        assert!(
            actions.iter().any(
                |a| matches!(a, OrchestratorAction::Inject { body, .. } if body.contains("ARBIT"))
            ),
            "收齐2份ARG后应注入ARBIT: {actions:?}"
        );
        // 终局: judge REPORT → PASS
        let with_report = [
            letters.clone(),
            vec![l(5, "judge", "judge", "REPORT=辩论完成")],
        ]
        .concat();
        let ctx2 = OrchestratorCtx {
            letters: &with_report,
            shops: &shops,
        };
        let actions2 = orch.on_letter(&with_report[4], &ctx2);
        assert!(actions2
            .iter()
            .any(|a| matches!(a, OrchestratorAction::Complete { state, .. } if state == "PASS")));
    }

    /// 回归 T9: 维修队 — VERIFY到达后自动交叉校验并注入正确TOTAL
    #[test]
    fn saboteur_cross_verifies() {
        let orch = SaboteurOrchestrator {
            judge: "judge".into(),
        };
        let shops = vec![
            "judge".into(),
            "coder".into(),
            "breaker".into(),
            "verifier_a".into(),
            "verifier_b".into(),
        ];
        let letters = vec![
            l(1, "judge", "coder", "TASK=计算1..30平方和"),
            l(2, "judge", "breaker", "TASK=计算31..60平方和"),
            l(3, "judge", "verifier_a", "CHECK=独立核算1..60全程平方和"),
            l(4, "judge", "verifier_b", "CHECK=独立核算1..60全程平方和"),
            l(5, "coder", "judge", "RESULT=coder||1..30||9455"),
            l(6, "breaker", "judge", "RESULT=breaker||31..60||99999"), // 故障值
            l(7, "verifier_a", "judge", "VERIFY=1..60||73810"),
            l(8, "verifier_b", "judge", "VERIFY=1..60||73810"),
        ];
        let ctx = OrchestratorCtx {
            letters: &letters,
            shops: &shops,
        };
        let actions = orch.on_letter(&letters[7], &ctx);
        // 双核算一致(73810) → 应注入共识TOTAL, 且不能采信故障值99999
        assert!(actions.iter().any(|a| matches!(a, OrchestratorAction::Inject { body, .. } if body.contains("=73810") && !body.contains("99999"))),
            "VERIFY收齐后应注入共识TOTAL=73810: {actions:?}");
        // 终局: 材料齐 + 正确TOTAL → PASS
        let with_report = [
            letters.clone(),
            vec![l(9, "judge", "judge", "REPORT=TOTAL=73810||故障已剔除")],
        ]
        .concat();
        let ctx2 = OrchestratorCtx {
            letters: &with_report,
            shops: &shops,
        };
        let st = orch.finalize(&ctx2);
        assert!(
            matches!(st, FinalState::Pass),
            "材料齐+TOTAL正确应PASS: {st:?}"
        );
    }

    /// 回归 T3: 流水线 — 最后工序的WORK信到达后应可判定完成
    #[test]
    fn pipeline_detects_completion() {
        let orch = PipelineOrchestrator {
            stages: vec![
                ("planner".into(), "写三行大纲".into()),
                ("writer".into(), "按大纲写五句".into()),
                ("editor".into(), "改一句表达".into()),
            ],
        };
        let shops = vec!["planner".into(), "writer".into(), "editor".into()];
        let letters = vec![
            l(
                1,
                "planner",
                "writer",
                "WORK=大纲||1.AI协作 2.信件通信 3.写审回路",
            ),
            l(
                2,
                "writer",
                "editor",
                "WORK=五句短文||AI agents cooperate through letters...",
            ),
            l(3, "editor", "editor", "REPORT=RESULT=最终五句||处理链完成"),
        ];
        let ctx = OrchestratorCtx {
            letters: &letters,
            shops: &shops,
        };
        let state = orch.finalize(&ctx);
        assert_eq!(state, FinalState::Pass, "三步全走完+REPORT → 应PASS");
        // 缺writer的WORK → 应FAIL并指出缺失工序
        let partial = vec![letters[0].clone(), letters[2].clone()];
        let ctx2 = OrchestratorCtx {
            letters: &partial,
            shops: &shops,
        };
        let state2 = orch.finalize(&ctx2);
        assert!(
            matches!(state2, FinalState::Fail(ref msg) if msg.contains("writer")),
            "缺writer工序应FAIL: {state2:?}"
        );
    }

    /// 回归 T6: 写审 — complete loop + REPORT → PASS
    #[test]
    fn review_completes() {
        let orch = ReviewOrchestrator {
            judge: "judge".into(),
            writer: "writer".into(),
            reviewers: vec![
                ("review_a".into(), "检查A".into(), "kw_a".into()),
                ("review_b".into(), "检查B".into(), "kw_b".into()),
            ],
            reviewer_prefix: "review".into(),
        };
        let shops = vec![
            "judge".into(),
            "writer".into(),
            "review_a".into(),
            "review_b".into(),
        ];
        let mut letters = vec![
            l(1, "judge", "writer", "TASK=写代码"),
            l(2, "writer", "judge", "DRAFT=```python\npass\n```"),
            l(3, "judge", "review_a", "REVIEW=群审||底稿"),
            l(4, "judge", "review_b", "REVIEW=群审||底稿"),
            l(5, "review_a", "judge", "NEEDS_FIX=review_a||检查A: 缺失"),
            l(6, "review_b", "judge", "CLEAN=review_b||检查B: 已覆盖"),
        ];
        // 第一波收齐 → 应注入ARBIT(有NEEDS_FIX → REJECT方向)
        let ctx = OrchestratorCtx {
            letters: &letters,
            shops: &shops,
        };
        let actions = orch.on_letter(&letters[5], &ctx);
        assert!(
            actions.iter().any(
                |a| matches!(a, OrchestratorAction::Inject { body, .. } if body.contains("ARBIT"))
            ),
            "2/2回信收齐应注入ARBIT"
        );
        // 修正后第二波全CLEAN → REPORT → PASS
        letters.extend(vec![
            l(7, "judge", "writer", "REJECT=补检查A"),
            l(8, "writer", "judge", "REVISED=```python\nassert kw_a\n```"),
            l(9, "judge", "review_a", "REVIEW=群审||修正稿"),
            l(10, "judge", "review_b", "REVIEW=群审||修正稿"),
            l(11, "review_a", "judge", "CLEAN=review_a||检查A: 已覆盖"),
            l(12, "review_b", "judge", "CLEAN=review_b||检查B: 已覆盖"),
            l(13, "judge", "judge", "REPORT=全部审查通过"),
        ]);
        let ctx2 = OrchestratorCtx {
            letters: &letters,
            shops: &shops,
        };
        let state = orch.finalize(&ctx2);
        assert_eq!(state, FinalState::Pass);
    }
}

#[cfg(test)]
mod new_tests {
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

    /// 回归: 空审查组不能在写手交稿后立即收齐并注入裁决
    #[test]
    fn review_with_no_reviewers_times_out() {
        let orch = ReviewOrchestrator {
            judge: "judge".into(),
            writer: "writer".into(),
            reviewers: Vec::new(),
            reviewer_prefix: "review".into(),
        };
        let letters = vec![l(1, "writer", "judge", "DRAFT=```python\npass\n```")];
        let ctx = OrchestratorCtx {
            letters: &letters,
            shops: &["judge".into(), "writer".into()],
        };

        assert!(orch.on_letter(&letters[0], &ctx).is_empty());
        assert_eq!(orch.finalize(&ctx), FinalState::Timeout);
    }

    /// 回归: 零时刻交稿仍应按最新稿计算审查波次
    #[test]
    fn review_zero_timestamp_uses_latest_draft_wave() {
        let orch = ReviewOrchestrator {
            judge: "judge".into(),
            writer: "writer".into(),
            reviewers: vec![("review_a".into(), "检查A".into(), "kw_a".into())],
            reviewer_prefix: "review".into(),
        };
        let letters = vec![
            l(0, "writer", "judge", "DRAFT=```python\npass\n```"),
            l(1, "orchestrator", "review_a", "REVIEW=群审||底稿"),
            l(2, "review_a", "judge", "CLEAN=review_a||检查A: 已覆盖"),
        ];
        let ctx = OrchestratorCtx {
            letters: &letters,
            shops: &["judge".into(), "writer".into(), "review_a".into()],
        };
        let actions = orch.on_letter(&letters[2], &ctx);

        assert!(actions.iter().any(|action| matches!(action, OrchestratorAction::Inject { body, .. } if body.contains("ARBIT="))));
    }

    /// 回归: 重复交稿不得被旧波次代发标记吞掉
    #[test]
    fn review_revision_gets_new_dispatch() {
        let orch = ReviewOrchestrator {
            judge: "judge".into(),
            writer: "writer".into(),
            reviewers: vec![("review_a".into(), "检查A".into(), "kw_a".into())],
            reviewer_prefix: "review".into(),
        };
        let letters = vec![
            l(1, "writer", "judge", "DRAFT=```python\npass\n```"),
            l(2, "orchestrator", "review_a", "REVIEW=群审||初稿"),
            l(3, "writer", "judge", "REVISED=```python\nfix\n```"),
        ];
        let ctx = OrchestratorCtx {
            letters: &letters,
            shops: &["judge".into(), "writer".into(), "review_a".into()],
        };
        let actions = orch.on_letter(&letters[2], &ctx);

        assert!(actions.iter().any(
            |action| matches!(action, OrchestratorAction::Inject { from, to, body, .. }
            if from == "orchestrator" && to == "review_a" && body.contains("fix"))
        ));
    }

    /// 回归 T10: 规模广播 — 收齐 N 家回声后自动注入 ARBIT
    #[test]
    fn broadcast_collects_echoes() {
        let orch = BroadcastOrchestrator {
            chief: "chief".into(),
            echo_shops: vec!["e1".into(), "e2".into(), "e3".into()],
            message: "报名".into(),
        };
        let shops = vec!["chief".into(), "e1".into(), "e2".into(), "e3".into()];
        let letters = vec![
            l(1, "chief", "e1", "ECHO=报名"),
            l(2, "chief", "e2", "ECHO=报名"),
            l(3, "chief", "e3", "ECHO=报名"),
            l(4, "e1", "chief", "ECHO=e1|收到"),
            l(5, "e2", "chief", "ECHO=e2|收到"),
            l(6, "e3", "chief", "ECHO=e3|收到"),
        ];
        let ctx = OrchestratorCtx {
            letters: &letters,
            shops: &shops,
        };
        // 第6封信(e3的ECHO)到达 → 3/3收齐 → 注入ARBIT
        let actions = orch.on_letter(&letters[5], &ctx);
        assert!(
            actions.iter().any(
                |a| matches!(a, OrchestratorAction::Inject { body, .. } if body.contains("ARBIT"))
            ),
            "3/3 ECHO收齐应注入ARBIT: {actions:?}"
        );
        // REPORT到达 → Complete
        let with_report = letters.clone();
        let report = l(7, "chief", "chief", "REPORT=COUNT=3|全部3家回声");
        let ctx2 = OrchestratorCtx {
            letters: &[with_report.clone(), vec![report.clone()]].concat(),
            shops: &shops,
        };
        let actions2 = orch.on_letter(&report, &ctx2);
        assert!(actions2
            .iter()
            .any(|a| matches!(a, OrchestratorAction::Complete { state, .. } if state == "PASS")));
    }

    /// 回归 T10: 部分回声 → 不判负(Timeout)
    #[test]
    fn broadcast_partial_not_fail() {
        let orch = BroadcastOrchestrator {
            chief: "chief".into(),
            echo_shops: vec!["e1".into(), "e2".into(), "e3".into()],
            message: "报名".into(),
        };
        let shops = vec!["chief".into(), "e1".into(), "e2".into(), "e3".into()];
        let letters = vec![
            l(1, "e1", "chief", "ECHO=e1|收到"),
            l(2, "e2", "chief", "ECHO=e2|收到"),
            // e3 没回
        ];
        let ctx = OrchestratorCtx {
            letters: &letters,
            shops: &shops,
        };
        let state = orch.finalize(&ctx);
        assert_eq!(state, FinalState::Timeout, "部分回声不应判负");
    }

    /// 回归 T6/T1-V3: LetterSchema — REPORT 缺围栏被拒
    #[test]
    fn schema_rejects_missing_fence() {
        let schema = LetterSchema::report_with_code();
        // 有围栏 → 通过
        assert!(schema
            .validate("REPORT=全部通过\n```python\ndef f(): pass\n```")
            .is_ok());
        // 无围栏 → 拒绝(教学性错误)
        let err = schema.validate("REPORT=全部通过，代码已完成").unwrap_err();
        assert!(err.contains("缺少必需内容"), "错误信息应说明缺失: {err}");
    }

    /// LetterSchema — 错误前缀被拒
    #[test]
    fn schema_rejects_wrong_prefix() {
        let schema = LetterSchema::draft_code();
        // 正确前缀 → 通过
        assert!(schema.validate("DRAFT=```python\ncode\n```").is_ok());
        // 错误前缀 → 拒绝
        let err = schema.validate("这是我写的代码").unwrap_err();
        assert!(err.contains("必须以"), "错误信息应说明前缀: {err}");
    }

    /// LetterSchema — 多关键词校验
    #[test]
    fn schema_multiple_keywords() {
        let schema = LetterSchema::new("REVISED=", &["```python", "return"]);
        assert!(schema
            .validate("REVISED=```python\ndef f(): return True\n```")
            .is_ok());
        assert!(schema.validate("REVISED=```python\npass\n```").is_err());
    }
}
