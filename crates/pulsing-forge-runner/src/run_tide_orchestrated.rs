// run_tide_orchestrated — 编排器 + 真 LLM(codex) 的完整产品路径
// 这是 Forge 的最终形态: 确定性层验证协议 → 潮汐层执行语义, 两层用同一套编排器

use pulsing_forge_core::{FinalState, Orchestrator, OrchestratorAction, OrchestratorCtx};
use pulsing_forge_hub::{Acl, TideHub};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// 构建编排器(根据场景 toml 的 orchestrator 字段)
fn build_orchestrator(sc: &crate::Scenario) -> (Vec<String>, Box<dyn Orchestrator>) {
    match sc.orchestrator.as_str() {
        "relay" => {
            let stages: Vec<(String, String)> = sc
                .stages
                .iter()
                .map(|s| (s.shop.clone(), s.duty.clone()))
                .collect();
            let shops: Vec<String> = stages.iter().map(|(s, _)| s.clone()).collect();
            let input = if sc.meta.relay_input.is_empty() {
                "pulsing makes agents cooperate".to_string()
            } else {
                sc.meta.relay_input.clone()
            };
            (
                shops,
                Box::new(pulsing_forge_core::RelayChainOrchestrator::new(
                    stages, &input,
                )),
            )
        }
        "debate" => {
            let shops = vec!["judge".into(), "pro_a".into(), "pro_b".into()];
            (
                shops,
                Box::new(pulsing_forge_core::DebateOrchestrator {
                    judge: "judge".into(),
                    topic: if sc.debate_topic.is_empty() {
                        "test".into()
                    } else {
                        sc.debate_topic.clone()
                    },
                }),
            )
        }
        "pipeline" => {
            let stages: Vec<(String, String)> = sc
                .stages
                .iter()
                .map(|s| (s.shop.clone(), s.duty.clone()))
                .collect();
            let shops: Vec<String> = stages.iter().map(|(s, _)| s.clone()).collect();
            (
                shops,
                Box::new(pulsing_forge_core::PipelineOrchestrator { stages }),
            )
        }
        "saboteur" => {
            let shops = vec![
                "judge".into(),
                "coder".into(),
                "breaker".into(),
                "verifier_a".into(),
                "verifier_b".into(),
            ];
            (
                shops,
                Box::new(pulsing_forge_core::SaboteurOrchestrator {
                    judge: "judge".into(),
                }),
            )
        }
        "broadcast" => {
            // 回声店数量经 task.n_reviewers 传入(默认5); 店名沿用旧批的 e1..eN
            let n = if sc.task.n_reviewers > 0 {
                sc.task.n_reviewers
            } else {
                5
            };
            let echo_shops: Vec<String> = (1..=n).map(|i| format!("e{i}")).collect();
            let mut shops = vec!["chief".into()];
            shops.extend(echo_shops.iter().cloned());
            let msg = if sc.task.template.is_empty() {
                String::new()
            } else {
                sc.task.template.clone()
            };
            (
                shops,
                Box::new(pulsing_forge_core::BroadcastOrchestrator {
                    chief: "chief".into(),
                    echo_shops,
                    message: msg,
                }),
            )
        }
        "review" => {
            let reviewers: Vec<(String, String, String)> = sc
                .reviewers
                .iter()
                .map(|r| (r.name.clone(), r.duty.clone(), r.keyword.clone()))
                .collect();
            let shops = vec!["judge".into(), "writer".into()]
                .into_iter()
                .chain(reviewers.iter().map(|(n, _, _)| n.clone()))
                .collect();
            (
                shops,
                Box::new(pulsing_forge_core::ReviewOrchestrator {
                    judge: "judge".into(),
                    writer: "writer".into(),
                    reviewers,
                    reviewer_prefix: "review".into(),
                }),
            )
        }
        other => {
            eprintln!("未知编排器: {other}");
            std::process::exit(2);
        }
    }
}

/// 构建 ACL(按编排器类型)
fn build_acl(orch_type: &str, shops: &[String]) -> Acl {
    let rules: Vec<(String, Vec<String>)> = match orch_type {
        "relay" => {
            let mut r = Vec::new();
            for i in 0..shops.len() {
                let mut allow = vec!["orchestrator".to_string(), shops[i].clone()];
                if i + 1 < shops.len() {
                    allow.push(shops[i + 1].clone());
                }
                r.push((shops[i].clone(), allow));
            }
            r
        }
        "debate" => vec![
            ("judge".into(), vec!["*".into()]),
            ("pro_a".into(), vec!["judge".into()]),
            ("pro_b".into(), vec!["judge".into()]),
        ],
        "pipeline" => {
            let mut r = Vec::new();
            for i in 0..shops.len() {
                let mut allow = vec![shops[i].clone()];
                if i + 1 < shops.len() {
                    allow.push(shops[i + 1].clone());
                }
                r.push((shops[i].clone(), allow));
            }
            r
        }
        "saboteur" => vec![
            ("judge".into(), vec!["*".into()]),
            ("coder".into(), vec!["judge".into()]),
            ("breaker".into(), vec!["judge".into()]),
            ("verifier_a".into(), vec!["judge".into()]),
            ("verifier_b".into(), vec!["judge".into()]),
        ],
        "broadcast" => {
            let mut r = vec![("chief".to_string(), vec!["*".to_string()])];
            for s in shops.iter().filter(|s| *s != "chief") {
                r.push((s.clone(), vec!["chief".to_string()]));
            }
            r
        }
        "review" => {
            let mut r = vec![
                ("judge".into(), vec!["*".into()]),
                ("writer".into(), vec!["judge".into()]),
            ];
            // 审查店只回 judge
            r.push(("review_*".into(), vec!["judge".into()]));
            r
        }
        _ => vec![],
    };
    Acl { rules }
}

/// 编排器 + 真 LLM 的场景运行
pub fn run_tide_orchestrated(sc: &crate::Scenario, run_dir: &str) -> String {
    let port = 12410u16;
    let _ = std::fs::remove_dir_all(run_dir);
    std::fs::create_dir_all(format!("{run_dir}/config")).expect("创建目录");

    // provider.toml
    let api_key = std::env::var("FORGE_API_KEY").unwrap_or_else(|_| "missing-key".to_string());
    let base_url =
        std::env::var("FORGE_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:4011/v1".to_string());
    std::fs::write(
        format!("{run_dir}/config/provider.toml"),
        format!("base_url = \"{base_url}\"\napi_key = \"{api_key}\"\nwire_api = \"responses\"\n"),
    )
    .unwrap();

    // 1. 构建编排器 + 店册
    let (shops, orch) = build_orchestrator(sc);
    println!("编排器: {} ({} 店)", sc.orchestrator, shops.len());

    // 2. 创建 TideHub(真 codex 回合管理)
    let mut hub = TideHub::new(run_dir, port).expect("TideHub");
    for shop in &shops {
        hub.add_shop(shop, "glm-5-turbo");
    }
    hub.codex_bin =
        std::env::var("FORGE_CODEX_BIN").unwrap_or_else(|_| "/home/daimeng/bin/codex".to_string());
    hub.provider_path = std::fs::canonicalize(format!("{run_dir}/config/provider.toml"))
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| format!("{run_dir}/config/provider.toml"));
    hub.acl = build_acl(&sc.orchestrator, &shops);
    hub.auto_silence = Some(("review".into(), "judge".into()));

    let hub = Arc::new(hub);

    // 3. 预置规程信(先发, 记录时间戳, 调度器跳过这些)
    let empty: Vec<pulsing_forge_core::Letter> = vec![];
    let setup_ctx = OrchestratorCtx {
        letters: &empty,
        shops: &shops,
    };
    for s in orch.setup(&setup_ctx) {
        let _ = hub.send(&s.shop, &s.to, &s.body, s.silent, false);
    }
    let setup_done_ts = hub.journal.replay().last().map(|l| l.ts_ms).unwrap_or(0);

    // 4. 启动 TCP 控制口 + 店调度器(后台线程)
    let hub2 = hub.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("rt");
        rt.block_on(async {
            let h = hub2.clone();
            tokio::spawn(async move {
                if let Err(e) = pulsing_forge_hub::serve_control(h, port).await {
                    // 端口劫持防护: 绑不上口说明有僵尸forge占着(被杀批次的残尸),
                    // 此时 attach 会把信写进僵尸的账本 — 必须大声退出而非静默降级
                    eprintln!("控制口绑定失败(疑似僵尸forge占用{port}): {e}");
                    std::process::exit(3);
                }
            });
            let shop_list: Vec<String> = hub2.shops.keys().cloned().collect();
            // FORGE_REMOTE_SHOPS="e1,e2,..": 这些店的回合由远程沙盒的 worker
            // 自行 drain+执行(forge cell), 本地调度器不为它们起 codex
            let remote: Vec<String> = std::env::var("FORGE_REMOTE_SHOPS")
                .unwrap_or_default()
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            for s in shop_list {
                if remote.contains(&s) {
                    println!("店 {s} 为远程worker(跳过本地调度)");
                    continue;
                }
                let h = hub2.clone();
                let shop = s.clone();
                let skip = setup_done_ts;
                tokio::spawn(async move {
                    pulsing_forge_hub::shop_scheduler_from(h, shop, skip).await;
                });
            }
            loop {
                tokio::time::sleep(Duration::from_secs(3600)).await;
            }
        });
    });
    std::thread::sleep(Duration::from_millis(1500));

    // 5. 点火(经 TCP, 非静默 → 触发第一个 codex 回合)
    let (ign_to, ign_body) = orch.ignition();
    println!(
        "点火 → {ign_to}: {}",
        ign_body.chars().take(30).collect::<String>().as_str()
    );
    let _ = hub.send(
        "orchestrator",
        &ign_to,
        &format!("{ign_body} 任务: {}", sc.task.template),
        false,
        false,
    );

    // 6. 编排器循环: 轮询账本、检测闸门、注入指令(5 秒一查, codex 回合需时间)
    let timeout_secs: u64 = std::env::var("FORGE_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    let start = Instant::now();
    let mut final_state = FinalState::Timeout;
    let mut processed: usize = 0; // 已处理过的信件数(增量回调, 防重复注入)
                                  // 机器强制闭合的宽限计时: 终局店 → 首次发现"闸门满足但未收尾"的时刻
    let mut closure_wait: std::collections::HashMap<String, Instant> =
        std::collections::HashMap::new();
    // 进度看门狗(Super-agent 吸收): 外部 Control Loop, 从账本识别无进展模式并诊断踢
    let wd_cfg = pulsing_forge_core::WatchdogConfig::default();
    let mut wd_fired: std::collections::HashMap<(String, String), std::time::Instant> =
        std::collections::HashMap::new(); // (店, 告警类) → 上次触发时刻, 60s限频

    loop {
        std::thread::sleep(Duration::from_secs(5));
        let elapsed = start.elapsed().as_secs();

        let letters = hub.journal.replay();
        let mut injected = false;

        // 进度看门狗: 同构循环/杂信风暴 → 诊断性踢(每店每类60秒一次, 幂等)
        for alert in pulsing_forge_core::watchdog_inspect(&letters, &wd_cfg) {
            let key = match &alert {
                pulsing_forge_core::WatchdogAlert::Loop { shop, .. } => {
                    (shop.clone(), "loop".into())
                }
                pulsing_forge_core::WatchdogAlert::Chatter { shop, .. } => {
                    (shop.clone(), "chatter".into())
                }
            };
            let cooldown_ok = wd_fired
                .get(&key)
                .map(|t| t.elapsed() >= Duration::from_secs(60))
                .unwrap_or(true);
            if cooldown_ok {
                wd_fired.insert(key, std::time::Instant::now());
                let (shop, why) = match &alert {
                    pulsing_forge_core::WatchdogAlert::Loop { shop, count, .. } => {
                        (shop.clone(), format!("同构循环×{count}"))
                    }
                    pulsing_forge_core::WatchdogAlert::Chatter { shop, count, .. } => {
                        (shop.clone(), format!("杂信风暴×{count}"))
                    }
                };
                println!("  [看门狗] 踢 → {shop}: {why}");
                let _ = hub.send(
                    "watchdog",
                    &shop,
                    &pulsing_forge_core::kick_body(&alert),
                    false,
                    false,
                );
                injected = true;
            }
        }

        // 处理编排器回调(只处理新到的信, 防止同一封尾巴信被反复回调)
        for letter in letters.iter().skip(processed) {
            let ctx = OrchestratorCtx {
                letters: &letters,
                shops: &shops,
            };
            for action in orch.on_letter(letter, &ctx) {
                match action {
                    OrchestratorAction::Inject {
                        from,
                        to,
                        body,
                        silent,
                    } => {
                        println!(
                            "  [编排器] 注入 → {to}: {}",
                            body.chars().take(25).collect::<String>().as_str()
                        );
                        let _ = hub.send(&from, &to, &body, silent, false);
                        injected = true;
                    }
                    OrchestratorAction::Kick { shop, reason } => {
                        println!("  [编排器] 踢 → {shop}: {reason}");
                        let _ = hub.send(
                            "orchestrator",
                            &shop,
                            &format!("KICK=继续任务: {reason}"),
                            false,
                            false,
                        );
                        injected = true;
                    }
                    OrchestratorAction::Complete { state, detail } => {
                        println!("  [编排器] 终局: {state} — {detail}");
                        final_state = match state.as_str() {
                            "PASS" => FinalState::Pass,
                            _ => FinalState::Degenerate(detail),
                        };
                    }
                    _ => {}
                }
            }
        }
        processed = letters.len();

        // 注入后重读(可能同步触发了新回合)
        let letters2 = if injected {
            hub.journal.replay()
        } else {
            hub.journal.replay()
        };
        let ctx2 = OrchestratorCtx {
            letters: &letters2,
            shops: &shops,
        };

        // 机器强制闭合(保底): 闸门已满足但终局店迟迟不收尾(实测法官抽风开小会),
        // 宽限60秒后代笔落账 — 机器管协议的收尾端
        if let Some((to, body)) = orch.forced_closure(&ctx2) {
            let first_seen = closure_wait.entry(to.clone()).or_insert_with(Instant::now);
            if first_seen.elapsed() >= Duration::from_secs(60) {
                println!("  [机器闭合] {to} 60s未收尾, 代笔 REPORT");
                let _ = hub.send(&to, &to, &body, true, false);
                closure_wait.remove(&to);
            }
        }

        let state = orch.finalize(&ctx2);
        if state != FinalState::Timeout {
            final_state = state;
            break;
        }

        if elapsed > timeout_secs {
            // 死限分类: 有REPORT但finalize不认 = 早产出局; 完全无产出 = 纯超时
            let any_report = letters2.iter().any(|l| l.body.starts_with("REPORT="));
            if any_report {
                println!(
                    "  超时 {timeout_secs}s ({} 封信, 有早产REPORT)",
                    letters2.len()
                );
                final_state = FinalState::Degenerate("超时: REPORT存在但终局材料不齐".into());
            } else {
                println!("  超时 {timeout_secs}s ({} 封信)", letters2.len());
                final_state = FinalState::Timeout;
            }
            break;
        }
        if elapsed % 30 == 0 {
            println!("  等待中: {elapsed}s, {} 封信", letters.len());
        }
    }

    // 7. 判分输出
    let letters = hub.journal.replay();
    let turns: u64 = hub
        .shops
        .values()
        .map(|s| s.turns_done.load(std::sync::atomic::Ordering::Relaxed))
        .sum();
    let state_str = match &final_state {
        FinalState::Pass => "PASS".to_string(),
        FinalState::Degenerate(d) => format!("DEGENERATE({d})"),
        FinalState::Fail(f) => format!("FAIL({f})"),
        FinalState::Timeout => "TIMEOUT".to_string(),
    };

    format!(
        "scenario: {} ({} + codex)\nstate: {}\nletters: {}\nturns: {}\nduration: {}s\n",
        sc.meta.name,
        sc.orchestrator,
        state_str,
        letters.len(),
        turns,
        start.elapsed().as_secs()
    )
}

/// 经 TCP 控制口发信(非直接调 hub.send — codex 的 MCP 工具走这条路)
fn tide_send(hub: &Arc<TideHub>, port: u16, from: &str, to: &str, body: &str, silent: bool) {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).expect("连接控制口");
    let cmd = serde_json::json!({
        "tool": "pulsing_send",
        "args": {"_shop": from, "to": to, "message": body, "silent": silent}
    });
    stream.write_all(format!("{cmd}\n").as_bytes()).expect("写");
    let mut buf = [0u8; 1024];
    let _ = stream.read(&mut buf);
}
