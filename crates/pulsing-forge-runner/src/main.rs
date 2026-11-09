//! forge-cli — `forge run <scenario.toml>`: 场景即数据, 一条命令一个多agent回路。
//! Phase 0 内置确定性店员(脚本层, 无LLM): 潮汐层(codex)经 adapter 接入(见docs)。

mod run_tide_orchestrated;

mod codex_adapter;
mod dashboard;

use pulsing_forge_core::{report_is_final, CollectGate, VerdictTable};
use pulsing_forge_hub::{Acl, Hub, OutLetter};
use std::collections::HashMap;

#[derive(Debug, serde::Deserialize)]
struct Scenario {
    meta: Meta,
    task: Task,
    #[serde(default)]
    reviewers: Vec<Reviewer>,
    #[serde(default = "default_adapter")]
    adapter: String,
    /// 编排器类型: review / pipeline / debate / saboteur / hub(默认=旧写审)
    #[serde(default = "default_orchestrator")]
    orchestrator: String,
    /// 流水线阶段(pipeline 编排器用)
    #[serde(default)]
    stages: Vec<Stage>,
    /// 辩论主题(debate 编排器用)
    #[serde(default)]
    debate_topic: String,
}
fn default_adapter() -> String {
    "builtin".to_string()
}
fn default_orchestrator() -> String {
    "legacy".to_string()
}

#[derive(Debug, serde::Deserialize)]
struct Stage {
    shop: String,
    duty: String,
}
#[derive(Debug, serde::Deserialize)]
struct Meta {
    name: String,
    #[serde(default)]
    relay_input: String,
}
#[derive(Debug, serde::Deserialize)]
struct Task {
    template: String,
    /// 埋雷: 任务不含但领域标准要求的检查(审查店应抓出)
    hidden_check: String,
    hidden_keyword: String,
    #[serde(default = "default_n")]
    n_reviewers: usize,
}
#[derive(Debug, serde::Deserialize)]
struct Reviewer {
    name: String,
    duty: String,
    keyword: String,
}
fn default_n() -> usize {
    4
}

fn out(to: &str, body: &str, silent: bool) -> OutLetter {
    OutLetter {
        to: to.into(),
        body: body.into(),
        silent,
    }
}

/// 确定性写审回路(脚本店员版): 完整走 写→审→裁决→打回→修正→复审→双通道收尾。
/// 每一步都是 taxonomy 里对应原语的调用现场。
pub fn run(sc: &Scenario, run_dir: &str) -> String {
    let mut shops = vec!["judge".to_string(), "writer".to_string()];
    for r in &sc.reviewers {
        shops.push(r.name.clone());
    }
    // ACL: 写手只寄仲裁者; 审查店只寄仲裁者; 仲裁者全通
    let mut acl_pairs: Vec<(String, Vec<String>)> = vec![
        ("writer".into(), vec!["judge".into()]),
        ("judge".into(), vec!["*".into()]),
    ];
    for r in &sc.reviewers {
        acl_pairs.push((format!("{}*", r.name), vec!["judge".into()]));
    }
    // ACL 规则名用前缀匹配审查店
    let acl = Acl { rules: acl_pairs };

    let mut hub = Hub::new(run_dir, shops, acl.clone()).unwrap();
    hub.auto_silence = Some(("review".into(), "judge".into()));

    // ── 店员(确定性) ──
    let task_text = format!("TASK={}", sc.task.template);
    let hidden = sc.task.hidden_keyword.clone();
    let trap_desc = sc.task.hidden_check.clone();
    let n = sc.task.n_reviewers;

    // 写手: 初稿故意不含隐藏检查(照任务信字面); 打回后按意见照单全收
    let all_keywords: Vec<String> = sc.reviewers.iter().map(|r| r.keyword.clone()).collect();
    hub.register(
        "writer",
        Box::new(move |trig, _| {
            let b = &trig.body;
            if b.starts_with("TASK=") {
                let draft = "DRAFT=```python\ndef validate(x):\n    # 按任务信实现\n    pass\n```"
                    .to_string();
                vec![out("judge", &draft, false)]
            } else if b.starts_with("REJECT=") {
                // 确定性修正: REJECT裁决表中点名的每个检查各补一行
                let mut checks = String::new();
                for k in &all_keywords {
                    if b.contains(k) {
                        checks.push_str(&format!("\n    assert '{k}'  # 已补"));
                    }
                }
                if checks.is_empty() {
                    checks = format!(
                        "\n    assert '{k}'",
                        k = all_keywords.first().cloned().unwrap_or_default()
                    );
                }
                let draft =
                    format!("REVISED=```python\ndef validate(x):{checks}\n    return True\n```");
                vec![out("judge", &draft, false)]
            } else {
                vec![]
            }
        }),
    );

    // 审查店: 各查一件事 — duty_keyword 在稿中即 CLEAN, 否则 NEEDS_FIX
    for r in &sc.reviewers {
        let kw = r.keyword.clone();
        let duty = r.duty.clone();
        let name = r.name.clone();
        hub.register(
            &r.name,
            Box::new(move |trig, _| {
                if !trig.body.contains("DRAFT") && !trig.body.contains("REVISED") {
                    return vec![];
                }
                let verdict = if trig.body.contains(&kw) {
                    format!("CLEAN={name}||{duty}: 已覆盖")
                } else {
                    format!("NEEDS_FIX={name}||{duty}: 缺失({kw})")
                };
                vec![out("judge", &verdict, false)] // hub 的 auto_silence 会转静默
            }),
        );
    }

    // ── 机器侧: 闸门驱动(这正是 forge-core 的调用现场) ──
    // judge 店员是"弱仲裁者"(只转发): 真正的协议由下面的闸门循环执行 —
    // 双通道收尾的"机器通道"即本函数, 不依赖任何 LLM 回合。
    let reviewers_prefix = "review".to_string();
    let gate = CollectGate {
        reviewers_prefix: reviewers_prefix.clone(),
        arbitrator: "judge".into(),
        min_replies: n,
    };

    hub.ignite("writer", &task_text).unwrap();

    // 主循环: 波次驱动(简化潮汐 — 每波 = 群发 → 收信 → 裁决)
    let mut waves = 0;
    let mut final_report: Option<String> = None;
    while waves < 3 && final_report.is_none() {
        waves += 1;
        // 1) 取最新稿(登记处: 账本即registry, 不依赖信箱)
        let letters = hub.letters();
        let draft_ts = letters
            .iter()
            .filter(|l| {
                l.from == "writer"
                    && (l.body.starts_with("DRAFT=") || l.body.starts_with("REVISED="))
            })
            .map(|l| l.ts_ms)
            .max()
            .unwrap_or(0);
        if draft_ts == 0 {
            break;
        }
        let draft_body = letters
            .iter()
            .rev()
            .find(|l| l.from == "writer" && l.ts_ms == draft_ts)
            .unwrap()
            .body
            .clone();
        // 2) 群审(机器代发: 底稿逐字嵌入 — taxonomy #11)
        let names: Vec<String> = sc.reviewers.iter().map(|r| r.name.clone()).collect();
        let fan_body = format!("REVIEW=群审||底稿如下||{draft_body}");
        hub.broadcast("judge", &names, &fan_body, false);
        // 等审查店回合完成(确定性: 广播即同步完成)
        // 3) 收齐闸门 + 裁决表(taxonomy #12)
        let letters = hub.letters();
        // 波锚=最新稿之后的第一封群发信: 同步店员下回信会夹在本波群发之间,
        // 用max会把自己的回信排除在波外(实测answered=1/4)
        let wave_ts = letters
            .iter()
            .filter(|l| l.from == "judge" && l.body.starts_with("REVIEW=") && l.ts_ms > draft_ts)
            .map(|l| l.ts_ms)
            .min()
            .unwrap_or(0);
        let (answered, done) = gate.collect(&letters, wave_ts);
        eprintln!(
            "[debug] wave_ts={wave_ts} answered={} done={done} n={n}",
            answered.len()
        );
        if !done {
            final_report = Some("REPORT=BUDGET_OUT||回信未收齐".into());
            break;
        }
        let vt = VerdictTable { gate: &gate };
        let (table, nfix) = vt.render(&letters, wave_ts);
        if nfix == 0 {
            // 4a) 双通道收尾: 机器默认合账(taxonomy #9)
            let _ = hub.send(
                "judge",
                "judge",
                &format!("REPORT=全部审查通过||裁决表:\n{table}"),
                true,
            );
            final_report = Some(format!("REPORT=全部审查通过(波次{waves})"));
        } else {
            // 4b) 机器代投打回: 从裁决表生成REJECT(taxonomy #10), 交给writer修正
            let rej = format!("REJECT=意见汇总||{table}||写手动作: 补齐后重交");
            let _ = hub.send("judge", "writer", &rej, false);
        }
    }
    if final_report.is_none() {
        final_report = Some("REPORT=FAIL||超过3波未收敛".into());
    }

    // ── 判分: 完整回路校验 + 真值(taxonomy #8 + checker) ──
    let letters = hub.letters();
    let complete = report_is_final(&letters, "judge", &reviewers_prefix, n, n);
    let final_script_ok = letters
        .iter()
        .rev()
        .find(|l| {
            l.from == "writer" && (l.body.starts_with("DRAFT=") || l.body.starts_with("REVISED="))
        })
        .map(|l| l.body.contains(&hidden))
        .unwrap_or(false);
    let closed_ok = letters
        .iter()
        .any(|l| l.from == "judge" && l.body.starts_with("REPORT=全部审查通过"));
    let state = if complete && final_script_ok && closed_ok {
        "PASS"
    } else if complete && closed_ok {
        "DEGENERATE"
    } else {
        "FAIL"
    };
    let turns: u64 = hub.turns.values().sum();
    let report = format!(
        "scenario: {}\nstate: {}\nwaves: {}\nturns: {}\nletters: {}\nfinal_report: {}\ncomplete_loop: {}\nhidden_check_present: {}\n",
        sc.meta.name, state, waves, turns, letters.len(), final_report.unwrap(), complete, final_script_ok);
    report
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // forge init
    if args.len() >= 2 && args[1] == "init" {
        let name = args.get(2).map(|s| s.as_str()).unwrap_or("my-scenario");
        let tmpl = format!(
            r#"# 场景: {name}
# forge run {name}.toml
[meta]
name = "{name}"

[task]
template = "写一个函数完成你的任务"
hidden_check = "描述领域标准要求但任务信未提及的检查"
hidden_keyword = "某个关键词"
n_reviewers = 4

[[reviewers]]
name = "review_a"
duty = "检查项A"
keyword = "keyword_a"

[[reviewers]]
name = "review_b"
duty = "检查项B"
keyword = "keyword_b"

[[reviewers]]
name = "review_c"
duty = "检查项C"
keyword = "keyword_c"

[[reviewers]]
name = "review_d"
duty = "检查项D"
keyword = "keyword_d"
"#
        );
        let fname = format!("{name}.toml");
        std::fs::write(&fname, tmpl).expect("写模板");
        println!("✅ 已创建 {fname} — 编辑后 forge run {fname}");
        return;
    }
    if args.len() >= 2 && args[1] == "attach" {
        // forge attach --control <port|ip:port> --shop <name> — MCP stdio→TCP 桥
        // (由 codex 作为 MCP server 子进程调用; ip:port 形态用于 forge cell 远程信箱)
        let mut control = "127.0.0.1:12400".to_string();
        let mut shop = None;
        let mut it = args.iter().peekable();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--control" => {
                    if let Some(p) = it.next() {
                        control = match p.parse::<u16>() {
                            Ok(port) => format!("127.0.0.1:{port}"),
                            Err(_) => p.clone(),
                        };
                    }
                }
                "--shop" => {
                    if let Some(s) = it.next() {
                        shop = Some(s.clone());
                    }
                }
                _ => {}
            }
        }
        let rt = tokio::runtime::Runtime::new().expect("rt");
        rt.block_on(async {
            if let Err(e) = pulsing_forge_hub::run_attach(control, shop).await {
                eprintln!("attach 错误: {e}");
            }
        });
        return;
    }
    if args.len() >= 2 && args[1] == "serve" {
        // forge serve — 启动潮汐层服务器(控制口+店调度器)
        let port: u16 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12400);
        let log_dir = format!("/tmp/forge-tide-{}", std::process::id());
        let mut hub = pulsing_forge_hub::TideHub::new(&log_dir, port).expect("创建TideHub");
        // 默认加 judge + writer 店(可用环境变量扩展)
        hub.add_shop("judge", "glm-5-turbo");
        hub.add_shop("writer", "glm-5-turbo");
        hub.codex_bin = std::env::var("FORGE_CODEX_BIN")
            .unwrap_or_else(|_| "/home/daimeng/bin/codex".to_string());
        hub.provider_path = std::env::var("FORGE_PROVIDER")
            .unwrap_or_else(|_| "/tmp/forge-tide/config/provider.toml".to_string());
        let hub = std::sync::Arc::new(hub);
        println!("潮汐层服务器: 控制口 :{port} 日志 {log_dir}");
        println!(
            "环境变量: FORGE_CODEX_BIN={} FORGE_PROVIDER={}",
            hub.codex_bin, hub.provider_path
        );
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            if let Err(e) = pulsing_forge_hub::run_tide(hub).await {
                eprintln!("潮汐层错误: {e}");
            }
        });
        return;
    }
    if args.len() >= 2 && args[1] == "test-tide" {
        // forge test-tide — 验证 TCP 控制口 + 4 工具管道(不需要 LLM)
        test_tide();
        return;
    }
    if args.len() < 2 || args[1] != "run" || args.len() < 3 {
        eprintln!(
            "用法:
  forge init [name]        创建场景模板
  forge run <file.toml>    运行场景(确定性层)
  forge serve [port]       启动潮汐层服务器
  forge test-tide          验证 TCP+MCP 管道"
        );
        std::process::exit(2);
    }
    let content = std::fs::read_to_string(&args[2]).expect("读场景文件");
    let sc: Scenario = toml::from_str(&content).expect("场景解析");
    let run_dir = format!(
        "runs/{}-{}",
        sc.meta.name,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );
    // adapter 分发: codex=真LLM潮汐层, builtin=确定性层
    if sc.adapter == "codex" {
        // 编排器 + 真 LLM = 完整产品路径
        if sc.orchestrator != "legacy" {
            println!(
                "{}",
                run_tide_orchestrated::run_tide_orchestrated(&sc, &run_dir)
            );
            println!("产物: {run_dir}/events-*.jsonl");
            return;
        }
        println!("{}", run_codex(&sc, &run_dir));
        println!("产物: {run_dir}/events-*.jsonl");
        return;
    }
    // orchestrator 分发: pipeline/debate/saboteur 走编排器路径
    if sc.orchestrator != "legacy" {
        println!("{}", run_orchestrated(&sc, &run_dir));
        println!("产物: {run_dir}/events-*.jsonl");
        return;
    }
    println!("{}", run(&sc, &run_dir));
    println!("产物: {run_dir}/events-*.jsonl");
    // 内置大屏: Mermaid 泳道时序图
    let mut journal: Vec<serde_json::Value> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&run_dir) {
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with("events-") || !name.ends_with(".jsonl") {
                continue;
            }
            if let Ok(content) = std::fs::read_to_string(entry.path()) {
                for line in content.lines() {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                        if v["ev"] == "reply_out" {
                            journal.push(v);
                        }
                    }
                }
            }
        }
    }
    dashboard::render_sequence_html(&journal, &sc.meta.name, &format!("{run_dir}/sequence.html"));
    println!("大屏: {run_dir}/sequence.html");
}

/// 验证 TCP 控制口 + 4 工具管道: 不需要 LLM/codex, 只测 plumbing
fn test_tide() {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;

    let port = 12499u16;
    let log_dir = "/tmp/forge-test-tide";
    let _ = std::fs::remove_dir_all(log_dir);

    // 1. 创建 TideHub 并在后台线程启动 TCP 服务器
    let mut hub = pulsing_forge_hub::TideHub::new(log_dir, port).expect("TideHub");
    hub.add_shop("alpha", "test");
    hub.add_shop("beta", "test");
    let acl = pulsing_forge_hub::Acl::from_pairs(&[
        ("alpha", &["beta", "alpha"]),
        ("beta", &["alpha"]),
    ]);
    hub.acl = acl;
    // 不设置 codex_bin → run_turn 会失败, 但 TCP 管道可验证
    let hub = std::sync::Arc::new(hub);

    let hub2 = hub.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("rt");
        rt.block_on(async {
            let _ = pulsing_forge_hub::serve_control(hub2, port).await;
        });
    });
    std::thread::sleep(std::time::Duration::from_millis(500));

    let mut pass = 0;
    let mut fail = 0;

    let send_cmd = |json: &str| -> String {
        let mut s = TcpStream::connect(("127.0.0.1", port)).expect("连接");
        s.write_all(format!("{json}\n").as_bytes()).expect("写");
        let mut line = String::new();
        BufReader::new(s).read_line(&mut line).expect("读");
        line.trim().to_string()
    };

    // 2. 测试 pulsing_send (alpha → beta, 静默)
    let r = send_cmd(
        r#"{"tool":"pulsing_send","args":{"_shop":"alpha","to":"beta","message":"hello","silent":true}}"#,
    );
    if r.contains("已送达") {
        pass += 1;
        println!("✅ pulsing_send: {r}");
    } else {
        fail += 1;
        println!("❌ pulsing_send: {r}");
    }

    // 3. 测试 pulsing_drain_mail (beta 读信)
    let r = send_cmd(r#"{"tool":"pulsing_drain_mail","args":{"_shop":"beta"}}"#);
    if r.contains("hello") {
        pass += 1;
        println!("✅ pulsing_drain_mail: 信件已读到");
    } else {
        fail += 1;
        println!("❌ pulsing_drain_mail: {r}");
    }

    // 4. 测试 ACL (alpha 只能发 beta/alpha, 不能发 unknown)
    let r = send_cmd(
        r#"{"tool":"pulsing_send","args":{"_shop":"alpha","to":"unknown","message":"x","silent":true,"_from_codex":true}}"#,
    );
    if r.contains("不在") {
        pass += 1;
        println!("✅ ACL 越权拒绝: {r}");
    } else {
        fail += 1;
        println!("❌ ACL: {r}");
    }

    // 5. 测试 pulsing_broadcast (alpha → beta, 白名单收拢)
    let r = send_cmd(
        r#"{"tool":"pulsing_broadcast","args":{"_shop":"alpha","to_list":["beta","unknown"],"message":"bc","silent":true,"_from_codex":true}}"#,
    );
    if r.contains("已群发 1") {
        pass += 1;
        println!("✅ pulsing_broadcast 收拢: {r}");
    } else {
        fail += 1;
        println!("❌ broadcast: {r}");
    }

    // 6. 测试 pulsing_health
    let r = send_cmd(r#"{"tool":"pulsing_health","args":{}}"#);
    if r.contains("店=2") {
        pass += 1;
        println!("✅ pulsing_health: {r}");
    } else {
        fail += 1;
        println!("❌ health: {r}");
    }

    // 7. 验证账本
    let journal = hub.journal.replay();
    if journal.len() >= 2 {
        pass += 1;
        println!("✅ 账本: {} 封信", journal.len());
    } else {
        fail += 1;
        println!("❌ 账本: {} 封", journal.len());
    }

    println!("--- 管道验证: {pass} PASS / {fail} FAIL ---");
    if fail > 0 {
        std::process::exit(1);
    }
}

/// 潮汐层场景运行: 真LLM agent 通过 codex exec 每回合调用
fn run_codex(sc: &Scenario, run_dir: &str) -> String {
    use pulsing_forge_hub::TideHub;
    use std::sync::Arc;

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
    .expect("写 provider");

    // 1. 创建 TideHub
    let mut hub = TideHub::new(run_dir, port).expect("TideHub");
    hub.add_shop("judge", "glm-5-turbo");
    hub.add_shop("writer", "glm-5-turbo");
    for r in &sc.reviewers {
        hub.add_shop(&r.name, "glm-5-turbo");
    }
    hub.codex_bin =
        std::env::var("FORGE_CODEX_BIN").unwrap_or_else(|_| "/home/daimeng/bin/codex".to_string());
    hub.provider_path = format!("{run_dir}/config/provider.toml");
    hub.auto_silence = Some(("review".into(), "judge".into()));

    // ACL
    let mut rules: Vec<(String, Vec<String>)> = vec![
        ("writer".into(), vec!["judge".into()]),
        ("judge".into(), vec!["*".into()]),
    ];
    for r in &sc.reviewers {
        rules.push((format!("{}*", r.name), vec!["judge".into()]));
    }
    hub.acl = Acl { rules };

    // forge 自身路径(attach 子命令用)
    let forge_bin = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "forge".to_string());

    let hub = Arc::new(hub);

    // 2. 启动 TCP 服务器 + 店调度器(后台线程)
    let hub2 = hub.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("rt");
        rt.block_on(async {
            let h = hub2.clone();
            tokio::spawn(async move {
                if let Err(e) = pulsing_forge_hub::serve_control(h, port).await {
                    eprintln!("控制口错误: {e}");
                }
            });
            // 店调度器
            let shops: Vec<String> = {
                let h = hub2.clone();
                // 从 Arc<TideHub> 获取店名列表
                h.shops.keys().cloned().collect()
            };
            for s in shops {
                let h = hub2.clone();
                let shop = s.clone();
                tokio::spawn(async move {
                    pulsing_forge_hub::shop_scheduler(h, shop).await;
                });
            }
            // 保持运行
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            }
        });
    });
    std::thread::sleep(std::time::Duration::from_millis(1000));

    // 3. 预置规程信(编排器通道, 不走 TCP)
    let reviewer_names: Vec<String> = sc.reviewers.iter().map(|r| r.name.clone()).collect();
    let task_text = format!("TASK={}", sc.task.template);

    // judge 规程
    let _ = hub.send("orch", "judge", &format!(
        "【指挥规程judge】见TASK=: pulsing_send(to='writer', silent=false, message='TASK=原文||写手动作:写完整脚本放```python围栏后 pulsing_send(to=\"judge\",silent=false,message=\"DRAFT=```python全文```\")')\n\
         见DRAFT=/REVISED=: pulsing_broadcast(to_list={reviewer_names:?}, message='REVIEW=群审||各店只查自己职责||底稿如下||'+脚本原文)\n\
         收齐后: 全CLEAN→REPORT收尾; 有不通→一封REJECT打回(含缺失检查名)\n\
         铁律: 绝不自己写代码; 收件人只填店名。"
    ), true, false);

    // writer 规程
    let _ = hub.send("orch", "writer",
        "【写手规程writer】见TASK=: 按任务写python函数, 完整脚本放```python围栏, pulsing_send(to='judge', silent=false, message='DRAFT=```python全文```')\n\
         见REJECT=: 按意见补齐后重交 message='REVISED=```python全文```'\n\
         铁律: 只给judge发信; 回信必须带```围栏。", true, false);

    // 审查店规程
    for r in &sc.reviewers {
        let _ = hub.send("orch", &r.name, &format!(
            "【审查规程{name}】见REVIEW=: 只查「{duty}」一事({keyword}应在稿中出现)。\n\
             合格: pulsing_send(to='judge', silent=true, message='CLEAN={name}||{duty}: 已覆盖')\n\
             不合格: pulsing_send(to='judge', silent=true, message='NEEDS_FIX={name}||{duty}: 缺失({keyword})')\n\
             铁律: 只回judge且silent=true。", name = r.name, duty = r.duty, keyword = r.keyword
        ), true, false);
    }

    // 4. 点火
    println!("点火: {task_text}");
    let _ = hub.send(
        "orch",
        "judge",
        &format!("{task_text}。第一步 pulsing_drain_mail() 读规程; 第二步派单给writer; 然后结束。"),
        false,
        false,
    );

    // 5. 等待终局(REPORT or 超时)
    let timeout_secs = 300u64; // 5 分钟
    let start = std::time::Instant::now();
    let n_reviewers = sc.reviewers.len();
    loop {
        std::thread::sleep(std::time::Duration::from_secs(5));
        let letters = hub.journal.replay();
        let has_report = letters
            .iter()
            .any(|l| l.from == "judge" && l.body.starts_with("REPORT="));
        let elapsed = start.elapsed().as_secs();
        if has_report {
            println!("✅ judge 已收尾 ({elapsed}s, {} 封信)", letters.len());
            break;
        }
        if elapsed > timeout_secs {
            println!("⏰ 超时 {timeout_secs}s ({} 封信)", letters.len());
            break;
        }
        if elapsed % 30 == 0 {
            println!("  等待中: {elapsed}s, {} 封信", letters.len());
        }
    }

    // 6. 判分
    let letters = hub.journal.replay();
    let complete = report_is_final(&letters, "judge", "review", n_reviewers, n_reviewers);
    let hidden = &sc.task.hidden_keyword;
    let final_script_ok = letters
        .iter()
        .rev()
        .find(|l| {
            l.from == "writer" && (l.body.starts_with("DRAFT=") || l.body.starts_with("REVISED="))
        })
        .map(|l| l.body.contains(hidden))
        .unwrap_or(false);
    let state = if complete && final_script_ok {
        "PASS"
    } else if final_script_ok {
        "DEGENERATE"
    } else {
        "FAIL"
    };
    let turns: u64 = hub
        .shops
        .values()
        .map(|s| s.turns_done.load(std::sync::atomic::Ordering::Relaxed))
        .sum();

    format!(
        "scenario: {} (codex)\nstate: {}\nletters: {}\nturns: {}\ncomplete_loop: {}\nhidden_check: {}\n",
        sc.meta.name, state, letters.len(), turns, complete, final_script_ok
    )
}

/// 编排器驱动的场景运行: 替代硬编码 run() 的新路径
fn run_orchestrated(sc: &Scenario, run_dir: &str) -> String {
    use pulsing_forge_core::{FinalState, Orchestrator, OrchestratorAction, OrchestratorCtx};

    let _ = std::fs::remove_dir_all(run_dir);
    std::fs::create_dir_all(run_dir).expect("创建目录");

    // 根据编排器类型构建店册和编排器实例
    let (shops, orch): (Vec<String>, Box<dyn Orchestrator>) = match sc.orchestrator.as_str() {
        "pipeline" => {
            let stages: Vec<(String, String)> = sc
                .stages
                .iter()
                .map(|s| (s.shop.clone(), s.duty.clone()))
                .collect();
            let shop_names: Vec<String> = stages.iter().map(|(s, _)| s.clone()).collect();
            (
                shop_names,
                Box::new(pulsing_forge_core::PipelineOrchestrator { stages }),
            )
        }
        "debate" => {
            let shops = vec![
                "judge".to_string(),
                "pro_a".to_string(),
                "pro_b".to_string(),
            ];
            (
                shops,
                Box::new(pulsing_forge_core::DebateOrchestrator {
                    judge: "judge".to_string(),
                    topic: if sc.debate_topic.is_empty() {
                        "多agent协作利大于弊".to_string()
                    } else {
                        sc.debate_topic.clone()
                    },
                }),
            )
        }
        "saboteur" => {
            let shops = vec![
                "judge".to_string(),
                "coder".to_string(),
                "breaker".to_string(),
                "verifier_a".to_string(),
                "verifier_b".to_string(),
            ];
            (
                shops,
                Box::new(pulsing_forge_core::SaboteurOrchestrator {
                    judge: "judge".to_string(),
                }),
            )
        }
        "relay" => {
            let stages: Vec<(String, String)> = sc
                .stages
                .iter()
                .map(|s| (s.shop.clone(), s.duty.clone()))
                .collect();
            let shop_names: Vec<String> = stages.iter().map(|(s, _)| s.clone()).collect();
            let input = sc.meta.relay_input.clone();
            (
                shop_names,
                Box::new(pulsing_forge_core::RelayChainOrchestrator::new(
                    stages, &input,
                )),
            )
        }
        "broadcast" => {
            let mut shops = vec!["chief".to_string()];
            shops.extend((1..=10).map(|i| format!("e{}", i)));
            (
                shops,
                Box::new(pulsing_forge_core::BroadcastOrchestrator {
                    chief: "chief".into(),
                    echo_shops: (1..=10).map(|i| format!("e{}", i)).collect(),
                    message: "报名".into(),
                }),
            )
        }
        "review" | "legacy" => {
            // 默认走旧写审路径(确定性层)
            return run(sc, run_dir);
        }
        other => {
            eprintln!("未知编排器: {other}, 可选: review/pipeline/debate/saboteur");
            std::process::exit(2);
        }
    };

    // 创建确定性 Hub
    let acl_rules: Vec<(String, Vec<String>)> = match sc.orchestrator.as_str() {
        "pipeline" => {
            let mut rules = Vec::new();
            for i in 0..shops.len() {
                let mut allow = vec![shops[i].clone()]; // 自己
                if i + 1 < shops.len() {
                    allow.push(shops[i + 1].clone());
                } // 下游
                rules.push((shops[i].clone(), allow));
            }
            rules
        }
        "debate" => vec![
            ("judge".into(), vec!["*".into()]),
            ("pro_a".into(), vec!["judge".into()]),
            ("pro_b".into(), vec!["judge".into()]),
        ],
        "saboteur" => vec![
            ("judge".into(), vec!["*".into()]),
            ("coder".into(), vec!["judge".into()]),
            ("breaker".into(), vec!["judge".into()]),
            ("verifier_a".into(), vec!["judge".into()]),
            ("verifier_b".into(), vec!["judge".into()]),
        ],
        "relay" => {
            let mut rules = Vec::new();
            for i in 0..shops.len() {
                let mut allow = vec!["orchestrator".to_string(), shops[i].clone()];
                if i + 1 < shops.len() {
                    allow.push(shops[i + 1].clone());
                }
                rules.push((shops[i].clone(), allow));
            }
            rules
        }
        _ => vec![],
    };

    let mut hub = Hub::new(run_dir, shops.clone(), Acl { rules: acl_rules }).expect("Hub");

    // 注册确定性店员(简化版: 按编排器类型行为)
    match sc.orchestrator.as_str() {
        "pipeline" => {
            for (i, shop) in shops.iter().enumerate() {
                let is_last = i == shops.len() - 1;
                let next = if !is_last {
                    shops[i + 1].clone()
                } else {
                    shop.clone()
                };
                let shop2 = shop.clone();
                hub.register(
                    shop,
                    Box::new(move |trig, _| {
                        if trig.body.starts_with("TASK=") || trig.body.starts_with("WORK=") {
                            let excerpt: String = trig.body.chars().take(40).collect();
                            if is_last {
                                vec![OutLetter {
                                    to: shop2.clone(),
                                    body: format!("REPORT=RESULT=流水线完成||{excerpt}"),
                                    silent: true,
                                }]
                            } else {
                                vec![OutLetter {
                                    to: next.clone(),
                                    body: format!("WORK=工序完成||{excerpt}"),
                                    silent: false,
                                }]
                            }
                        } else {
                            vec![]
                        }
                    }),
                );
            }
        }
        "debate" => {
            for (shop, side) in [("pro_a", "支持"), ("pro_b", "反对")] {
                let j = "judge".to_string();
                let s = side.to_string();
                let shop2 = shop.to_string();
                hub.register(
                    shop,
                    Box::new(move |trig, _| {
                        if trig.body.starts_with("ASK=") {
                            vec![OutLetter {
                                to: j.clone(),
                                body: format!("ARG={}||{}论据: 多agent协作{}", s, s, s),
                                silent: false,
                            }]
                        } else {
                            vec![]
                        }
                    }),
                );
            }
            // judge: 收到TASK发ASK, 收到ARBIT发REPORT
            let j2 = "judge".to_string();
            hub.register(
                "judge",
                Box::new(move |trig, _| {
                    if trig.body.starts_with("TASK=") {
                        vec![
                            OutLetter {
                                to: "pro_a".into(),
                                body: "ASK=请给支持论据".into(),
                                silent: false,
                            },
                            OutLetter {
                                to: "pro_b".into(),
                                body: "ASK=请给反对论据".into(),
                                silent: false,
                            },
                        ]
                    } else if trig.body.starts_with("ARBIT=") {
                        vec![OutLetter {
                            to: j2.clone(),
                            body: "REPORT=辩论完成||正反论据已收||裁定: 协作有利".into(),
                            silent: true,
                        }]
                    } else {
                        vec![]
                    }
                }),
            );
        }
        "saboteur" => {
            hub.register(
                "coder",
                Box::new(|trig, _| {
                    if trig.body.starts_with("TASK=") {
                        vec![OutLetter {
                            to: "judge".into(),
                            body: "RESULT=coder||1..30||9455".into(),
                            silent: false,
                        }]
                    } else {
                        vec![]
                    }
                }),
            );
            hub.register(
                "breaker",
                Box::new(|trig, _| {
                    if trig.body.starts_with("TASK=") {
                        vec![OutLetter {
                            to: "judge".into(),
                            body: "RESULT=breaker||31..60||99999".into(),
                            silent: false,
                        }] // 故意错
                    } else {
                        vec![]
                    }
                }),
            );
            for v in ["verifier_a", "verifier_b"] {
                hub.register(
                    v,
                    Box::new(|trig, _| {
                        if trig.body.starts_with("CHECK=") {
                            vec![OutLetter {
                                to: "judge".into(),
                                body: "VERIFY=1..60||73810".into(),
                                silent: false,
                            }]
                        } else {
                            vec![]
                        }
                    }),
                );
            }
            let j2 = "judge".to_string();
            hub.register(
                "judge",
                Box::new(move |trig, _| {
                    if trig.body.starts_with("TASK=") {
                        vec![
                            OutLetter {
                                to: "coder".into(),
                                body: "TASK=计算1..30平方和".into(),
                                silent: false,
                            },
                            OutLetter {
                                to: "breaker".into(),
                                body: "TASK=计算31..60平方和".into(),
                                silent: false,
                            },
                            OutLetter {
                                to: "verifier_a".into(),
                                body: "CHECK=独立核算1..60全程平方和".into(),
                                silent: false,
                            },
                            OutLetter {
                                to: "verifier_b".into(),
                                body: "CHECK=独立核算1..60全程平方和".into(),
                                silent: false,
                            },
                        ]
                    } else if trig.body.starts_with("ARBIT=") {
                        vec![OutLetter {
                            to: j2.clone(),
                            body: "REPORT=TOTAL=73810||内鬼已被纠正".into(),
                            silent: true,
                        }]
                    } else {
                        vec![]
                    }
                }),
            );
        }
        "relay" => {
            for (i, shop) in shops.iter().enumerate() {
                let is_last = i == shops.len() - 1;
                let next = if !is_last {
                    shops[i + 1].clone()
                } else {
                    shop.clone()
                };
                let shop2 = shop.clone();
                hub.register(
                    shop,
                    Box::new(move |trig, _| {
                        if trig.body.starts_with("TASK=") || trig.body.starts_with("RELAY=") {
                            if is_last {
                                vec![OutLetter {
                                    to: shop2.clone(),
                                    body: "REPORT=RELAY_RESULT=最终文本".to_string(),
                                    silent: true,
                                }]
                            } else {
                                vec![OutLetter {
                                    to: next.clone(),
                                    body: "RELAY=变换后文本".to_string(),
                                    silent: false,
                                }]
                            }
                        } else {
                            vec![]
                        }
                    }),
                );
            }
        }
        "broadcast" => {
            for shop in shops.iter().skip(1) {
                // skip chief
                let chief = "chief".to_string();
                let shop2 = shop.clone();
                hub.register(
                    shop,
                    Box::new(move |trig, _| {
                        if trig.body.starts_with("ECHO=") {
                            vec![OutLetter {
                                to: chief.clone(),
                                body: format!("ECHO={}|收到", shop2),
                                silent: true,
                            }]
                        } else {
                            vec![]
                        }
                    }),
                );
            }
            // chief: 广播 + 收齐后 REPORT
            let shops2 = shops.clone();
            hub.register(
                "chief",
                Box::new(move |trig, _| {
                    if trig.body.starts_with("TASK=") {
                        // 直接广播到所有回声店
                        let targets: Vec<String> = shops2.iter().skip(1).cloned().collect();
                        targets
                            .iter()
                            .map(|t| OutLetter {
                                to: t.clone(),
                                body: "ECHO=报名".to_string(),
                                silent: false,
                            })
                            .collect()
                    } else if trig.body.starts_with("ARBIT=") {
                        vec![OutLetter {
                            to: "chief".into(),
                            body: format!("REPORT=COUNT={}|全部回声", shops2.len() - 1),
                            silent: true,
                        }]
                    } else {
                        vec![]
                    }
                }),
            );
        }
        _ => {}
    }

    // 预置规程信
    let empty_letters = vec![];
    let empty_shops = shops.clone();
    let setup_ctx = OrchestratorCtx {
        letters: &empty_letters,
        shops: &empty_shops,
    };
    for s in orch.setup(&setup_ctx) {
        let _ = hub.send(&s.shop, &s.to, &s.body, s.silent);
    }

    // 点火
    let (ign_to, ign_body) = orch.ignition();
    hub.ignite(&ign_to, &format!("{ign_body} 任务: {}", sc.task.template))
        .unwrap();

    // 编排器回调循环: 检查最新信、触发闸门
    let mut final_state = FinalState::Timeout;
    for _tick in 0..100 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let letters = hub.letters();
        let ctx = OrchestratorCtx {
            letters: &letters,
            shops: &shops,
        };
        let mut injected = false;
        if let Some(last) = letters.last() {
            let actions = orch.on_letter(last, &ctx);
            for act in actions {
                match act {
                    OrchestratorAction::Inject {
                        from,
                        to,
                        body,
                        silent,
                    } => {
                        let _ = hub.send(&from, &to, &body, silent);
                        injected = true; // 注入可能同步触发新回合 → 需重读信件
                    }
                    OrchestratorAction::Complete { state, detail } => {
                        final_state = match state.as_str() {
                            "PASS" => FinalState::Pass,
                            _ => FinalState::Degenerate(detail),
                        };
                    }
                    _ => {}
                }
            }
        }
        // 注入后重读信件(确定性 Hub 同步触发新回合, REPORT 已产生)
        let letters2 = if injected { hub.letters() } else { letters };
        let ctx2 = OrchestratorCtx {
            letters: &letters2,
            shops: &shops,
        };
        let state = orch.finalize(&ctx2);
        if state != FinalState::Timeout {
            final_state = state;
            break;
        }
    }

    let letters = hub.letters();
    let turns: u64 = hub.turns.values().sum();
    let state_str = match &final_state {
        FinalState::Pass => "PASS".to_string(),
        FinalState::Degenerate(d) => format!("DEGENERATE({d})"),
        FinalState::Fail(f) => format!("FAIL({f})"),
        FinalState::Timeout => "TIMEOUT".to_string(),
    };
    format!(
        "scenario: {} ({})\nstate: {}\nletters: {}\nturns: {}\n",
        sc.meta.name,
        sc.orchestrator,
        state_str,
        letters.len(),
        turns
    )
}
