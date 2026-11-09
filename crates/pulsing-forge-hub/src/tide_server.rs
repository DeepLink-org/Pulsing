//! forge-runtime::tide_server — TCP 控制口 + 回合调度器。
//! codex 的 MCP 工具调用经 attach(TCP) 到达这里; 非静默信触发回合。

use crate::tide::TideHub;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

/// 控制口单条命令的回执
#[derive(serde::Serialize)]
pub struct ToolReply {
    pub ok: bool,
    pub result: String,
    pub error: String,
}

impl ToolReply {
    fn ok(result: &str) -> Self {
        ToolReply {
            ok: true,
            result: result.to_string(),
            error: String::new(),
        }
    }
    fn err(e: &str) -> Self {
        ToolReply {
            ok: false,
            result: String::new(),
            error: e.to_string(),
        }
    }
}

/// 处理一行控制命令, 返回 JSON 回执
pub fn handle_line(hub: &TideHub, line: &str) -> String {
    let v: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            return serde_json::to_string(&ToolReply::err(&format!("JSON解析: {e}"))).unwrap()
        }
    };
    let tool = v["tool"].as_str().unwrap_or("");
    let args = &v["args"];
    let from_codex = args.get("_from_codex").is_some();

    match tool {
        "pulsing_send" => {
            let to = args["to"].as_str().unwrap_or("");
            let msg = args["message"].as_str().unwrap_or("");
            let silent = args["silent"].as_bool().unwrap_or(false);
            // _shop 是调用方身份(attach 注入)
            let from = args["_shop"].as_str().unwrap_or("unknown");
            match hub.send(from, to, msg, silent, from_codex) {
                Ok(_) => serde_json::to_string(&ToolReply::ok("已送达并记账")).unwrap(),
                Err(e) => serde_json::to_string(&ToolReply::err(&e)).unwrap(),
            }
        }
        "pulsing_broadcast" => {
            let from = args["_shop"].as_str().unwrap_or("unknown");
            let msg = args["message"].as_str().unwrap_or("");
            let silent = args["silent"].as_bool().unwrap_or(false);
            let targets: Vec<String> = args["to_list"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            if targets.is_empty() {
                return serde_json::to_string(&ToolReply::err("缺 to_list")).unwrap();
            }
            let sent = hub.broadcast(from, &targets, msg, silent, from_codex);
            serde_json::to_string(&ToolReply::ok(&format!("已群发 {} 店", sent.len()))).unwrap()
        }
        "pulsing_drain_mail" => {
            let shop = args["_shop"].as_str().unwrap_or("unknown");
            let letters = hub.drain(shop);
            // 按字符截断(中文信切字节会panic)
            let bodies: Vec<String> = letters
                .iter()
                .map(|l| {
                    format!(
                        "[{}→{}] {}",
                        l.from,
                        l.to,
                        l.body.chars().take(200).collect::<String>()
                    )
                })
                .collect();
            serde_json::to_string(&ToolReply::ok(&bodies.join("\n---\n"))).unwrap()
        }
        "pulsing_recall" => {
            // 链式召回(Super-agent 吸收): 按序号取全文, 或按关键词搜信
            // (SA2挑战实证: 只按ID召回, 模型在33次召回里找不到目标信 — 需要搜索)
            let id = args["letter_id"].as_u64().unwrap_or(0);
            let query = args["query"].as_str().unwrap_or("");
            let caller = args["_shop"].as_str().unwrap_or("unknown");
            hub.journal.append_tool(
                caller,
                &format!("pulsing_recall letter_id={id} query={query}"),
            );
            if !query.is_empty() {
                let hits: Vec<String> = hub
                    .journal
                    .replay()
                    .into_iter()
                    .filter(|l| l.body.contains(query))
                    .take(5)
                    .map(|l| {
                        format!(
                            "[信#{}] {}→{}: {}",
                            l.seq,
                            l.from,
                            l.to,
                            l.body.chars().take(80).collect::<String>()
                        )
                    })
                    .collect();
                if hits.is_empty() {
                    serde_json::to_string(&ToolReply::err(&format!("账本中无含\"{query}\"的信")))
                        .unwrap()
                } else {
                    serde_json::to_string(&ToolReply::ok(&format!(
                        "命中{}封(取全文用 letter_id):\n{}",
                        hits.len(),
                        hits.join("\n")
                    )))
                    .unwrap()
                }
            } else {
                match hub.journal.replay().into_iter().find(|l| l.seq == id) {
                    Some(l) => serde_json::to_string(&ToolReply::ok(&format!(
                        "[信#{seq}] 来自 {from} → {to}: {body}",
                        seq = l.seq,
                        from = l.from,
                        to = l.to,
                        body = l.body
                    )))
                    .unwrap(),
                    None => serde_json::to_string(&ToolReply::err(&format!(
                        "信件#{id}不存在(可改用 query=关键词 搜索)"
                    )))
                    .unwrap(),
                }
            }
        }
        "pulsing_health" => {
            let shops = hub.shops.len();
            let turns: u64 = hub
                .shops
                .values()
                .map(|s| s.turns_done.load(std::sync::atomic::Ordering::Relaxed))
                .sum();
            serde_json::to_string(&ToolReply::ok(&format!(
                "店={shops} 回合={turns} 账本写失败={}",
                hub.journal
                    .write_fails
                    .load(std::sync::atomic::Ordering::Relaxed)
            )))
            .unwrap()
        }
        _ => serde_json::to_string(&ToolReply::err(&format!("未知工具: {tool}"))).unwrap(),
    }
}

/// TCP 控制口服务器: 每连接一行命令一行回执
/// FORGE_BIND=0.0.0.0 时对外监听(forge cell: 远程沙盒的 worker attach 进来)
pub async fn serve_control(hub: Arc<TideHub>, port: u16) -> anyhow::Result<()> {
    let bind = std::env::var("FORGE_BIND").unwrap_or_else(|_| "127.0.0.1".to_string());
    let listener = TcpListener::bind((bind.as_str(), port)).await?;
    println!("tide 控制口 listen={bind}:{port}");
    loop {
        let (socket, _) = listener.accept().await?;
        let hub = hub.clone();
        tokio::spawn(async move {
            let (rd, mut wr) = socket.into_split();
            let mut reader = BufReader::new(rd);
            let mut line = String::new();
            if reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                let reply = handle_line(&hub, line.trim());
                let _ = wr.write_all(format!("{reply}\n").as_bytes()).await;
            }
        });
    }
}

/// 回合调度: 只触发点火之后的信(跳过 setup 规程信 — 否则模型把规程文本当任务输入)
pub async fn shop_scheduler(hub: Arc<TideHub>, shop: String) {
    shop_scheduler_from(hub, shop, 0).await;
}

pub async fn shop_scheduler_from(hub: Arc<TideHub>, shop: String, skip_before: u64) {
    let mut last_ts: u64 = skip_before;
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let letters = hub
            .mailboxes
            .get(&shop)
            .map(|m| m.lock().unwrap().clone())
            .unwrap_or_default();
        let has_new = letters
            .iter()
            .any(|l| !l.body.starts_with("KICK=") && l.ts_ms > last_ts);
        if !has_new {
            continue;
        }

        // 去抖: 等信流入静默(2秒无新信)再触发, 扇入风暴时攒一批一次处理
        // (30店广播时逐封触发会烧穿收单者的回合预算与配额 — taxonomy #7 扇入风暴)
        let mut settled = letters;
        let mut quiet_polls = 0u32;
        let mut total_polls = 0u32;
        while quiet_polls < 4 && total_polls < 12 {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            total_polls += 1;
            let cur = hub
                .mailboxes
                .get(&shop)
                .map(|m| m.lock().unwrap().clone())
                .unwrap_or_default();
            let arrived = cur.len() > settled.len()
                || cur.iter().any(|l| {
                    !l.body.starts_with("KICK=")
                        && l.ts_ms > last_ts
                        && !settled
                            .iter()
                            .any(|s| s.ts_ms == l.ts_ms && s.body == l.body)
                });
            if arrived {
                quiet_polls = 0;
                settled = cur;
            } else {
                quiet_polls += 1;
            }
        }

        if let Some(t) = settled.iter().rev().find(|l| !l.body.starts_with("KICK=")) {
            last_ts = t.ts_ms;
        }
        // 清空信箱但保留全部信件用于 turn_prompt(规程信不丢)
        let all_letters = settled;
        if let Some(mb) = hub.mailboxes.get(&shop) {
            mb.lock().unwrap().clear();
        }
        let h2 = hub.clone();
        let s2 = shop.clone();
        let _ = tokio::task::spawn_blocking(move || h2.run_turn_full(&s2, &all_letters)).await;
    }
}

/// 启动完整潮汐层: 控制口 + 全部店调度器
pub async fn run_tide(hub: Arc<TideHub>) -> anyhow::Result<()> {
    let port = hub.listen_port;
    // 控制口
    let ctrl = tokio::spawn(serve_control(hub.clone(), port));
    // 店调度器
    for shop in hub.shops.keys() {
        let hub2 = hub.clone();
        let s = shop.clone();
        tokio::spawn(shop_scheduler(hub2, s));
    }
    ctrl.await??;
    Ok(())
}

/// attach: MCP stdio → TCP 控制口桥
/// codex 的 MCP 子进程; 读 stdin JSON-RPC, 转发工具调用到控制口, 写回 stdout
/// control_addr: "端口"(→127.0.0.1) 或完整 "ip:port"(forge cell 远程信箱)
pub async fn run_attach(control_addr: String, shop: Option<String>) -> anyhow::Result<()> {
    use tokio::io::AsyncReadExt;
    let mut stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];

    loop {
        let n = stdin.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        // 按行分割处理
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let line_str = String::from_utf8_lossy(&line).trim().to_string();
            if line_str.is_empty() {
                continue;
            }

            let msg: serde_json::Value = match serde_json::from_str(&line_str) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let method = msg["method"].as_str().unwrap_or("");
            let id = msg.get("id").cloned();

            let response = match method {
                "initialize" => Some(serde_json::json!({
                    "jsonrpc": "2.0", "id": id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "forge-tide", "version": "0.1.0"}
                    }
                })),
                "notifications/initialized" | "notifications/cancelled" => None,
                "ping" => Some(serde_json::json!({
                    "jsonrpc": "2.0", "id": id, "result": {}
                })),
                "tools/list" => Some(serde_json::json!({
                    "jsonrpc": "2.0", "id": id,
                    "result": {"tools": tools_schema()}
                })),
                "tools/call" => {
                    let params = &msg["params"];
                    let tool = params["name"].as_str().unwrap_or("");
                    let mut args = params["arguments"].clone();
                    // 注入调用方身份
                    if let (Some(s), Some(o)) = (shop.as_ref(), args.as_object_mut()) {
                        o.insert("_shop".into(), serde_json::json!(s));
                        o.insert("_from_codex".into(), serde_json::json!(true));
                    }
                    // 转发到控制口
                    let fwd = serde_json::json!({"tool": tool, "args": args});
                    let reply = forward_to_control(&control_addr, &fwd.to_string()).await;
                    Some(serde_json::json!({
                        "jsonrpc": "2.0", "id": id,
                        "result": {"content": [{"type": "text", "text": reply}]}
                    }))
                }
                _ => Some(serde_json::json!({
                    "jsonrpc": "2.0", "id": id,
                    "error": {"code": -32601, "message": format!("Unknown method: {method}")}
                })),
            };

            if let Some(resp) = response {
                let mut out = serde_json::to_string(&resp).unwrap();
                out.push('\n');
                let _ = stdout.write_all(out.as_bytes()).await;
                let _ = stdout.flush().await;
            }
        }
    }
    Ok(())
}

async fn forward_to_control(addr: &str, line: &str) -> String {
    match TcpStream::connect(addr).await {
        Ok(mut stream) => {
            use tokio::io::AsyncWriteExt;
            if stream
                .write_all(format!("{line}\n").as_bytes())
                .await
                .is_err()
            {
                return r#"{"ok":false,"error":"控制口写入失败"}"#.to_string();
            }
            let mut buf = Vec::new();
            use tokio::io::AsyncReadExt;
            let _ = stream.read_to_end(&mut buf).await;
            String::from_utf8_lossy(&buf).trim().to_string()
        }
        Err(e) => format!(r#"{{"ok":false,"error":"控制口连接失败: {e}"}}"#),
    }
}

fn tools_schema() -> serde_json::Value {
    serde_json::json!([
        {"name": "pulsing_send",
         "description": "发送一封信给指定店",
         "inputSchema": {"type": "object", "properties": {
             "to": {"type": "string", "description": "收件店名"},
             "message": {"type": "string", "description": "信件正文"},
             "silent": {"type": "boolean", "description": "true=只入箱不唤醒"}
         }, "required": ["to", "message"]}},
        {"name": "pulsing_broadcast",
         "description": "群发一封信给多个店",
         "inputSchema": {"type": "object", "properties": {
             "to_list": {"type": "array", "items": {"type": "string"}},
             "message": {"type": "string"},
             "silent": {"type": "boolean"}
         }, "required": ["to_list", "message"]}},
        {"name": "pulsing_drain_mail",
         "description": "取走并清空自己信箱里的全部信",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "pulsing_recall",
         "description": "按信件序号取回全文, 或按关键词搜索账本信件(返回序号+摘要)",
         "inputSchema": {"type": "object", "properties": {
             "letter_id": {"type": "integer", "description": "信#后的数字序号(取全文)"},
             "query": {"type": "string", "description": "关键词(搜索含它的信, 与letter_id二选一)"}
         }}},
        {"name": "pulsing_health",
         "description": "查看系统健康状态",
         "inputSchema": {"type": "object", "properties": {}}},
    ])
}
