// codex 适配器 — 把 forge-cli 的潮汐层接到 pulsing-mcp 真栈。
// 调用链: forge-cli → 控制口(TCP) → pulsing-mcp shard → codex exec → LLM

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::Command;
use std::time::Duration;

pub struct CodexAdapter {
    pub binary: String,
    pub control_port: u16,
    pub listen_addr: String,
    pub log_dir: String,
    pub provider_path: String,
    pub codex_bin: String,
    pub shops: Vec<String>,
    pub acl_json: String,
    pub max_turns: u32,
}

impl CodexAdapter {
    pub fn deploy(&self) -> Result<(), String> {
        std::fs::create_dir_all(format!("{}/config", self.log_dir)).map_err(|e| e.to_string())?;
        let _ = Command::new("sh")
            .arg("-c")
            .arg(format!(
                "pkill -f 'pulsing-mcp shard --team {}/t[e]am' 2>/dev/null; sleep 2",
                self.log_dir
            ))
            .output();
        let mut toml = String::from("shop = \"starter\"\n");
        for s in &self.shops {
            toml.push_str(&format!(
                "\n[[worker]]\nname = '{}'\nrole = '{}'\nmodel = \"glm-5-turbo\"\nlisten = '{}'\n",
                s, s, self.listen_addr
            ));
        }
        std::fs::write(format!("{}/team.toml", self.log_dir), toml).map_err(|e| e.to_string())?;
        let env = format!(
            "CODEX_PULSING_LOG_DIR={} CODEX_PULSING_PROVIDER={} CODEX_PULSING_CODEX_BIN={} \
             CODEX_PULSING_CLERK_LIMIT=6 CODEX_PULSING_MAX_TURNS={} \
             CODEX_PULSING_LISTEN={} CODEX_PULSING_SHOPS={} CODEX_PULSING_ACL='{}'",
            self.log_dir,
            self.provider_path,
            self.codex_bin,
            self.max_turns,
            self.listen_addr,
            self.shops.join(","),
            self.acl_json
        );
        Command::new("sh")
            .arg("-c")
            .arg(format!(
                "env {} setsid nohup {} shard --team {}/team.toml > {}/shard0.log 2>&1 < /dev/null &",
                env, self.binary, self.log_dir, self.log_dir
            ))
            .output()
            .map_err(|e| e.to_string())?;
        std::thread::sleep(Duration::from_secs(6));
        let log =
            std::fs::read_to_string(format!("{}/shard0.log", self.log_dir)).unwrap_or_default();
        if log.contains("listen") {
            Ok(())
        } else {
            Err(format!(
                "shard 未就绪: {}",
                log.lines().last().unwrap_or("?")
            ))
        }
    }

    pub fn send(&self, shop: &str, to: &str, msg: &str, silent: bool) -> Result<String, String> {
        let mut stream =
            TcpStream::connect(("127.0.0.1", self.control_port)).map_err(|e| e.to_string())?;
        stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
        stream.set_write_timeout(Some(Duration::from_secs(10))).ok();
        let body = serde_json::json!({
            "tool": "pulsing_send",
            "args": {"_shop": shop, "to": to, "message": msg, "silent": silent}
        });
        stream
            .write_all(format!("{body}\n").as_bytes())
            .map_err(|e| e.to_string())?;
        let mut buf = [0u8; 4096];
        let n = stream.read(&mut buf).map_err(|e| e.to_string())?;
        Ok(String::from_utf8_lossy(&buf[..n]).trim().to_string())
    }

    pub fn ignite(&self, shop: &str, msg: &str) -> Result<String, String> {
        self.send(shop, shop, msg, false)
    }

    pub fn journal(&self) -> Vec<serde_json::Value> {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.log_dir) {
            for entry in rd.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if !name.starts_with("events-") || !name.ends_with(".jsonl") {
                    continue;
                }
                if let Ok(content) = std::fs::read_to_string(entry.path()) {
                    for line in content.lines() {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                            if v["ev"] == "reply_out" {
                                out.push(v);
                            }
                        }
                    }
                }
            }
        }
        out.sort_by_key(|v| v["ts_ms"].as_u64().unwrap_or(0));
        out
    }

    pub fn shutdown(&self) {
        let _ = Command::new("sh")
            .arg("-c")
            .arg(format!(
                "pkill -f 'pulsing-mcp shard --team {}/t[e]am' 2>/dev/null",
                self.log_dir
            ))
            .output();
    }
}
