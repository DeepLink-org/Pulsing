//! forge-runtime — 确定性协调运行时: 信箱中枢 + 账本 + ACL/守卫 + 脚本店员。
//! 两层运行时之一: 本 crate 是"确定性层"(无LLM, 可CI), 潮汐层(pulsing-mcp)经
//! adapter 接入同一场景。元原则不变: 机器管协议, 店员管语义。

use pulsing_forge_core::Letter;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

/// ACL: 前缀通配的收件白名单。空表 = 不限制。
/// 收录实测三类守卫: 越权拒绝(报错教模型)、自回声熔断、广播收拢。
#[derive(Debug, Default, Clone)]
pub struct Acl {
    /// 规则表: 键为店名或前缀*(如 "review_*"), 值为允许收件人(可含前缀*/"*")
    pub rules: Vec<(String, Vec<String>)>,
}

impl Acl {
    pub fn from_pairs(pairs: &[(&str, &[&str])]) -> Self {
        Acl {
            rules: pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.iter().map(|s| s.to_string()).collect()))
                .collect(),
        }
    }
    fn matches(pat: &str, name: &str) -> bool {
        pat.strip_suffix('*')
            .map(|p| name.starts_with(p))
            .unwrap_or(pat == name)
    }
    fn allowed_for(&self, from: &str) -> Option<&Vec<String>> {
        self.rules
            .iter()
            .find(|(k, _)| Acl::matches(k, from))
            .map(|(_, v)| v)
    }
    /// 越权检查: 返回 Err(教学性错误) 或 Ok(收拢后的目标)。
    /// broadcast 用 `collapse`: 白名单外目标自动略去(报错会引发重试风暴, taxonomy #2)。
    pub fn check_send(&self, from: &str, to: &str) -> Result<(), String> {
        match self.allowed_for(from) {
            None => Ok(()),
            Some(allow) if allow.iter().any(|t| Acl::matches(t, to)) => Ok(()),
            Some(allow) => Err(format!(
                "收件人 {to:?} 不在 {from:?} 的白名单里; 允许: {allow:?}"
            )),
        }
    }
    pub fn collapse(&self, from: &str, targets: &[String]) -> Vec<String> {
        match self.allowed_for(from) {
            None => targets.to_vec(),
            Some(allow) => targets
                .iter()
                .filter(|t| allow.iter().any(|p| Acl::matches(p, t)))
                .cloned()
                .collect(),
        }
    }
}

/// 账本: 每店一个 JSONL 文件, exactly-once 追加。
pub struct Journal {
    dir: PathBuf,
    pub write_fails: std::sync::atomic::AtomicU64,
    next_seq: std::sync::atomic::AtomicU64,
}

impl Journal {
    pub fn new(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        let j = Journal {
            dir,
            write_fails: std::sync::atomic::AtomicU64::new(0),
            next_seq: std::sync::atomic::AtomicU64::new(1),
        };
        // 回放存量账本, 把 seq 接续到最大值之后(支持跨进程追加不重号)
        let max_seq = j.replay().iter().map(|l| l.seq).max().unwrap_or(0);
        j.next_seq
            .store(max_seq + 1, std::sync::atomic::Ordering::SeqCst);
        Ok(j)
    }
    pub fn append(&self, l: &mut Letter) {
        l.seq = self
            .next_seq
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let path = self.dir.join(format!("events-{}.jsonl", l.from));
        let line = serde_json::json!({
            "seq": l.seq, "ts_ms": l.ts_ms, "from": l.from, "to": l.to, "body": l.body,
            "ev": "reply_out",
        });
        let r = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| writeln!(f, "{line}"));
        if r.is_err() {
            self.write_fails
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    /// codex stderr 落账(潮汐层诊断: 无工具调用秒退的死因原文)
    pub fn append_stderr(&self, shop: &str, stderr: &str) {
        let path = self.dir.join(format!("events-{shop}.jsonl"));
        let line = serde_json::json!({
            "ev": "codex_err", "shop": shop,
            "stderr": stderr.chars().take(1500).collect::<String>(),
        });
        let _ = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| writeln!(f, "{line}"));
    }
    /// 工具调用落账(可观测性: recall 等无信件副作用的调用也留痕)
    pub fn append_tool(&self, shop: &str, detail: &str) {
        let path = self.dir.join(format!("events-{shop}.jsonl"));
        let line = serde_json::json!({
            "ev": "tool_call", "shop": shop,
            "detail": detail.chars().take(300).collect::<String>(),
        });
        let _ = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| writeln!(f, "{line}"));
    }
    pub fn replay(&self) -> Vec<Letter> {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            let mut files: Vec<_> = rd.flatten().map(|e| e.path()).collect();
            files.sort();
            for f in files {
                let name = f
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                if !name.starts_with("events-") || !name.ends_with(".jsonl") {
                    continue;
                }
                if let Ok(content) = std::fs::read_to_string(&f) {
                    for line in content.lines() {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                            if v["ev"] == "reply_out" {
                                out.push(Letter {
                                    seq: v["seq"].as_u64().unwrap_or(0),
                                    ts_ms: v["ts_ms"].as_u64().unwrap_or(0),
                                    from: v["from"].as_str().unwrap_or("").into(),
                                    to: v["to"].as_str().unwrap_or("").into(),
                                    body: v["body"].as_str().unwrap_or("").into(),
                                });
                            }
                        }
                    }
                }
            }
        }
        out.sort_by_key(|l| l.ts_ms);
        out
    }
}

/// 脚本店员: 一个确定性函数即一个 agent(无LLM, 可CI)。
/// 输入=唤醒信+账本快照, 输出=要发的信。
pub type AgentFn = Box<dyn Fn(&Letter, &[Letter]) -> Vec<OutLetter>>;

#[derive(Debug, Clone)]
pub struct OutLetter {
    pub to: String,
    pub body: String,
    pub silent: bool,
}

/// 中枢: 店册+信箱+守卫+账本。drain 为破坏性读(taxonomy #3 的根源在产品里
/// 由"账本即登记处"原则对冲: 一切回看走 replay())。
pub struct Hub {
    pub shops: Vec<String>,
    pub acl: Acl,
    mailboxes: HashMap<String, Vec<Letter>>,
    pub journal: Journal,
    pub agents: HashMap<String, AgentFn>,
    pub turns: HashMap<String, u64>,
    now_ms: u64,
    /// 审查店→仲裁者回信自动静默(taxonomy #4: ACK乒乓熔断)
    pub auto_silence: Option<(String, String)>, // (prefix, arbitrator)
}

impl Hub {
    pub fn new(dir: impl Into<PathBuf>, shops: Vec<String>, acl: Acl) -> std::io::Result<Self> {
        let mut mailboxes = HashMap::new();
        for s in &shops {
            mailboxes.insert(s.clone(), Vec::new());
        }
        Ok(Hub {
            shops,
            acl,
            mailboxes,
            journal: Journal::new(dir)?,
            agents: HashMap::new(),
            turns: HashMap::new(),
            now_ms: 0,
            auto_silence: None,
        })
    }

    pub fn register(&mut self, shop: &str, agent: AgentFn) {
        self.agents.insert(shop.to_string(), agent);
    }

    fn tick(&mut self) -> u64 {
        self.now_ms += 1;
        self.now_ms
    }

    /// 发信入口: ACL → 自回声熔断 → 自动静默 → 记账 → 入箱 → (非静默则排队回合)。
    /// 返回 Err = 被守卫拒绝(教学性错误)。
    pub fn send(&mut self, from: &str, to: &str, body: &str, silent: bool) -> Result<(), String> {
        self.acl.check_send(from, to)?;
        if from == to && !silent {
            return Err("禁止给自己发非静默信(会唤醒自己造成死循环); 留痕请用 silent".into());
        }
        let silent = match &self.auto_silence {
            Some((prefix, arb)) if from.starts_with(prefix) && to == arb => true,
            _ => silent,
        };
        let mut l = Letter {
            seq: 0,
            ts_ms: self.tick(),
            from: from.into(),
            to: to.into(),
            body: body.into(),
        };
        self.journal.append(&mut l);
        self.mailboxes
            .entry(to.to_string())
            .or_default()
            .push(l.clone());
        if !silent {
            self.run_turn(to);
        }
        Ok(())
    }

    /// 广播: 越权目标自动收拢(不报错, taxonomy #2)。
    pub fn broadcast(
        &mut self,
        from: &str,
        targets: &[String],
        body: &str,
        silent: bool,
    ) -> Vec<String> {
        let effective = self.acl.collapse(from, targets);
        for t in effective.clone() {
            let _ = self.send(from, &t, body, silent);
        }
        effective
    }

    /// 破坏性读(回看走 journal.replay()).
    pub fn drain(&mut self, shop: &str) -> Vec<Letter> {
        self.mailboxes
            .get_mut(shop)
            .map(|m| std::mem::take(m))
            .unwrap_or_default()
    }

    /// 回合: 取唤醒信, 调店员函数, 发出其全部产出。
    fn run_turn(&mut self, shop: &str) {
        *self.turns.entry(shop.to_string()).or_insert(0) += 1;
        let mailbox = self.drain(shop);
        let Some(trigger) = mailbox.last().cloned() else {
            return;
        };
        let snapshot = self.journal.replay();
        let outs = match self.agents.get(shop) {
            Some(f) => f(&trigger, &snapshot),
            None => return,
        };
        for o in outs {
            let _ = self.send(shop, &o.to, &o.body, o.silent);
        }
    }

    /// 外部点火(编排器通道, 不受店员ACL约束面: 编排器是规则之源)。
    pub fn ignite(&mut self, to: &str, body: &str) -> Result<(), String> {
        let mut l = Letter {
            seq: 0,
            ts_ms: self.tick(),
            from: "orchestrator".into(),
            to: to.into(),
            body: body.into(),
        };
        self.journal.append(&mut l);
        self.mailboxes
            .entry(to.to_string())
            .or_default()
            .push(l.clone());
        self.run_turn(to);
        Ok(())
    }

    pub fn letters(&self) -> Vec<Letter> {
        self.journal.replay()
    }
    pub fn turn_count(&self, shop: &str) -> u64 {
        *self.turns.get(shop).unwrap_or(&0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn out(to: &str, body: &str, silent: bool) -> OutLetter {
        OutLetter {
            to: to.into(),
            body: body.into(),
            silent,
        }
    }
    fn clean(dir: &str) {
        let _ = std::fs::remove_dir_all(dir);
    }

    /// 回归 #1: 自回声熔断 — 非静默自发信被拒, 静默留痕放行
    #[test]
    fn echo_guard() {
        clean("/tmp/forge-test-echo");
        let mut hub = Hub::new(
            "/tmp/forge-test-echo",
            vec!["a".into(), "b".into()],
            Acl::default(),
        )
        .unwrap();
        assert!(hub.send("a", "a", "note", false).is_err());
        assert!(hub.send("a", "a", "note", true).is_ok());
    }

    /// 回归 #2: ACL 拒绝越权 + 广播收拢
    #[test]
    fn acl_reject_and_collapse() {
        clean("/tmp/forge-test-acl");
        let acl = Acl::from_pairs(&[("review_*", &["judge"]), ("judge", &["*"])]);
        let mut hub = Hub::new(
            "/tmp/forge-test-acl",
            vec![
                "judge".into(),
                "writer".into(),
                "review_1".into(),
                "review_2".into(),
            ],
            acl,
        )
        .unwrap();
        assert!(hub.send("review_1", "writer", "DRAFT", false).is_err());
        let sent = hub.broadcast(
            "review_1",
            &["judge".into(), "writer".into()],
            "REVIEW",
            false,
        );
        assert_eq!(sent, vec!["judge".to_string()]);
        let sent2 = hub.broadcast(
            "judge",
            &["review_1".into(), "review_2".into()],
            "群审",
            false,
        );
        assert_eq!(sent2.len(), 2);
    }

    /// 回归 #4: 审查回信自动静默
    #[test]
    fn auto_silence_reviewers() {
        clean("/tmp/forge-test-silence");
        let acl = Acl::from_pairs(&[("r*", &["judge"])]);
        let mut hub = Hub::new(
            "/tmp/forge-test-silence",
            vec!["judge".into(), "r1".into()],
            acl,
        )
        .unwrap();
        hub.auto_silence = Some(("r".into(), "judge".into()));
        hub.register("judge", Box::new(|_, _| vec![out("r1", "ACK", false)]));
        let _ = hub.send("r1", "judge", "CLEAN=r1", false);
        assert_eq!(hub.turn_count("judge"), 0);
    }

    /// 回归 #3(对冲): drain 破坏性读后, 账本 replay 仍完整
    #[test]
    fn ledger_is_registry() {
        clean("/tmp/forge-test-ledger");
        let mut hub = Hub::new(
            "/tmp/forge-test-ledger",
            vec!["a".into(), "b".into()],
            Acl::default(),
        )
        .unwrap();
        let _ = hub.send("a", "b", "hello", true);
        let drained = hub.drain("b");
        assert_eq!(drained.len(), 1);
        assert!(hub.drain("b").is_empty());
        assert_eq!(hub.letters().len(), 1);
    }
}
