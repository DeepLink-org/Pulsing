//! 控制口鉴权 — per-shop 令牌（taxonomy 假 ACL 事故的最终封堵）。
//! 原则: 编排器与店员各持自己的令牌；越权冒名在协议层即被拒。

/// 令牌环: 每个 shop 一个独立令牌 + 一个编排器主令牌。
#[derive(Debug, Clone)]
pub struct TokenRing {
    orchestrator: String,
    shops: std::collections::HashMap<String, String>,
}

impl TokenRing {
    /// 从密钥派生（不落盘、不进 git——调用方负责 secret 管理）。
    pub fn derive(master_secret: &str, shops: &[String]) -> Self {
        let mut map = std::collections::HashMap::new();
        for s in shops {
            map.insert(
                s.clone(),
                TokenRing::hmac(master_secret, &format!("shop:{s}")),
            );
        }
        Self {
            orchestrator: TokenRing::hmac(master_secret, "orchestrator"),
            shops: map,
        }
    }

    /// HMAC-SHA256 简化版（FNV 双轮混合; 生产版换 ring crate 的真 HMAC）。
    fn hmac(secret: &str, msg: &str) -> String {
        let mut h = Self::fnv(msg);
        h ^= Self::fnv(secret);
        format!("{h:016x}")
    }
    fn fnv(s: &str) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in s.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    pub fn orchestrator_token(&self) -> &str {
        &self.orchestrator
    }
    pub fn shop_token(&self, shop: &str) -> Option<&str> {
        self.shops.get(shop).map(|x| x.as_str())
    }

    /// 验证: 令牌能解锁哪个身份。None = 无效令牌。
    pub fn authenticate(&self, token: &str) -> Option<Identity> {
        if token == self.orchestrator {
            return Some(Identity::Orchestrator);
        }
        self.shops
            .iter()
            .find(|(_, t)| t.as_str() == token)
            .map(|(s, _)| Identity::Shop(s.clone()))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Identity {
    Orchestrator,
    Shop(String),
}

/// 控制口动作的鉴权矩阵:
/// - Orchestrator: 一切动作（含冒名发信——它是规则之源）
/// - Shop(own): 只能以自己身份发信/读自己的信箱
/// - Shop(other): 拒绝（冒名即重放假 ACL 事故）
pub fn authorize(actor: &Identity, action: Action, target_shop: &str) -> Result<(), String> {
    match (actor, action) {
        (Identity::Orchestrator, _) => Ok(()),
        (Identity::Shop(me), Action::Send { as_shop })
        | (Identity::Shop(me), Action::Drain { shop: as_shop })
            if me == as_shop && me == target_shop =>
        {
            Ok(())
        }
        (Identity::Shop(me), _) => Err(format!(
            "店 {me:?} 试图以 {target_shop:?} 身份行动 — 冒名被拒"
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
    Send { as_shop: &'static str },
    Drain { shop: &'static str },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 回归假ACL: 店不能用别人令牌冒名发信
    #[test]
    fn shop_cannot_impersonate() {
        let ring = TokenRing::derive("secret-key", &["writer".into(), "judge".into()]);
        let judge_tok = ring.shop_token("judge").unwrap();
        let id = ring.authenticate(judge_tok).unwrap();
        assert_eq!(id, Identity::Shop("judge".into()));
        let r = authorize(&id, Action::Send { as_shop: "writer" }, "writer");
        assert!(r.is_err());
    }

    /// 店可以用自己令牌做自己事
    #[test]
    fn shop_can_act_as_self() {
        let ring = TokenRing::derive("secret-key", &["writer".into()]);
        let tok = ring.shop_token("writer").unwrap();
        let id = ring.authenticate(tok).unwrap();
        assert!(authorize(&id, Action::Send { as_shop: "writer" }, "writer").is_ok());
        assert!(authorize(&id, Action::Drain { shop: "writer" }, "writer").is_ok());
    }

    /// 编排器全通; 令牌不可跨密钥伪造
    #[test]
    fn orchestrator_and_cross_secret() {
        let ring1 = TokenRing::derive("secret-a", &["w".into()]);
        let ring2 = TokenRing::derive("secret-b", &["w".into()]);
        assert!(ring1.authenticate(ring1.orchestrator_token()).is_some());
        // 不同密钥派生出不同令牌
        assert_ne!(ring1.shop_token("w"), ring2.shop_token("w"));
        let id = ring1.authenticate(ring1.orchestrator_token()).unwrap();
        assert!(authorize(&id, Action::Send { as_shop: "w" }, "w").is_ok());
    }
}
