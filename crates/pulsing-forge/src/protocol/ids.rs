use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

macro_rules! opaque_id {
    ($name:ident, $prefix:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new() -> Self {
                Self(format!("{}_{}", $prefix, Uuid::new_v4().simple()))
            }

            pub fn from_string(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

opaque_id!(SessionId, "ses");
opaque_id!(TurnId, "turn");
opaque_id!(EventId, "evt");
opaque_id!(CommandId, "cmd");
opaque_id!(CandidateId, "cand");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_opaque_and_round_trip() {
        let id = SessionId::new();
        let encoded = serde_json::to_string(&id).unwrap();
        let decoded: SessionId = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, id);
        assert!(id.as_str().starts_with("ses_"));
    }
}
