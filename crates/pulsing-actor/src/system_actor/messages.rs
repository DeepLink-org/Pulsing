//! SystemActor message type definitions

use serde::{Deserialize, Serialize};

/// SystemActor request messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SystemMessage {
    // ============= Actor Management =============
    /// Create actor (used by Python extension)
    CreateActor {
        /// Actor type name
        actor_type: String,
        /// Actor instance name
        name: String,
        /// Constructor parameters (JSON)
        #[serde(default)]
        params: serde_json::Value,
        /// Whether to make public
        #[serde(default = "default_true")]
        public: bool,
    },

    /// Stop actor
    StopActor {
        /// Actor name
        name: String,
    },

    // ============= Queries =============
    /// List all actors
    ListActors,

    /// Get specific actor info
    GetActor {
        /// Actor name
        name: String,
    },

    /// Get system metrics
    GetMetrics,

    /// Get node info
    GetNodeInfo,

    /// Health check
    HealthCheck,

    /// Ping (for connectivity test)
    Ping,

    // ============= Extensions =============
    /// Extension message (for Python and other extensions)
    Extension {
        /// Handler name
        handler: String,
        /// Payload data
        payload: serde_json::Value,
    },
}

fn default_true() -> bool {
    true
}

/// SystemActor response messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SystemResponse {
    /// Operation successful (no data)
    Ok,

    /// Operation failed
    Error {
        /// Error message
        message: String,
    },

    /// Actor created successfully
    ActorCreated {
        /// Actor ID
        actor_id: u64,
        /// Actor name
        name: String,
        /// Node ID
        node_id: u64,
        /// Available methods list (for Python actors)
        #[serde(default)]
        methods: Vec<String>,
    },

    /// Actor list
    ActorList {
        /// List of actor info
        actors: Vec<ActorInfo>,
    },

    /// Single actor info
    ActorInfo(ActorInfo),

    /// System metrics
    Metrics {
        /// Actor count
        actors_count: usize,
        /// Total messages processed
        messages_total: u64,
        /// Total actors created
        actors_created: u64,
        /// Total actors stopped
        actors_stopped: u64,
        /// Uptime in seconds
        uptime_secs: u64,
    },

    /// Node info
    NodeInfo {
        /// Node ID
        node_id: u64,
        /// Address
        addr: String,
        /// Uptime in seconds
        uptime_secs: u64,
    },

    /// Health status
    Health {
        /// Status
        status: String,
        /// Actor count
        actors_count: usize,
        /// Uptime in seconds
        uptime_secs: u64,
    },

    /// Pong response
    Pong {
        /// Node ID
        node_id: u64,
        /// Timestamp
        timestamp: u64,
    },
}

/// Actor info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorInfo {
    /// Actor name
    pub name: String,
    /// Actor ID (local ID)
    pub actor_id: u64,
    /// Actor type
    pub actor_type: String,
    /// Uptime in seconds
    pub uptime_secs: u64,
    /// Whether public
    pub public: bool,
    /// Actor metadata (e.g., Python class info)
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

/// Actor status info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorStatusInfo {
    /// Status
    pub status: String,
    /// Message count
    pub message_count: u64,
    /// Last active time
    pub last_active: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_message_serialization() {
        let msg = SystemMessage::ListActors;
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("ListActors"));

        let msg = SystemMessage::GetActor {
            name: "test".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("GetActor"));
        assert!(json.contains("test"));
    }

    #[test]
    fn test_system_response_serialization() {
        let resp = SystemResponse::Ok;
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("Ok"));

        let resp = SystemResponse::Error {
            message: "test error".to_string(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("Error"));
        assert!(json.contains("test error"));
    }

    #[test]
    fn test_actor_info_serialization() {
        let info = ActorInfo {
            name: "test".to_string(),
            actor_id: 123,
            actor_type: "TestActor".to_string(),
            uptime_secs: 60,
            public: true,
            metadata: std::collections::HashMap::new(),
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("123"));
    }

    #[test]
    fn test_create_actor_deserialization() {
        // Test JSON deserialization (Python compatibility)
        let json = r#"{"type":"CreateActor","actor_type":"Counter","name":"c1","params":{"init_value":10},"public":true}"#;
        let msg: SystemMessage = serde_json::from_str(json).unwrap();

        match msg {
            SystemMessage::CreateActor {
                actor_type,
                name,
                params,
                public,
            } => {
                assert_eq!(actor_type, "Counter");
                assert_eq!(name, "c1");
                assert_eq!(params["init_value"], 10);
                assert!(public);
            }
            _ => panic!("Expected CreateActor"),
        }
    }
}
