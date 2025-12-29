//! Actor factory trait - supports extensions

use super::SystemResponse;
use async_trait::async_trait;

/// Boxed actor factory type
pub type BoxedActorFactory = Box<dyn ActorFactory>;

/// Actor factory trait
///
/// Used to extend SystemActor's actor creation capability.
/// Python bindings can implement their own factory to support Python actor creation.
#[async_trait]
pub trait ActorFactory: Send + Sync {
    /// Check if a specific actor type is supported
    fn supports(&self, actor_type: &str) -> bool;

    /// List supported actor types
    fn supported_types(&self) -> Vec<String>;

    /// Handle extension messages
    ///
    /// Used to support custom extension features like Python actor creation.
    async fn handle_extension(&self, handler: &str, _payload: serde_json::Value) -> SystemResponse {
        SystemResponse::Error {
            message: format!("Unknown extension handler: {}", handler),
        }
    }
}

/// Default actor factory (supports queries only, no creation)
pub struct DefaultActorFactory;

#[async_trait]
impl ActorFactory for DefaultActorFactory {
    fn supports(&self, _actor_type: &str) -> bool {
        false
    }

    fn supported_types(&self) -> Vec<String> {
        vec![]
    }

    async fn handle_extension(&self, handler: &str, _payload: serde_json::Value) -> SystemResponse {
        SystemResponse::Error {
            message: format!(
                "Extension '{}' not supported. Use Python extension for actor creation.",
                handler
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_factory() {
        let factory = DefaultActorFactory;
        assert!(!factory.supports("TestActor"));
        assert!(factory.supported_types().is_empty());
    }

    #[tokio::test]
    async fn test_default_factory_extension() {
        let factory = DefaultActorFactory;
        let response = factory
            .handle_extension("test", serde_json::Value::Null)
            .await;

        match response {
            SystemResponse::Error { message } => {
                assert!(message.contains("test"));
            }
            _ => panic!("Expected Error response"),
        }
    }
}
