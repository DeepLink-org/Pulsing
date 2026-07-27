//! Forge's canonical tool registry.
//!
//! A registered executor is the single source of truth for both model-visible
//! metadata and runtime dispatch. Agent clients select tools by name, but never
//! maintain a second copy of their schemas.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

use crate::executor::ToolExecutor;

/// A callable tool name that preserves an optional namespace.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct ToolName {
    pub namespace: Option<String>,
    pub name: String,
}

impl ToolName {
    pub fn plain(name: impl Into<String>) -> Self {
        Self {
            namespace: None,
            name: name.into(),
        }
    }

    pub fn namespaced(namespace: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            namespace: Some(namespace.into()),
            name: name.into(),
        }
    }

    /// Name sent over the current flat model tool protocol.
    pub fn model_name(&self) -> String {
        match &self.namespace {
            Some(namespace) => format!("{namespace}{}", self.name),
            None => self.name.clone(),
        }
    }
}

impl fmt::Display for ToolName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.model_name())
    }
}

impl From<&str> for ToolName {
    fn from(value: &str) -> Self {
        Self::plain(value)
    }
}

impl From<String> for ToolName {
    fn from(value: String) -> Self {
        Self::plain(value)
    }
}

/// Provider-neutral metadata owned by the executable tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: ToolName,
    pub description: String,
    pub input_schema: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
}

impl ToolSpec {
    pub fn function(
        name: impl Into<ToolName>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema: normalize_object_schema(input_schema),
            output_schema: None,
        }
    }

    /// Current Forge LLM clients consume Anthropic-style function definitions;
    /// OpenAI conversion happens in `llm::message`.
    pub fn model_definition(&self) -> Value {
        json!({
            "name": self.name.model_name(),
            "description": self.description,
            "input_schema": self.input_schema,
        })
    }
}

fn normalize_object_schema(mut schema: Value) -> Value {
    if let Some(obj) = schema.as_object_mut() {
        obj.entry("type").or_insert_with(|| json!("object"));
        obj.entry("properties")
            .or_insert_with(|| Value::Object(Default::default()));
    }
    schema
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ToolRegistryError {
    #[error("tool name must not be empty")]
    EmptyName,
    #[error("tool executor name {executor:?} does not match spec name {spec:?}")]
    NameMismatch { executor: String, spec: String },
    #[error("duplicate tool registration: {0}")]
    Duplicate(String),
    #[error("unknown tool selected for this Forge runtime: {0}")]
    Unknown(String),
    #[error("tool is not directly model-visible: {0}")]
    NotDirect(String),
}

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn ToolExecutor>>,
}

impl ToolRegistry {
    pub fn from_executors(
        executors: impl IntoIterator<Item = Box<dyn ToolExecutor>>,
    ) -> Result<Self, ToolRegistryError> {
        let mut registry = Self {
            tools: HashMap::new(),
        };
        for executor in executors {
            registry.register(executor)?;
        }
        Ok(registry)
    }

    pub fn register(&mut self, executor: Box<dyn ToolExecutor>) -> Result<(), ToolRegistryError> {
        let executor_name = executor.tool_name().trim().to_string();
        if executor_name.is_empty() {
            return Err(ToolRegistryError::EmptyName);
        }
        let spec_name = executor.spec().name.model_name();
        if executor_name != spec_name {
            return Err(ToolRegistryError::NameMismatch {
                executor: executor_name,
                spec: spec_name,
            });
        }
        if self.tools.contains_key(&executor_name) {
            return Err(ToolRegistryError::Duplicate(executor_name));
        }
        self.tools.insert(executor_name, executor);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&dyn ToolExecutor> {
        self.tools.get(name).map(Box::as_ref)
    }

    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.tools.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn definitions_for(&self, names: &[String]) -> Result<Vec<Value>, ToolRegistryError> {
        names
            .iter()
            .map(|name| {
                let tool = self
                    .tools
                    .get(name)
                    .ok_or_else(|| ToolRegistryError::Unknown(name.clone()))?;
                if !tool.exposure().is_direct() {
                    return Err(ToolRegistryError::NotDirect(name.clone()));
                }
                Ok(tool.spec().model_definition())
            })
            .collect()
    }

    pub fn direct_definitions(&self) -> Vec<Value> {
        let mut tools: Vec<_> = self
            .tools
            .values()
            .filter(|tool| tool.exposure().is_direct())
            .map(|tool| tool.spec().model_definition())
            .collect();
        tools.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
        tools
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::ToolCallContext;
    use crate::error::ToolError;
    use crate::executor::{ToolExecutorFuture, ToolExposure};
    use crate::result::ToolResult;

    struct TestTool {
        runtime_name: &'static str,
        spec_name: &'static str,
        exposure: ToolExposure,
    }

    impl ToolExecutor for TestTool {
        fn tool_name(&self) -> &str {
            self.runtime_name
        }

        fn spec(&self) -> ToolSpec {
            ToolSpec::function(self.spec_name, "test", json!({"type": "object"}))
        }

        fn exposure(&self) -> ToolExposure {
            self.exposure
        }

        fn handle<'a>(
            &'a self,
            _ctx: &'a ToolCallContext,
            _arguments: Value,
        ) -> ToolExecutorFuture<'a> {
            Box::pin(async { Ok::<_, ToolError>(ToolResult::ok("ok")) })
        }
    }

    #[test]
    fn rejects_name_mismatch_and_duplicates() {
        let mismatch = ToolRegistry::from_executors([Box::new(TestTool {
            runtime_name: "runtime",
            spec_name: "schema",
            exposure: ToolExposure::Direct,
        }) as Box<dyn ToolExecutor>])
        .err()
        .unwrap();
        assert!(matches!(mismatch, ToolRegistryError::NameMismatch { .. }));

        let duplicate = ToolRegistry::from_executors([
            Box::new(TestTool {
                runtime_name: "same",
                spec_name: "same",
                exposure: ToolExposure::Direct,
            }) as Box<dyn ToolExecutor>,
            Box::new(TestTool {
                runtime_name: "same",
                spec_name: "same",
                exposure: ToolExposure::Direct,
            }) as Box<dyn ToolExecutor>,
        ])
        .err()
        .unwrap();
        assert_eq!(duplicate, ToolRegistryError::Duplicate("same".into()));
    }

    #[test]
    fn definitions_are_derived_from_registered_executors() {
        let registry = ToolRegistry::from_executors([
            Box::new(TestTool {
                runtime_name: "visible",
                spec_name: "visible",
                exposure: ToolExposure::Direct,
            }) as Box<dyn ToolExecutor>,
            Box::new(TestTool {
                runtime_name: "deferred",
                spec_name: "deferred",
                exposure: ToolExposure::Deferred,
            }) as Box<dyn ToolExecutor>,
        ])
        .unwrap();

        assert_eq!(registry.direct_definitions().len(), 1);
        assert_eq!(registry.direct_definitions()[0]["name"], "visible");
        assert_eq!(
            registry.definitions_for(&["deferred".into()]).unwrap_err(),
            ToolRegistryError::NotDirect("deferred".into())
        );
    }

    #[test]
    fn every_builtin_executor_has_one_canonical_definition() {
        let registry = ToolRegistry::from_executors(crate::handlers::builtin_handlers()).unwrap();
        let names = registry.names();
        let definitions = registry.direct_definitions();

        assert_eq!(names.len(), 22);
        assert_eq!(definitions.len(), names.len());
        for definition in definitions {
            let name = definition["name"].as_str().unwrap();
            assert!(names.iter().any(|candidate| candidate == name));
            assert!(
                definition["description"]
                    .as_str()
                    .is_some_and(|value| !value.is_empty())
            );
            assert_eq!(definition["input_schema"]["type"], "object");
            assert!(definition["input_schema"]["properties"].is_object());
        }
    }
}
