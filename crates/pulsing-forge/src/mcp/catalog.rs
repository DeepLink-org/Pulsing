//! MCP catalog builder with Codex precedence: Plugin < Config < Compatibility < Extension.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::config::McpServerConfig;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum McpServerSource {
    Plugin { plugin_id: String },
    Config,
    Compatibility { id: String },
    Extension { id: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum RegistrationPrecedence {
    Plugin(Reverse<usize>),
    Config,
    Compatibility,
    Extension(usize),
}

impl RegistrationPrecedence {
    fn tier(self) -> u8 {
        match self {
            Self::Plugin(_) => 0,
            Self::Config => 1,
            Self::Compatibility => 2,
            Self::Extension(_) => 3,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct McpServerRegistration {
    name: String,
    source: McpServerSource,
    config: McpServerConfig,
    precedence: RegistrationPrecedence,
}

impl McpServerRegistration {
    pub fn from_config(name: String, config: McpServerConfig) -> Self {
        Self::new(
            name,
            McpServerSource::Config,
            config,
            RegistrationPrecedence::Config,
        )
    }

    pub fn from_plugin(
        name: String,
        plugin_id: String,
        plugin_order: usize,
        config: McpServerConfig,
    ) -> Self {
        Self::new(
            name,
            McpServerSource::Plugin { plugin_id },
            config,
            RegistrationPrecedence::Plugin(Reverse(plugin_order)),
        )
    }

    fn new(
        name: String,
        source: McpServerSource,
        config: McpServerConfig,
        precedence: RegistrationPrecedence,
    ) -> Self {
        Self {
            name,
            source,
            config,
            precedence,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum McpServerConflictAction {
    Register(McpServerSource),
    Remove(McpServerSource),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpServerConflict {
    pub name: String,
    pub outcome: McpServerConflictAction,
    pub contenders: Vec<McpServerConflictAction>,
}

#[derive(Clone, Debug)]
pub struct ResolvedMcpServer {
    pub source: McpServerSource,
    pub config: McpServerConfig,
}

#[derive(Clone, Debug, Default)]
pub struct ResolvedMcpCatalog {
    pub servers: HashMap<String, ResolvedMcpServer>,
    pub conflicts: Vec<McpServerConflict>,
}

#[derive(Clone, Debug)]
enum CatalogAction {
    Register(Box<McpServerRegistration>),
    Remove {
        name: String,
        source: McpServerSource,
        precedence: RegistrationPrecedence,
    },
}

impl CatalogAction {
    fn name(&self) -> &str {
        match self {
            Self::Register(r) => &r.name,
            Self::Remove { name, .. } => name,
        }
    }

    fn precedence(&self) -> RegistrationPrecedence {
        match self {
            Self::Register(r) => r.precedence,
            Self::Remove { precedence, .. } => *precedence,
        }
    }

    fn conflict_action(&self) -> McpServerConflictAction {
        match self {
            Self::Register(r) => McpServerConflictAction::Register(r.source.clone()),
            Self::Remove { source, .. } => McpServerConflictAction::Remove(source.clone()),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct McpCatalogBuilder {
    actions: Vec<CatalogAction>,
    disabled_server_names: BTreeSet<String>,
}

impl McpCatalogBuilder {
    pub fn register(&mut self, registration: McpServerRegistration) {
        self.actions
            .push(CatalogAction::Register(Box::new(registration)));
    }

    pub fn disable(&mut self, name: String) {
        self.disabled_server_names.insert(name);
    }

    pub fn build(mut self) -> ResolvedMcpCatalog {
        self.actions.sort_by_key(CatalogAction::precedence);

        let mut winners = BTreeMap::<String, CatalogAction>::new();
        let mut actions_by_name_and_tier = BTreeMap::<(String, u8), Vec<&CatalogAction>>::new();
        for action in &self.actions {
            winners.insert(action.name().to_string(), action.clone());
            actions_by_name_and_tier
                .entry((action.name().to_string(), action.precedence().tier()))
                .or_default()
                .push(action);
        }

        let mut conflicts = Vec::new();
        for ((name, _), actions) in actions_by_name_and_tier {
            if actions.len() < 2 {
                continue;
            }
            let Some(outcome) = winners.get(&name).map(CatalogAction::conflict_action) else {
                continue;
            };
            conflicts.push(McpServerConflict {
                name,
                outcome,
                contenders: actions
                    .into_iter()
                    .map(CatalogAction::conflict_action)
                    .collect(),
            });
        }

        let mut disabled = self.disabled_server_names;
        let servers = winners
            .into_iter()
            .filter_map(|(name, action)| match action {
                CatalogAction::Register(registration) => {
                    let mut registration = *registration;
                    if !registration.config.enabled || disabled.contains(&name) {
                        registration.config.enabled = false;
                        disabled.insert(name.clone());
                    }
                    Some((
                        name,
                        ResolvedMcpServer {
                            source: registration.source,
                            config: registration.config,
                        },
                    ))
                }
                CatalogAction::Remove { .. } => None,
            })
            .collect();

        ResolvedMcpCatalog { servers, conflicts }
    }
}

pub fn build_default_catalog(
    plugin_servers: Vec<(String, String, usize, McpServerConfig)>,
    config_servers: HashMap<String, McpServerConfig>,
) -> ResolvedMcpCatalog {
    let mut builder = McpCatalogBuilder::default();
    for (name, plugin_id, order, config) in plugin_servers {
        builder.register(McpServerRegistration::from_plugin(
            name, plugin_id, order, config,
        ));
    }
    for (name, config) in config_servers {
        builder.register(McpServerRegistration::from_config(name, config));
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::config::McpServerTransportConfig;

    fn stdio_cfg() -> McpServerConfig {
        McpServerConfig {
            transport: McpServerTransportConfig::Stdio {
                command: "echo".into(),
                args: vec![],
                env: None,
                env_vars: vec![],
                cwd: None,
            },
            environment_id: "local".into(),
            enabled: true,
            required: false,
            supports_parallel_tool_calls: false,
            startup_timeout_sec: None,
            tool_timeout_sec: None,
            default_tools_approval_mode: None,
            enabled_tools: None,
            disabled_tools: None,
            scopes: None,
            oauth: None,
            oauth_resource: None,
            tools: Default::default(),
        }
    }

    #[test]
    fn config_overrides_plugin() {
        let mut plugin = stdio_cfg();
        plugin.enabled = false;
        let mut user = stdio_cfg();
        user.enabled = true;
        let catalog = build_default_catalog(
            vec![("demo".into(), "p@m".into(), 0, plugin)],
            HashMap::from([("demo".into(), user)]),
        );
        assert!(catalog.servers["demo"].config.enabled);
        assert!(matches!(
            catalog.servers["demo"].source,
            McpServerSource::Config
        ));
    }
}
