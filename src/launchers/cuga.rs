//! Launcher for the CUGA generalist agent harness
//! (<https://cuga.dev>).
//!
//! Unlike dedicated binaries (Claude Code, Pi, opencode), CUGA ships as a
//! Python package on PyPI. There is no standalone executable: the `cuga`
//! command is provided by the package's `cuga.cli:app` entry point and is
//! normally run through `uvx cuga` (or `pipx run cuga`), which bootstraps an
//! isolated environment containing the package and its dependencies.
//!
//! This launcher therefore:
//!  1. Resolves `uvx` on PATH (the launcher's `command()` returns `"uvx"`);
//!  2. Prepends the literal `"cuga"` argument in `launch()` so the spawned
//!     process runs `uvx cuga <user-args>`;
//!  3. Generates a CUGA `settings.toml` in the launcher state directory
//!     that wires the bound model (via `AgentModelBinding`) into the harness,
//!     and points CUGA at it through the `AGENT_SETTING_CONFIG` environment
//!     variable;
//!  4. Optionally enables filesystem and shell tools for basic agent
//!     capabilities;
//!  5. Optionally wires MCP servers from bound `McpBinding`s into a
//!     `mcp_servers.yaml` config file;
//!  6. Exposes the bound model's API key through `OPENAI_API_KEY` (CUGA's
//!     primary provider is OpenAI-compatible, which is what every
//!     granite-cli provider can serve).
//!
//! CUGA is configured through TOML settings files. The launcher writes a
//! generated file under `GRANITE_CLI_HOME/launcher-state/<id>/cuga/` and
//! redirects CUGA to it with `AGENT_SETTING_CONFIG`. The file contains
//! `[agent.*.model]` sections describing the bound model, matching the
//! layout CUGA documents for its model configuration files.

// Standard
use std::collections::HashSet;
use std::path::PathBuf;

// Third Party
use alog::{MessageLevel, alog_channel, use_channel};
use anyhow::Context;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

// Local
use crate::capabilities::{
    AgentModelBinding, BindingType, McpBinding, ResolvedCapability, ToolName,
};
use crate::launchers::base::{EnvBinding, LaunchContext, Launcher, LauncherMetadata, run_command};
use crate::providers::ApiType;
use crate::registry::{ConfigConstructable, ConstructError};
use crate::utils::resolve_shell_command;
use crate::utils::ui::Ui;

use_channel!("CUGA");

/*-- public --*/

/// Configuration for the CUGA launcher instance.
#[derive(Debug, Clone, Serialize, Deserialize, Default, schemars::JsonSchema)]
pub(crate) struct CugaLauncherConfig {
    /// Override path to the `uvx` binary for non-PATH installs.
    #[serde(default)]
    pub(crate) command_path: Option<String>,

    /// Enable filesystem tools (read_file, write_file, edit_file).
    #[serde(default)]
    pub(crate) enable_filesystem_tools: bool,

    /// Enable shell tools (run_command).
    #[serde(default)]
    pub(crate) enable_shell_tool: bool,

    /// Extra keys merged (shallow, last-write-wins) into the generated
    /// provider entry.
    #[serde(default)]
    pub(crate) provider_overrides: Option<serde_json::Value>,
}

pub(crate) struct CugaLauncher {
    instance_id: String,
    config: CugaLauncherConfig,
    /// Bound `AgentModelBinding`, if any.
    bound_binding: Option<AgentModelBinding>,
    /// Bound `McpBinding`s, if any.
    bound_mcp_bindings: Vec<McpBinding>,
}

impl ConfigConstructable for CugaLauncher {
    type Config = CugaLauncherConfig;

    fn new(instance_id: &str, cfg: &serde_json::Value) -> Result<Self, ConstructError> {
        let config: CugaLauncherConfig =
            serde_json::from_value(cfg.clone()).map_err(ConstructError::settings)?;
        Ok(Self {
            instance_id: instance_id.to_string(),
            config,
            bound_binding: None,
            bound_mcp_bindings: vec![],
        })
    }
}

impl crate::registry::Named for CugaLauncher {
    fn instance_id(&self) -> &str {
        &self.instance_id
    }
}

#[async_trait]
impl Launcher for CugaLauncher {
    fn command(&self) -> &str {
        // CUGA has no dedicated binary; we run it through `uvx`.
        "uvx"
    }

    async fn bind_capability(
        &mut self,
        capability: &dyn ResolvedCapability,
    ) -> anyhow::Result<()> {
        let supported = Self::metadata().supported_capabilities;
        let capability_types = capability.binding_types();
        if !capability_types.is_subset(&supported) {
            anyhow::bail!(
                "capability supports {:?} which this launcher does not support",
                capability_types
                    .difference(&supported)
                    .collect::<Vec<_>>()
            );
        }

        if capability_types.contains(&BindingType::Mcp) {
            let request = crate::capabilities::BindingRequest::Mcp(
                crate::capabilities::McpBindingRequest {
                    supported_transports: Self::metadata().mcp_transports.clone(),
                },
            );
            let binding = capability.bind(request).await?;
            match binding {
                crate::capabilities::Binding::Mcp(binding) => {
                    self.bound_mcp_bindings.push(binding);
                }
                other => anyhow::bail!(
                    "expected an Mcp binding, got {:?}",
                    other.binding_type()
                ),
            }
            return Ok(());
        }

        // CUGA's provider is `@ai-sdk/openai-compatible`, the one every
        // granite-cli provider can serve, so that is what we ask for.
        let request = crate::capabilities::BindingRequest::AgentModel(
            crate::capabilities::AgentModelBindingRequest { api_type: ApiType::OpenAI },
        );

        let binding = capability.bind(request).await?;
        match binding {
            crate::capabilities::Binding::AgentModel(binding) => {
                self.bound_binding = Some(binding);
            }
            other => anyhow::bail!(
                "expected an AgentModel binding, got {:?}",
                other.binding_type()
            ),
        }
        Ok(())
    }

    fn validate_command(&self) -> anyhow::Result<PathBuf> {
        resolve_shell_command(&self.config.command_path, "uvx")
    }

    fn map_tool_name(&self, tool: &ToolName) -> Option<String> {
        match tool {
            ToolName::FileRead => Some("read_file".to_string()),
            ToolName::FileWrite => Some("write_file".to_string()),
            ToolName::FileEdit => Some("edit_file".to_string()),
            ToolName::Shell => Some("run_command".to_string()),
            // CUGA doesn't have direct web fetch/search or file search built-ins;
            // MCP servers can provide them. Return None so the caller knows
            // not to pass them as native args.
            ToolName::WebFetch
            | ToolName::WebSearch
            | ToolName::Search
            | ToolName::FileSearch => None,
            ToolName::Mcp { server, tool: None } => Some(format!("{server}_*")),
            ToolName::Mcp { server, tool: Some(t) } => Some(format!("{server}_{t}")),
            ToolName::Other(raw) => Some(raw.clone()),
        }
    }

    /// Returns the environment overlay that configures CUGA.
    async fn env_overlay(&self, ctx: &LaunchContext) -> anyhow::Result<Vec<EnvBinding>> {
        let mut overlay = vec![];

        if self.bound_binding.is_none()
            && self.bound_mcp_bindings.is_empty()
            && !self.config.enable_filesystem_tools
            && !self.config.enable_shell_tool
        {
            return Ok(overlay);
        }

        // `AGENT_SETTING_CONFIG` is CUGA's documented env var for pointing it
        // at a settings file.
        let settings_path = cuga_settings_path(ctx)?;
        overlay.push(EnvBinding {
            key: "AGENT_SETTING_CONFIG".to_string(),
            value: settings_path.to_string_lossy().to_string(),
        });

        // API key for the provider.
        let api_key = self
            .bound_binding
            .as_ref()
            .and_then(|b| b.api_key.as_ref())
            .map(|k| k.0.clone())
            .filter(|k| !k.is_empty())
            .unwrap_or_else(|| PLACEHOLDER_API_KEY.to_string());

        overlay.push(EnvBinding {
            key: "OPENAI_API_KEY".to_string(),
            value: api_key,
        });

        // MCP servers config path (if any MCP bindings).
        if !self.bound_mcp_bindings.is_empty() {
            let mcp_path = cuga_mcp_servers_path(ctx)?;
            overlay.push(EnvBinding {
                key: "MCP_SERVERS_FILE".to_string(),
                value: mcp_path.to_string_lossy().to_string(),
            });
        }

        Ok(overlay)
    }

    /// Writes the generated `settings.toml` and `mcp_servers.yaml` (if needed)
    /// and then execs `uvx cuga <args>`.
    async fn launch(
        &self,
        args: &[String],
        ctx: &LaunchContext,
        ui: &dyn Ui,
    ) -> anyhow::Result<std::process::ExitStatus> {
        // Build the settings file first, if needed.
        if self.bound_binding.is_some()
            || self.bound_mcp_bindings.is_empty() == false
            || self.config.enable_filesystem_tools
            || self.config.enable_shell_tool
        {
            let settings_path = cuga_settings_path(ctx)?;

            if ctx.dry_run {
                ui.info(&format!(
                    "Would write CUGA settings to {}",
                    settings_path.display()
                ));
            } else {
                if let Some(parent) = settings_path.parent() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("Failed to create {}", parent.display()))?;
                }
                let toml_str = generate_settings_toml(self, ctx)
                    .with_context(|| "Failed to generate CUGA settings")?;
                std::fs::write(&settings_path, toml_str)
                    .with_context(|| format!("Failed to write {}", settings_path.display()))?;
                ui.info(&format!(
                    "Wrote CUGA settings to {}",
                    settings_path.display()
                ));
            }

            // Write MCP servers config if we have MCP bindings.
            if !self.bound_mcp_bindings.is_empty() {
                let mcp_path = cuga_mcp_servers_path(ctx)?;
                if ctx.dry_run {
                    ui.info(&format!(
                        "Would write MCP servers config to {}",
                        mcp_path.display()
                    ));
                } else {
                    if let Some(parent) = mcp_path.parent() {
                        std::fs::create_dir_all(parent)
                            .with_context(|| format!("Failed to create {}", parent.display()))?;
                    }
                    let yaml_str = generate_mcp_servers_yaml(self);
                    std::fs::write(&mcp_path, yaml_str)
                        .with_context(|| format!("Failed to write {}", mcp_path.display()))?;
                    ui.info(&format!(
                        "Wrote MCP servers config to {}",
                        mcp_path.display()
                    ));
                }
            }
        }

        let binary = self.validate_command()?;
        let overlay = self.env_overlay(ctx).await?;
        alog_channel!(MessageLevel::Debug2, "Env Overlay: {:#?}", overlay);

        // `uvx cuga <args>`: prepend the literal "cuga" so uvx runs the
        // CUGA package's CLI entry point.
        let mut full_args = vec!["cuga".to_string()];
        full_args.extend_from_slice(args);

        run_command(binary, &overlay, &full_args, ctx, ui).await
    }
}

impl HasCugaLauncherMetadata for CugaLauncher {
    fn metadata() -> LauncherMetadata {
        LauncherMetadata {
            name: "CUGA".to_string(),
            description: "CUGA generalist agent harness".to_string(),
            default_command: "uvx".to_string(),
            supported_capabilities: HashSet::from([
                BindingType::AgentModel,
                BindingType::Mcp,
            ]),
            mcp_transports: crate::launchers::shared::mcp_cli::STDIO_HTTP_AND_SSE
                .iter()
                .copied()
                .collect(),
            tags: vec![
                "cuga".to_string(),
                "generalist".to_string(),
                "agent".to_string(),
                "mcp".to_string(),
            ],
        }
    }
}

/*-- private --*/

// HasCugaLauncherMetadata is the macro-generated trait; re-exported via mod.rs.
use crate::launchers::base::HasLauncherMetadata as HasCugaLauncherMetadata;

/// Stand-in credential for providers that need no auth. CUGA hides models
/// whose provider has no resolvable key, so the value must be non-empty.
const PLACEHOLDER_API_KEY: &str = "granite-cli";

/// The generated `settings.toml` filename, relative to the launcher state dir.
const SETTINGS_FILE: &str = "settings.toml";

/// The generated MCP servers YAML filename.
const MCP_SERVERS_FILE: &str = "mcp_servers.yaml";

/// The granite-cli-owned CUGA settings directory for this launcher instance.
fn cuga_settings_dir(ctx: &LaunchContext) -> anyhow::Result<PathBuf> {
    Ok(crate::config::Config::launcher_state_dir(&ctx.launcher_id)?.join("cuga"))
}

/// The full path to the generated `settings.toml`.
fn cuga_settings_path(ctx: &LaunchContext) -> anyhow::Result<PathBuf> {
    Ok(cuga_settings_dir(ctx)?.join(SETTINGS_FILE))
}

/// The full path to the generated `mcp_servers.yaml`.
fn cuga_mcp_servers_path(ctx: &LaunchContext) -> anyhow::Result<PathBuf> {
    Ok(cuga_settings_dir(ctx)?.join(MCP_SERVERS_FILE))
}

/// Generates the CUGA settings TOML content.
fn generate_settings_toml(launcher: &CugaLauncher, _ctx: &LaunchContext) -> anyhow::Result<String> {
    use toml::Value;

    let mut settings = Value::Table(toml::map::Map::new());

    // Agent model configuration sections.
    // CUGA uses [agent.code.model], [agent.planner.model], etc.
    let agent_sections = ["code", "planner", "chat", "shortcode"];

    if let Some(binding) = &launcher.bound_binding {
        for section in &agent_sections {
            let agent_key = format!("agent.{section}.model");
            let mut model_table = toml::map::Map::new();
            model_table.insert(
                "platform".to_string(),
                Value::String("openai".to_string()),
            );
            model_table.insert(
                "model_name".to_string(),
                Value::String(binding.model_name.clone()),
            );
            model_table.insert(
                "temperature".to_string(),
                Value::Float(0.1),
            );
            // max_tokens from the binding, or a sensible default.
            let max_tokens = binding
                .context_length
                .unwrap_or(131_072)
                .min(100_000);
            model_table.insert(
                "max_tokens".to_string(),
                Value::Integer(max_tokens as i64),
            );

            // Apply provider overrides.
            if let Some(overrides) = &launcher.config.provider_overrides {
                if let Some(obj) = overrides.as_object() {
                    for (key, value) in obj {
                        let toml_val = match value {
                            serde_json::Value::String(s) => toml::Value::String(s.clone()),
                            serde_json::Value::Number(n) => {
                                if n.is_i64() {
                                    toml::Value::Integer(n.as_i64().unwrap_or(0))
                                } else if n.is_u64() {
                                    toml::Value::Integer(n.as_u64().unwrap_or(0) as i64)
                                } else {
                                    toml::Value::Float(n.as_f64().unwrap_or(0.0))
                                }
                            }
                            serde_json::Value::Bool(b) => toml::Value::Boolean(*b),
                            _ => continue,
                        };
                        model_table.insert(key.clone(), toml_val);
                    }
                }
            }

            settings
                .as_table_mut()
                .unwrap()
                .insert(agent_key, Value::Table(model_table));
        }
    }

    // Feature toggles.
    let mut features = toml::map::Map::new();
    features.insert(
        "enable_filesystem_tools".to_string(),
        Value::Boolean(launcher.config.enable_filesystem_tools),
    );
    features.insert(
        "enable_shell_tool".to_string(),
        Value::Boolean(launcher.config.enable_shell_tool),
    );
    // Execution mode - default to 'api' (tools only).
    features.insert(
        "mode".to_string(),
        Value::String("api".to_string()),
    );

    settings
        .as_table_mut()
        .unwrap()
        .insert("features".to_string(), Value::Table(features));

    // Convert to TOML string.
    let toml_str = toml::to_string_pretty(&settings)?;
    Ok(toml_str)
}

/// Generates the MCP servers YAML content.
fn generate_mcp_servers_yaml(launcher: &CugaLauncher) -> String {
    use serde_yaml::Value;

    let mut servers = serde_yaml::Mapping::new();

    for (index, binding) in launcher.bound_mcp_bindings.iter().enumerate() {
        let server_name = format!("cuga_mcp_{}", index);
        let mut server_config = serde_yaml::Mapping::new();

        match binding {
            McpBinding::Stdio { command, args, env, .. } => {
                server_config.insert(
                    Value::String("command".to_string()),
                    Value::String(command.clone()),
                );
                let args_vec: Vec<Value> = args
                    .iter()
                    .map(|a| Value::String(a.clone()))
                    .collect();
                server_config.insert(Value::String("args".to_string()), Value::Sequence(args_vec));
                if !env.is_empty() {
                    let mut env_map = serde_yaml::Mapping::new();
                    for (k, v) in env {
                        env_map.insert(Value::String(k.clone()), Value::String(v.clone()));
                    }
                    server_config.insert(
                        Value::String("env".to_string()),
                        Value::Mapping(env_map),
                    );
                }
            }
            McpBinding::Http { url, headers, .. }
            | McpBinding::Sse { url, headers, .. } => {
                server_config.insert(Value::String("url".to_string()), Value::String(url.clone()));
                if !headers.is_empty() {
                    let mut headers_map = serde_yaml::Mapping::new();
                    for (k, v) in headers {
                        headers_map.insert(Value::String(k.clone()), Value::String(v.clone()));
                    }
                    server_config.insert(
                        Value::String("headers".to_string()),
                        Value::Mapping(headers_map),
                    );
                }
            }
        }

        servers.insert(Value::String(server_name), Value::Mapping(server_config));
    }

    serde_yaml::to_string(&Value::Mapping(servers)).unwrap_or_default()
}

/// Maps a canonical `ToolName` onto CUGA's own tool-name strings.
/*-- tests --*/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Named, Secret};
    use crate::utils::ui::base::tests::CaptureUi;

    fn launcher(cfg: serde_json::Value) -> CugaLauncher {
        CugaLauncher::new("cuga", &cfg).unwrap()
    }

    fn binding() -> AgentModelBinding {
        AgentModelBinding {
            api_type: ApiType::OpenAI,
            provider_name: "granite".to_string(),
            base_url: "http://localhost:11434".to_string(),
            model_name: "granite4.1:8b".to_string(),
            endpoint_path: "/v1/chat/completions".to_string(),
            api_key: None,
            verify_ssl: true,
            context_length: Some(131_072),
            custom_headers: None,
            supported_reasoning_efforts: vec![],
        }
    }

    fn ctx(dry_run: bool) -> LaunchContext {
        LaunchContext {
            launcher_id: "cuga".to_string(),
            working_dir: PathBuf::from("/tmp"),
            base_env: std::collections::HashMap::new(),
            dry_run,
            usage_tracker: None,
            model_proxy: None,
        }
    }

    // -- command resolution ----------------------------------------------------

    #[test]
    fn command_defaults_to_uvx() {
        assert_eq!(launcher(serde_json::json!({})).command(), "uvx");
    }

    // -- metadata / schema -----------------------------------------------------

    #[test]
    fn metadata_name_is_cuga() {
        let meta = CugaLauncher::metadata();
        assert_eq!(meta.name, "CUGA");
        assert_eq!(meta.default_command, "uvx");
        assert!(meta.supported_capabilities.contains(&BindingType::AgentModel));
        assert!(meta.supported_capabilities.contains(&BindingType::Mcp));
    }

    #[test]
    fn instance_id_round_trips_from_construction() {
        let l = CugaLauncher::new("cuga-local", &serde_json::json!({})).unwrap();
        assert_eq!(l.instance_id(), "cuga-local");
    }

    #[test]
    fn config_schema_exposes_command_path_overrides() {
        use crate::launchers::base::LauncherFactory;
        let mut factory = LauncherFactory::new();
        factory.register::<CugaLauncher>("cuga");
        let schema = factory.config_schema("cuga").unwrap();
        let props = schema
            .get("properties")
            .and_then(|p| p.as_object())
            .unwrap();
        assert!(props.contains_key("command_path"));
        assert!(props.contains_key("provider_overrides"));
    }

    // -- env overlay -----------------------------------------------------------

    #[tokio::test]
    async fn env_overlay_redirects_settings_and_exports_api_key() {
        let mut l = launcher(serde_json::json!({}));
        l.bound_binding = Some(binding());
        let overlay = l.env_overlay(&ctx(false)).await.unwrap();

        let settings = overlay
            .iter()
            .find(|b| b.key == "AGENT_SETTING_CONFIG")
            .expect("settings config redirect");
        assert!(
            settings.value.ends_with("cuga/settings.toml"),
            "{}",
            settings.value
        );

        let key = overlay
            .iter()
            .find(|b| b.key == "OPENAI_API_KEY")
            .expect("api key");
        assert_eq!(key.value, "granite-cli");
    }

    #[tokio::test]
    async fn env_overlay_uses_placeholder_when_provider_has_no_key() {
        let mut l = launcher(serde_json::json!({}));
        l.bound_binding = Some(AgentModelBinding {
            api_key: Some(Secret::from("")),
            ..binding()
        });
        let overlay = l.env_overlay(&ctx(false)).await.unwrap();
        let key = overlay
            .iter()
            .find(|b| b.key == "OPENAI_API_KEY")
            .unwrap();
        assert_eq!(key.value, "granite-cli");
    }

    #[tokio::test]
    async fn env_overlay_uses_real_key_when_present() {
        let mut l = launcher(serde_json::json!({}));
        l.bound_binding = Some(AgentModelBinding {
            api_key: Some(Secret::from("sk-test-123")),
            ..binding()
        });
        let overlay = l.env_overlay(&ctx(false)).await.unwrap();
        let key = overlay
            .iter()
            .find(|b| b.key == "OPENAI_API_KEY")
            .unwrap();
        assert_eq!(key.value, "sk-test-123");
    }

    // -- tool name mapping -----------------------------------------------------

    #[test]
    fn map_tool_name_maps_filesystem_tools() {
        let l = launcher(serde_json::json!({}));
        assert_eq!(l.map_tool_name(&ToolName::FileRead), Some("read_file".to_string()));
        assert_eq!(l.map_tool_name(&ToolName::FileWrite), Some("write_file".to_string()));
        assert_eq!(l.map_tool_name(&ToolName::FileEdit), Some("edit_file".to_string()));
        assert_eq!(l.map_tool_name(&ToolName::Shell), Some("run_command".to_string()));
    }

    #[test]
    fn map_tool_name_maps_mcp_tools() {
        let l = launcher(serde_json::json!({}));
        assert_eq!(
            l.map_tool_name(&ToolName::Mcp {
                server: "github".to_string(),
                tool: None
            }),
            Some("github_*".to_string())
        );
        assert_eq!(
            l.map_tool_name(&ToolName::Mcp {
                server: "filesystem".to_string(),
                tool: Some("read".to_string())
            }),
            Some("filesystem_read".to_string())
        );
    }

    #[test]
    fn map_tool_name_returns_none_for_web_tools() {
        let l = launcher(serde_json::json!({}));
        assert_eq!(l.map_tool_name(&ToolName::WebFetch), None);
        assert_eq!(l.map_tool_name(&ToolName::WebSearch), None);
        assert_eq!(l.map_tool_name(&ToolName::Search), None);
        assert_eq!(l.map_tool_name(&ToolName::FileSearch), None);
    }
}
