//! `DecisionMCPCapability`: backs the `jev-mcp` stdio MCP server
//! (`github:codaaiteam/jev-mcp`) with a model that supports
//! [`ModelFunction::Decision`], so a launched coding agent can call it as a
//! decision tool.
//!
//! Unlike `vision-mcp`, no server runs in-process: `bind()` hands the
//! launcher a stdio command (`npx`) and the connection details of the
//! resolved model's Jev endpoint as environment variables, and the
//! downstream tool spawns the server itself.

use crate::capabilities::base::{
    Binding, BindingRequest, BindingType, Capability, CapabilityMetadata, Dependency,
    HasCapabilityMetadata, McpBinding, McpBindingRequest, McpTransportKind,
};
use crate::capabilities::requirement::ModelRequirement;
use crate::models::{ConfiguredModel, ModelFunction};
use crate::providers::ApiType;
use crate::registry::{ConfigConstructable, ConstructError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_valid::Validate;
use std::collections::{HashMap, HashSet};

/// The npm package spec `npx` fetches to run the server.
const JEV_MCP_PACKAGE: &str = "github:codaaiteam/jev-mcp";

/*-- public -------------------------------------------------------------------*/

#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema, Validate)]
pub struct DecisionMCPCapabilityConfig {
    /// Key into the configured models map (the user-chosen instance ID) for
    /// the decision model this capability serves.
    #[validate(min_length = 1, message = "no model is selected")]
    pub model_id: String,
    /// Pins `jev-mcp` to a release, tag, branch or commit (the `#<ref>` of
    /// the npm git spec). Unpinned when absent.
    #[serde(default)]
    #[validate(min_length = 1, message = "version must not be empty")]
    pub version: Option<String>,
}

pub struct DecisionMCPCapability {
    instance_id: String,
    config: DecisionMCPCapabilityConfig,
}

/// [`DecisionMCPCapability`] with the model its `model_id` names. Built only by
/// `resolve_refs`, so holding one is what says the name resolved and the
/// model meets what this capability's metadata requires of it.
pub struct ResolvedDecisionMCPCapability {
    inner: DecisionMCPCapability,
    configured_model: ConfiguredModel,
}

impl ConfigConstructable for DecisionMCPCapability {
    type Config = DecisionMCPCapabilityConfig;

    fn new(instance_id: &str, cfg: &serde_json::Value) -> Result<Self, ConstructError> {
        let config: DecisionMCPCapabilityConfig =
            serde_json::from_value(cfg.clone()).map_err(ConstructError::settings)?;
        config
            .validate()
            .map_err(|e| crate::capabilities::base::invalid_settings(&e))?;
        Ok(Self {
            instance_id: instance_id.to_string(),
            config,
        })
    }
}

impl crate::registry::Named for DecisionMCPCapability {
    fn instance_id(&self) -> &str {
        &self.instance_id
    }
}

impl crate::registry::Named for ResolvedDecisionMCPCapability {
    fn instance_id(&self) -> &str {
        self.inner.instance_id()
    }
}

impl crate::capabilities::CapabilityInfo for DecisionMCPCapability {
    fn name(&self) -> &str {
        "Decision MCP Server"
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn binding_types(&self) -> HashSet<BindingType> {
        HashSet::from([BindingType::Mcp])
    }
}

impl crate::capabilities::CapabilityInfo for ResolvedDecisionMCPCapability {
    fn name(&self) -> &str {
        crate::capabilities::CapabilityInfo::name(&self.inner)
    }

    fn description(&self) -> &str {
        crate::capabilities::CapabilityInfo::description(&self.inner)
    }

    fn binding_types(&self) -> HashSet<BindingType> {
        crate::capabilities::CapabilityInfo::binding_types(&self.inner)
    }
}

impl Capability for DecisionMCPCapability {
    fn resolve_refs(
        self: Box<Self>,
        models: &dyn crate::models::ModelLookup,
    ) -> anyhow::Result<Box<dyn crate::capabilities::ResolvedCapability>> {
        let configured_model = crate::capabilities::base::resolve_declared_model(
            models,
            &Self::metadata(),
            &self.config.model_id,
        )?;
        Ok(Box::new(ResolvedDecisionMCPCapability {
            inner: *self,
            configured_model,
        }))
    }
}

#[async_trait]
impl crate::capabilities::ResolvedCapability for ResolvedDecisionMCPCapability {
    async fn bind(&self, request: BindingRequest) -> anyhow::Result<Binding> {
        let McpBindingRequest {
            supported_transports,
        } = match request {
            BindingRequest::Mcp(request) => request,
            other => anyhow::bail!(
                "DecisionMCPCapability does not handle {:?} binding requests",
                other.binding_type()
            ),
        };
        anyhow::ensure!(
            supported_transports.contains(&McpTransportKind::Stdio),
            "no MCP transport in common with the requesting launcher (it offered {supported_transports:?}; \
             DecisionMCPCapability only serves stdio)"
        );

        let model_id = &self.inner.config.model_id;
        let (provider, endpoint, model_name) = self.configured_model.resolve_provider_endpoint(
            model_id,
            ApiType::Jev,
            ModelFunction::Decision,
        )?;

        let package = match &self.inner.config.version {
            Some(version) => format!("{JEV_MCP_PACKAGE}#{version}"),
            None => JEV_MCP_PACKAGE.to_string(),
        };

        let env = HashMap::from([
            // The server insists on a key being set even when the endpoint
            // does not check one.
            (
                "TYPESAFE_API_KEY".to_string(),
                provider
                    .api_key()
                    .map(|k| k.0.clone())
                    .unwrap_or_else(|| "unused".to_string()),
            ),
            (
                "JEV_BASE_URL".to_string(),
                jev_base_url(provider.base_url(), endpoint.path()),
            ),
            ("JEV_MODEL".to_string(), model_name),
        ]);

        Ok(Binding::Mcp(McpBinding::Stdio {
            command: "npx".to_string(),
            args: vec!["-y".to_string(), package],
            env,
            timeout: None,
        }))
    }
}

impl HasCapabilityMetadata for DecisionMCPCapability {
    fn metadata() -> CapabilityMetadata {
        CapabilityMetadata {
            name: "Decision MCP Server".to_string(),
            description: DESCRIPTION.to_string(),
            dependencies: vec![Dependency::Model {
                config_key: "model_id".to_string(),
                requirement: ModelRequirement {
                    supported_functions: vec![ModelFunction::Decision],
                    ..Default::default()
                },
                resolved_id: None,
                required: true,
            }],
            tags: vec!["decision".to_string(), "mcp".to_string()],
            supported_binding_types: HashSet::from([BindingType::Mcp]),
        }
    }
}

/*-- private ------------------------------------------------------------------*/

const DESCRIPTION: &str =
    "Exposes a decision model as an MCP server (jev-mcp) for a launched coding agent to call.";

/// `jev-mcp` takes the full Jev endpoint URL (e.g.
/// `http://host:9797/v1/systemone`) as its base URL, so the endpoint path is
/// appended whole.
fn jev_base_url(base_url: &str, endpoint_path: &str) -> String {
    format!("{}{endpoint_path}", base_url.trim_end_matches('/'))
}

/*-- tests --------------------------------------------------------------------*/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::{CapabilityInfo, ResolvedCapability};
    use crate::providers::ApiEndpoint;
    use crate::utils::test_support::{FakeModel, FakeProvider};
    use std::sync::Arc;

    fn jev_provider(base_url: &str) -> FakeProvider {
        FakeProvider {
            instance_id: "my-jev".to_string(),
            base_url: base_url.to_string(),
            api_key: None,
            verify_ssl: true,
            api_types: vec![ApiType::Jev],
            endpoints: HashMap::from([(ModelFunction::Decision, vec![ApiEndpoint::JevSystemOne])]),
            alias: Some("ibm-granite/granite-switch-4.1-8b".to_string()),
        }
    }

    fn capability(provider: FakeProvider) -> ResolvedDecisionMCPCapability {
        capability_with(
            provider,
            serde_json::json!({ "model_id": "granite-switch-4.1-8b" }),
        )
    }

    fn capability_with(
        provider: FakeProvider,
        settings: serde_json::Value,
    ) -> ResolvedDecisionMCPCapability {
        let inner = DecisionMCPCapability::new("decision", &settings).unwrap();
        ResolvedDecisionMCPCapability {
            inner,
            configured_model: ConfiguredModel::for_test(
                Arc::new(FakeModel::text(vec![ModelFunction::Decision])),
                Arc::new(provider),
                None,
            ),
        }
    }

    fn request(transports: impl IntoIterator<Item = McpTransportKind>) -> BindingRequest {
        BindingRequest::Mcp(McpBindingRequest {
            supported_transports: transports.into_iter().collect(),
        })
    }

    #[test]
    fn binding_types_reports_mcp() {
        let cap = capability(jev_provider("http://localhost:9797"));
        assert_eq!(cap.binding_types(), HashSet::from([BindingType::Mcp]));
    }

    #[test]
    fn empty_model_id_is_rejected() {
        assert!(DecisionMCPCapability::new("d", &serde_json::json!({ "model_id": "" })).is_err());
    }

    #[test]
    fn metadata_requires_a_decision_model() {
        let meta = DecisionMCPCapability::metadata();
        let Dependency::Model { requirement, .. } = &meta.dependencies[0] else {
            panic!("expected a model dependency");
        };
        assert_eq!(
            requirement.supported_functions,
            vec![ModelFunction::Decision]
        );
    }

    #[tokio::test]
    async fn bind_returns_the_npx_stdio_command_with_jev_env() {
        let cap = capability(jev_provider("http://localhost:9797/"));
        let binding = cap.bind(request([McpTransportKind::Stdio])).await.unwrap();
        let Binding::Mcp(McpBinding::Stdio {
            command, args, env, ..
        }) = binding
        else {
            panic!("expected a stdio MCP binding");
        };
        assert_eq!(command, "npx");
        assert_eq!(args, vec!["-y", "github:codaaiteam/jev-mcp"]);
        assert_eq!(env["TYPESAFE_API_KEY"], "unused");
        assert_eq!(env["JEV_BASE_URL"], "http://localhost:9797/v1/systemone");
        assert_eq!(env["JEV_MODEL"], "ibm-granite/granite-switch-4.1-8b");
    }

    #[tokio::test]
    async fn bind_pins_the_package_when_a_version_is_set() {
        let cap = capability_with(
            jev_provider("http://localhost:9797"),
            serde_json::json!({ "model_id": "granite-switch-4.1-8b", "version": "v1.2.3" }),
        );
        let Binding::Mcp(McpBinding::Stdio { args, .. }) =
            cap.bind(request([McpTransportKind::Stdio])).await.unwrap()
        else {
            panic!("expected a stdio MCP binding");
        };
        assert_eq!(args, vec!["-y", "github:codaaiteam/jev-mcp#v1.2.3"]);
    }

    #[test]
    fn empty_version_is_rejected() {
        let cfg = serde_json::json!({ "model_id": "m", "version": "" });
        assert!(DecisionMCPCapability::new("d", &cfg).is_err());
    }

    #[tokio::test]
    async fn bind_fails_without_stdio_transport() {
        let cap = capability(jev_provider("http://localhost:9797"));
        let err = cap
            .bind(request([McpTransportKind::Http]))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("only serves stdio"));
    }

    #[tokio::test]
    async fn bind_fails_when_provider_has_no_jev_api() {
        let provider = FakeProvider {
            api_types: vec![ApiType::OpenAI],
            ..jev_provider("http://localhost:9797")
        };
        let cap = capability(provider);
        assert!(cap.bind(request([McpTransportKind::Stdio])).await.is_err());
    }

    #[tokio::test]
    async fn bind_rejects_other_binding_types() {
        let cap = capability(jev_provider("http://localhost:9797"));
        let other = BindingRequest::AgentModel(crate::capabilities::AgentModelBindingRequest {
            api_type: ApiType::OpenAI,
        });
        assert!(cap.bind(other).await.is_err());
    }
}
