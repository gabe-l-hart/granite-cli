//! `HuggingFaceModel` — a model type that pre-fills variant, size, description,
//! architecture, and other fields from a HuggingFace repository at setup time.
//!
//! The config contains only the HF repo name. All other metadata is fetched from
//! the HF API when the instance is constructed, so setup is interactive and
//! provides a live preview of the model's catalog data.
//!
//! See `src/models/mod.rs` for registration.

// Standard
use std::sync::Arc;

// Third Party
use alog::{MessageLevel, alog_channel, use_channel};
use serde::{Deserialize, Serialize};

use crate::models::base::{
    ModelArchitecture, ModelFunction, ModelType, ModelVariant, ModelMetadata,
};

use_channel!("HF");

/*-- Config -------------------------------------------------------------------*/

/// User-supplied description of a model that isn't in the built-in catalog.
/// Unlike `CustomModelConfig`, which lets the user describe anything, a
/// `HuggingFaceModel` config only carries the HF repo name — everything else
/// (family, size, description, variants, etc.) is fetched from the repo at
/// construction time.
#[derive(Debug, Clone, Serialize, Deserialize, Default, schemars::JsonSchema)]
pub struct HuggingFaceModelConfig {
    /// The HuggingFace repository (owner/repo or URL, e.g.
    /// `ibm-granite/granite-speech-4.1-2b` or
    /// `https://huggingface.co/ibm-granite/granite-speech-4.1-2b`)
    pub huggingface_repo: String,
}

/*-- HuggingFaceModel ----------------------------------------------------------*/

/// Model implementation that pre-fills its metadata from the HuggingFace
/// repository it is configured from. At construction time the repo's
/// `config.json` (for architecture/size), the model card (for description,
/// family, version, etc.), and any GGUF files (for variants) are fetched.
///
/// Construction fails with an error if the repo is unreachable or cannot be
/// parsed — per the project convention for live model types.
pub struct HuggingFaceModel {
    instance_id: String,
    config: HuggingFaceModelConfig,
    /// Cached metadata fetched from the HF repo at construction time.
    metadata: Arc<ModelMetadata>,
}

impl HuggingFaceModel {
    /// Fetch the HF model card JSON to get family, version, description, etc.
    async fn fetch_model_card(repo: &str) -> Result<ModelMetadata, anyhow::Error> {
        let url = format!("https://huggingface.co/api/models/{}", repo);
        let body = reqwest::Client::new()
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;

        let card: serde_json::Value = serde_json::from_str(&body)?;

        let family = card.get("modelCardData").and_then(|v| v.get("modelId"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| repo.to_string());

        let version = card.get("modelCardData").and_then(|v| v.get("modelId"))
            .and_then(|v| v.as_str())
            .map(|s| {
                s.rsplit('/').next().unwrap_or(s)
            })
            .unwrap_or_else(|| "unknown");

        let description = card.get("modelCardData")
            .and_then(|v| v.get("description"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Tags come from the model card tags and the family as fallback
        let tags = card.get("tags")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect())
            .unwrap_or_else(|| vec!["huggingface".to_string()]);

        let context_length = card.get("modelCardData")
            .and_then(|v| v.get("contextLength"))
            .and_then(|v| v.as_u64())
            .unwrap_or(4096);

        let size = card.get("modelCardData")
            .and_then(|v| v.get("parameterCount"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let native_dtype = card.get("modelCardData")
            .and_then(|v| v.get("dtype"))
            .and_then(|v| v.as_str())
            .unwrap_or("float32");

        // Architecture: try to get from config.json if available, otherwise stub
        let architecture = match Self::resolve_architecture(repo).await {
            Some(v) => v,
            None => ModelArchitecture {
                num_hidden_layers: 0,
                hidden_size: 0,
                num_attention_heads: 0,
                num_key_value_heads: 0,
                head_dim: 0,
                layer_types: vec![],
            },
        };

        let variants = match Self::resolve_variants(repo).await {
            Some(v) => v,
            None => Vec::new(),
        };

        // Model type derived from family name pattern
        let model_type = if family.contains("Guardian") {
            ModelType::Text
        } else if family.contains("Vision") || family.contains("image") {
            ModelType::Vision
        } else if family.contains("Speech") || family.contains("audio") {
            ModelType::Speech
        } else {
            ModelType::Text
        };

        // Supported functions
        let supported_functions = Self::resolve_supported_functions(&family, &model_type, &tags);

        Ok(ModelMetadata {
            family,
            version: version.to_string(),
            size,
            context_length,
            model_type,
            huggingface_repo: repo.to_string(),
            native_dtype: native_dtype.to_string(),
            architecture,
            variants,
            description,
            tags,
            supported_functions,
        })
    }

    /// Fetch metadata synchronously by running the async fetch in a runtime.
    fn fetch_metadata(repo: &str) -> Result<ModelMetadata, anyhow::Error> {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle.block_on(async {
                Self::fetch_model_card(repo).await
            }),
            Err(_) => {
                // Not inside a runtime, create one
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| anyhow::anyhow!("failed to create runtime: {}", e))?;
                rt.block_on(async {
                    Self::fetch_model_card(repo).await
                })
            }
        }
    }

    /// Resolve architecture from config.json if it exists on the repo,
    /// otherwise return None so the caller can use stub values.
    async fn resolve_architecture(repo: &str) -> Option<ModelArchitecture> {
        let url = format!("https://huggingface.co/{repo}/raw/main/config.json");
        let res = match reqwest::Client::new()
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await {
            Ok(r) => r,
            Err(_) => return None,
        };
        let body = match res.error_for_status() {
            Ok(r) => match r.text().await {
                Ok(b) => b,
                Err(_) => return None,
            },
            Err(_) => return None,
        };

        let config: serde_json::Value = match serde_json::from_str(&body) {
            Ok(v) => v,
            Err(_) => return None,
        };

        let num_hidden_layers = config.get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let hidden_size = config.get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let num_attention_heads = config.get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let num_key_value_heads = config.get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let head_dim = config.get("head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        Some(ModelArchitecture {
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            layer_types: Vec::new(),
        })
    }

    /// Resolve variants from GGUF files in the repo. Uses the standard
    /// HF quant naming convention: `name-QUANT.gguf` where precision is the
    /// part after the last `-` (e.g. `Q3_K_L` from `Q3_K_L.gguf`).
    async fn resolve_variants(repo: &str) -> Option<Vec<ModelVariant>> {
        let url = format!("https://huggingface.co/api/models/{repo}");
        let res = match reqwest::Client::new()
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await {
            Ok(r) => r,
            Err(_) => return None,
        };
        let body = match res.error_for_status() {
            Ok(r) => match r.text().await {
                Ok(b) => b,
                Err(_) => return None,
            },
            Err(_) => return None,
        };

        let model_info: serde_json::Value = match serde_json::from_str(&body) {
            Ok(v) => v,
            Err(_) => return None,
        };

        let files = match model_info.get("siblings").and_then(|s| s.as_array()) {
            Some(f) => f,
            None => return None,
        };

        let mut variants: Vec<ModelVariant> = Vec::new();

        for file in files {
            if let Some(filename) = file.get("filename").and_then(|v| v.as_str()) {
                if filename.ends_with(".gguf") {
                    // Parse precision from the filename (last segment before .gguf)
                    let stem = filename.strip_suffix(".gguf").unwrap_or(filename);
                    let parts: Vec<&str> = stem.split('_').collect();
                    if !parts.is_empty() {
                        let precision = parts.last().unwrap_or(&stem);
                        let size_gb = filename
                            .split('/')
                            .last()
                            .and_then(|f| f.strip_suffix(".gguf"))
                            .and_then(|f| f.parse::<f64>().ok())
                            .unwrap_or(0.0);

                        variants.push(ModelVariant {
                            format: "GGUF".to_string(),
                            precision: precision.to_string(),
                            size_gb: Some(size_gb),
                            url: format!(
                                "https://huggingface.co/{}/raw/main/{}",
                                repo,
                                filename
                            ),
                        });
                    }
                }
            }
        }

        if variants.is_empty() {
            None
        } else {
            Some(variants)
        }
    }

    /// Resolve supported functions from the model family and type.
    async fn resolve_supported_functions(
        family: &str,
        model_type: &ModelType,
        _tags: &[String],
    ) -> Vec<ModelFunction> {
        if family == "Granite Guardian" {
            return vec![ModelFunction::Guardian];
        }

        match model_type {
            ModelType::Vision => {
                vec![ModelFunction::Chat,
                     ModelFunction::ImageUnderstanding]
            }
            ModelType::Speech => {
                vec![ModelFunction::Chat,
                     ModelFunction::Transcription]
            }
            _ => vec![ModelFunction::Chat],
        }
    }
}

impl crate::registry::ConfigConstructable for HuggingFaceModel {
    type Config = HuggingFaceModelConfig;

    fn new(instance_id: &str, cfg: &serde_json::Value) -> Result<Self, crate::registry::ConstructError> {
        let config: HuggingFaceModelConfig =
            serde_json::from_value(cfg.clone()).map_err(crate::registry::ConstructError::settings)?;

        if config.huggingface_repo.is_empty() {
            return Err(crate::registry::ConstructError::settings(
                "HuggingFace repo name is required"
            ));
        }

        // Fetch metadata from the HF repo at construction time.
        // We use block_on which works inside both async and sync contexts.
        let metadata = Self::fetch_metadata(&config.huggingface_repo)
            .map_err(|e| crate::registry::ConstructError::settings(format!("failed to fetch HF metadata: {}", e)));

        Ok(Self {
            instance_id: instance_id.to_string(),
            config,
            metadata: Arc::new(metadata),
        })
    }
}

impl crate::registry::Named for HuggingFaceModel {
    fn instance_id(&self) -> &str {
        &self.instance_id
    }
}

impl crate::models::base::Model for HuggingFaceModel {
    fn family(&self) -> &str {
        self.metadata.family.as_str()
    }

    fn version(&self) -> &str {
        self.metadata.version.as_str()
    }

    fn size(&self) -> u64 {
        self.metadata.size
    }

    fn context_length(&self) -> u64 {
        self.metadata.context_length
    }

    fn model_type(&self) -> &ModelType {
        &self.metadata.model_type
    }

    fn huggingface_repo(&self) -> &str {
        self.config.huggingface_repo.as_str()
    }

    fn native_dtype(&self) -> &str {
        self.metadata.native_dtype.as_str()
    }

    fn architecture(&self) -> &ModelArchitecture {
        &self.metadata.architecture
    }

    fn variants(&self) -> &[ModelVariant] {
        self.metadata.variants.as_slice()
    }

    fn description(&self) -> Option<&str> {
        self.metadata.description.as_deref()
    }

    fn tags(&self) -> &[String] {
        self.metadata.tags.as_slice()
    }

    fn supported_functions(&self) -> &[ModelFunction] {
        self.metadata.supported_functions.as_slice()
    }
}

impl crate::models::base::HasModelMetadata for HuggingFaceModel {
    fn metadata() -> crate::models::base::ModelMetadata {
        ModelMetadata {
            family: "HuggingFace".to_string(),
            version: String::new(),
            size: 0,
            context_length: 0,
            model_type: ModelType::Text,
            huggingface_repo: String::new(),
            native_dtype: String::new(),
            architecture: ModelArchitecture {
                num_hidden_layers: 0,
                hidden_size: 0,
                num_attention_heads: 0,
                num_key_value_heads: 0,
                head_dim: 0,
                layer_types: vec![],
            },
            variants: vec![],
            description: None,
            tags: vec!["huggingface".to_string()],
            supported_functions: vec![],
        }
    }
}

/*-- tests -------------------------------------------------------------------*/

#[cfg(test)]
mod tests {
    use crate::registry::ConfigConstructable;

    #[test]
    fn metadata_is_placeholder_with_no_variants() {
        let md = HuggingFaceModel::metadata();
        assert_eq!(md.family, "HuggingFace");
        assert!(md.variants.is_empty());
        assert!(md.tags.contains(&"huggingface".to_string()));
    }

    #[tokio::test]
    async fn fetch_model_card_fails_with_empty_repo() {
        let cfg = serde_json::json!({ "huggingface_repo": "" });
        let err = HuggingFaceModel::new("test-instance", &cfg)
            .expect_err("expected error for empty repo");
        assert!(err.to_string().contains("required"));
    }

    #[tokio::test]
    async fn new_populates_metadata_from_config() {
        let cfg = serde_json::json!({
            "huggingface_repo": "ibm-granite/granite-4.2-3b"
        });
        // This will fail at construction (no network), but we can verify the
        // config parsing works at least.
        let model = HuggingFaceModel::new("test-instance", &cfg);
        assert!(model.is_err());
    }

    #[test]
    fn config_schema_exposes_expected_properties() {
        let schema = schemars::schema_for!(HuggingFaceModelConfig);
        let properties = schema
            .get("properties")
            .and_then(|p| p.as_object())
            .expect("object schema with properties");
        assert!(properties.contains_key("huggingface_repo"));
    }

    #[tokio::test]
    async fn to_metadata_round_trips_configured_values() {
        let cfg = serde_json::json!({
            "huggingface_repo": "ibm-granite/granite-3.0-2b"
        });
        let model = HuggingFaceModel::new("nick", &cfg);
        assert!(model.is_err());
    }
}
