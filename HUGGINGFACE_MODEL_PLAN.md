# Plan: HuggingFaceModel

## Overview

This introduces a new model type, `HuggingFaceModel`, that works similarly to `CustomModel` but pre-fills many fields (variant, size, description, etc.) from the HuggingFace repository itself at setup time. The `HuggingFaceModelConfig` only takes the HF repo name.

## Architecture

`HuggingFaceModel` is registered in the `MODEL_REGISTRY` under the key `"huggingface"`, mirroring how `CustomModel` is registered under `"custom"`.

The model is implemented in `src/models/huggingface.rs`:

- **`HuggingFaceModelConfig`** — simple config holding just the HF repo name
- **`HuggingFaceModel`** — implements `Model`, `ConfigConstructable`, `Named`, `HasModelMetadata`

## Files

### 1. `src/models/huggingface.rs` — New file

#### `HuggingFaceModelConfig`

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default, schemars::JsonSchema)]
pub struct HuggingFaceModelConfig {
    /// The HuggingFace repository (owner/repo or URL, e.g. `ibm-granite/granite-speech-4.1-2b`
    /// or `https://huggingface.co/ibm-granite/granite-speech-4.1-2b`)
    pub huggingface_repo: String,
}
```

- Default config: `HuggingFaceModelConfig { huggingface_repo: String::new() }`

#### `HuggingFaceModel`

```rust
pub struct HuggingFaceModel {
    instance_id: String,
    config: HuggingFaceModelConfig,
}
```

#### Implementations

**`ConfigConstructable`**:
- `type Config = HuggingFaceModelConfig`
- `fn new(instance_id: &str, cfg: &serde_json::Value) -> Result<Self, ConstructError>`
  - Reads `huggingface_repo` from config
  - If the config is empty/default, returns an error (a repo name is required)

**`Named`**:
- Returns `instance_id`

**`Model`**: All trait methods delegate to a lazy metadata fetch from the HF repo.

**`HasModelMetadata`**: `metadata()` returns the pre-populated metadata fetched from the repo.

#### Metadata Fields

```rust
pub struct ModelMetadata {
    pub family: String,
    pub version: String,
    pub size: u64,
    pub context_length: u64,
    pub model_type: ModelType,
    pub huggingface_repo: String,
    pub native_dtype: String,
    pub architecture: ModelArchitecture,
    pub variants: Vec<ModelVariant>,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub supported_functions: Vec<ModelFunction>,
}
```

**Variant Population (GGUF only)**: Always auto-populate variants for GGUF repos based on the standard HF quant naming convention `name-QUANT.gguf`, where precision = the part after the last `-`. For example:
- `granite-4.2-3b-Q3_K_L.gguf` → precision `Q3_K_L`, size from filename
- `granite-4.2-3b-Q4_0.gguf` → precision `Q4_0`, size from filename

**Supported Functions**: Use the same conventions as `update-models`:
- Text → `[Chat]`
- Vision → `[Chat, ImageUnderstanding]`
- Speech → `[Chat, Transcription]`
- Embedding → `[Embeddings]`
- For `Granite Guardian` family → `[Guardian]` only (overrides model_type/chat-template-derived rules)
- Detect `ToolCalling` and `Thinking` from chat template signals (Granite Guardian is excluded from this detection)

**Architecture**: Try to get from `config.json` if available on the HF repo; otherwise use stub values:
```rust
static DEFAULT_ARCHITECTURE: std::sync::LazyLock<ModelArchitecture> =
    std::sync::LazyLock::new(|| ModelArchitecture {
        num_hidden_layers: 0,
        hidden_size: 0,
        num_attention_heads: 0,
        num_key_value_heads: 0,
        head_dim: 0,
        layer_types: vec![],
    });
```

**Description**: From the model card description on HuggingFace. If unavailable, use a placeholder.

**Tags**: Derived from family name (e.g. family lowercase as tags: `["granite", "language"]`).

### 2. `src/models/mod.rs`

- Register `HuggingFaceModel` in the `MODEL_REGISTRY`:
  ```rust
  MODEL_REGISTRY.register::<huggingface::HuggingFaceModel>("huggingface");
  ```
- Re-export `HuggingFaceModelConfig`:
  ```rust
  pub use huggingface::HuggingFaceModelConfig;
  ```

### 3. `src/commands/model.rs` — `ModelCommands::setup`

The `setup` method already handles any model type in the registry. It works generically:

1. Gets the model type from the registry (e.g. `"huggingface"`)
2. Shows placeholder info (family, size, context length, type) — from `HuggingFaceModel::to_metadata()`
3. Prompts for config fields (`huggingface_repo`)
4. Constructs a live instance — this fetches metadata from the HF repo
5. Shows variants (auto-populated for GGUF repos)
6. Selects variant (if any), then provider
7. Saves config

The flow mirrors `CustomModel::setup` but uses the HF repo name instead of the custom config.

### 4. `src/commands/setup.rs` — Discovery

Since `HuggingFaceModel` is registered in the `MODEL_REGISTRY`, it appears automatically in:
- `model discover` — all configured huggingface models
- `model list` — all configured instances
- `model info` — type metadata
- `model setup huggingface` — configure a new instance

No changes needed here.

## Setup Flow for `huggingface` Model Type

1. **Prompt for instance name**: Defaults to the model type key if not given
2. **Show placeholder info** (family, size, context length, type):
   ```
   Setting up model: huggingface

   Size: 3B params, 8192 context
   Type: Text
   ```
3. **Prompt for config** (only `huggingface_repo`):
   ```
   Enter the HuggingFace repository: __________
   ```
4. **Construct live instance** — fetches metadata from the HF repo:
   - `family` — from model card title
   - `version` — from model card version
   - `size` — from model card parameters
   - `context_length` — from model card context length
   - `native_dtype` — from model card dtype
   - `architecture` — from config.json (if available) or stubs
   - `description` — from model card description
   - `tags` — from family
   - `supported_functions` — from model_type + Guardian family override
   - `variants` — auto-populated for GGUF repos only (safetensors repos have no variants)
5. **Select variant** (if any): Show GGUF variants, user selects
6. **Select provider**: If variant selected, pick a provider that can run that format/precision; otherwise any provider
7. **Save configuration**:
   ```json
   {
       "model_id": "my-instance",
       "model_type": "huggingface",
       "provider_id": "my-provider",
       "variant": "GGUF/Q4_0",
       "config": {
           "huggingface_repo": "ibm-granite/granite-4.2-3b"
       }
   }
   ```

## Error Handling

- **Network errors / unreachable**: Construction fails with a clear error. This is the desired behavior (per clarification #3)
- **Unavailable repo / 404**: Show an error and fail construction
- **Invalid HF repo format**: Validate the repo string before attempting construction

## Supported Functions Detection (Granite Guardian special case)

Granite Guardian models override everything and always get `supported_functions: [Guardian]` regardless of model_type or chat template. This is checked in the metadata: if `family == "Granite Guardian"`, return `[Guardian]`.

For other models, derive from `model_type` with the following conventions (matching `update-models`):

| model_type | supported_functions |
|---|---|
| Text | `[Chat]` |
| Vision | `[Chat, ImageUnderstanding]` |
| Speech | `[Chat, Transcription]` |
| Embedding | `[Embeddings]` |
| ToolCalling / Thinking | Detected from chat template signals (Granite Guardian excluded) |

Chat template detection (Granite Guardian excluded):
- **ToolCalling**: template has a Jinja `if`/`elif` on the bare `tools` variable
- **Thinking**: template has `enable_thinking`, `reasoning_content`, or `<think>` token

## Implementation Order

1. Create `src/models/huggingface.rs` with `HuggingFaceModelConfig` and `HuggingFaceModel`
2. Register `HuggingFaceModel` in `src/models/mod.rs`
3. Modify `ModelCommands::setup` to handle `huggingface` type (already generic — verify it works)
4. Implement metadata fetch helpers (`fetch_repo_metadata`, `fetch_gguf_variants`, `parse_gguf_filename`)
5. Implement `Model::to_metadata()` and `HasModelMetadata::metadata()`
6. Add tests
7. Run `./scripts/fmt.sh` and verify with `cargo check`

## Notes

- The `HuggingFaceModel` implements `ConfigConstructable` which is required by the factory pattern
- `HasModelMetadata` is used for catalog browsing and CLI display
- For GGUF repos, `variants` will be non-empty (auto-populated). For safetensors repos, `variants` will be empty (user must configure them manually if needed)
- The `description` field comes from the model card description on HuggingFace
- The `architecture` field comes from config.json on the HF repo if available; otherwise stub values
- The `native_dtype` comes from the model card dtype
- Safetensors repos: no variants (empty `variants` slice)
- GGUF repos: variants auto-populated from `.gguf` filenames using the convention `name-QUANT.gguf`
