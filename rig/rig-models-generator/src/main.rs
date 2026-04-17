use anyhow::{Context, Result, anyhow, bail};
use rig::client::{ModelListingClient, ProviderClient};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct CatalogFile {
    models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelEntry {
    provider: String,
    id: String,
    const_name: String,
    #[serde(default)]
    aliases: Vec<String>,
    operations: Vec<String>,
    #[serde(default)]
    completion: Option<CompletionEntry>,
    #[serde(default)]
    embedding: Option<EmbeddingEntry>,
    #[serde(default)]
    image_generation: Option<ImageGenerationEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct CompletionEntry {
    #[serde(default)]
    request_api: Option<String>,
    #[serde(default)]
    default_max_tokens: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
struct EmbeddingEntry {
    #[serde(default)]
    default_dimensions: Option<usize>,
    #[serde(default)]
    supports_dimensions_override: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct ImageGenerationEntry {
    #[serde(default)]
    response_format_mode: Option<String>,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root should exist")
        .to_path_buf()
}

fn catalog_path() -> PathBuf {
    repo_root().join("rig/rig-models/data/models.json")
}

fn generated_path() -> PathBuf {
    repo_root().join("rig/rig-models/src/generated.rs")
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "generate".to_string());

    match command.as_str() {
        "generate" => generate(false),
        "check" => generate(true),
        "verify-listers" => verify_listers(),
        other => bail!("unknown command `{other}`"),
    }
}

fn generate(check_only: bool) -> Result<()> {
    let catalog = load_catalog(&catalog_path())?;
    validate_catalog(&catalog)?;
    let rendered = render_generated(&catalog)?;

    if check_only {
        let existing = fs::read_to_string(generated_path())
            .context("failed to read checked-in generated catalog")?;
        if existing != rendered {
            bail!(
                "generated catalog is out of date; run `cargo run -p rig-models-generator -- generate`"
            );
        }
        return Ok(());
    }

    fs::write(generated_path(), rendered).context("failed to write generated catalog")?;
    Ok(())
}

fn load_catalog(path: &Path) -> Result<CatalogFile> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read catalog file {}", path.display()))?;
    let catalog = serde_json::from_str::<CatalogFile>(&raw)
        .with_context(|| format!("failed to parse catalog file {}", path.display()))?;
    Ok(catalog)
}

fn validate_catalog(catalog: &CatalogFile) -> Result<()> {
    let mut seen_model_ids = BTreeSet::new();
    let mut seen_const_names = BTreeSet::new();

    for model in &catalog.models {
        if !seen_model_ids.insert((model.provider.clone(), model.id.clone())) {
            bail!(
                "duplicate provider/id entry for provider `{}` model `{}`",
                model.provider,
                model.id
            );
        }

        if !seen_const_names.insert((model.provider.clone(), model.const_name.clone())) {
            bail!(
                "duplicate provider/const_name entry for provider `{}` const `{}`",
                model.provider,
                model.const_name
            );
        }

        for alias in &model.aliases {
            if !seen_const_names.insert((model.provider.clone(), alias.clone())) {
                bail!(
                    "duplicate provider/const_name entry for provider `{}` const `{}`",
                    model.provider,
                    alias
                );
            }
        }

        if model.operations.is_empty() {
            bail!(
                "catalog entry {}::{} must declare at least one operation",
                model.provider,
                model.const_name
            );
        }

        for operation in &model.operations {
            if !matches!(
                operation.as_str(),
                "completion"
                    | "embedding"
                    | "image_generation"
                    | "audio_generation"
                    | "transcription"
            ) {
                bail!(
                    "catalog entry {}::{} uses unknown operation `{}`",
                    model.provider,
                    model.const_name,
                    operation
                );
            }
        }

        if let Some(completion) = &model.completion
            && let Some(request_api) = &completion.request_api
            && !matches!(request_api.as_str(), "chat_completions" | "responses")
        {
            bail!(
                "catalog entry {}::{} uses unknown completion.request_api `{}`",
                model.provider,
                model.const_name,
                request_api
            );
        }

        if let Some(image_generation) = &model.image_generation
            && let Some(mode) = &image_generation.response_format_mode
            && !matches!(mode.as_str(), "include_b64_json" | "omit_field")
        {
            bail!(
                "catalog entry {}::{} uses unknown image_generation.response_format_mode `{}`",
                model.provider,
                model.const_name,
                mode
            );
        }
    }

    Ok(())
}

fn render_generated(catalog: &CatalogFile) -> Result<String> {
    let mut providers: BTreeMap<&str, Vec<&ModelEntry>> = BTreeMap::new();
    for model in &catalog.models {
        providers.entry(&model.provider).or_default().push(model);
    }

    let mut rendered = String::new();
    rendered.push_str("// @generated by rig-models-generator. Do not edit by hand.\n\n");
    rendered.push_str("use crate::{\n");
    rendered.push_str("    CompletionMetadata, CompletionRequestApi, EmbeddingMetadata, ImageGenerationMetadata,\n");
    rendered.push_str("    ImageGenerationResponseFormatMode, ModelMetadata, Operations,\n");
    rendered.push_str("};\n\n");

    for (provider, models) in &providers {
        if needs_non_upper_case_globals_allow(models) {
            rendered.push_str("#[allow(non_upper_case_globals)]\n");
        }
        writeln!(&mut rendered, "pub mod {provider} {{")?;
        let imports = module_imports(models);
        rendered.push_str("    use super::{\n");
        for chunk in imports.chunks(4) {
            rendered.push_str("        ");
            for (index, import) in chunk.iter().enumerate() {
                if index > 0 {
                    rendered.push(' ');
                }
                rendered.push_str(import);
                rendered.push(',');
            }
            rendered.push('\n');
        }
        rendered.push_str("    };\n\n");

        for model in models {
            writeln!(
                &mut rendered,
                "    pub const {}: &str = {:?};",
                model.const_name, model.id
            )?;
            for alias in &model.aliases {
                writeln!(
                    &mut rendered,
                    "    pub const {alias}: &str = {:?};",
                    model.id
                )?;
            }
        }

        if !models.is_empty() {
            rendered.push('\n');
        }

        rendered.push_str("    pub const ALL_MODELS: &[ModelMetadata] = &[\n");
        for model in models {
            writeln!(
                &mut rendered,
                "        ModelMetadata {{ provider: {:?}, id: {:?}, const_name: {:?}, operations: {}, completion: {}, embedding: {}, image_generation: {} }},",
                model.provider,
                model.id,
                model.const_name,
                render_operations(model),
                render_completion(model.completion.as_ref()),
                render_embedding(model.embedding.as_ref()),
                render_image_generation(model.image_generation.as_ref()),
            )?;
        }
        rendered.push_str("    ];\n\n");

        rendered.push_str("    pub fn lookup(id: &str) -> Option<&'static ModelMetadata> {\n");
        rendered.push_str("        match id {\n");
        for (index, model) in models.iter().enumerate() {
            writeln!(
                &mut rendered,
                "            {:?} => Some(&ALL_MODELS[{index}]),",
                model.id
            )?;
        }
        rendered.push_str("            _ => None,\n");
        rendered.push_str("        }\n");
        rendered.push_str("    }\n");
        rendered.push_str("}\n\n");
    }

    rendered.push_str("pub const ALL_MODELS: &[ModelMetadata] = &[\n");
    for models in providers.values() {
        for model in models {
            writeln!(
                &mut rendered,
                "    ModelMetadata {{ provider: {:?}, id: {:?}, const_name: {:?}, operations: {}, completion: {}, embedding: {}, image_generation: {} }},",
                model.provider,
                model.id,
                model.const_name,
                render_operations(model),
                render_completion(model.completion.as_ref()),
                render_embedding(model.embedding.as_ref()),
                render_image_generation(model.image_generation.as_ref()),
            )?;
        }
    }
    rendered.push_str("];\n\n");
    rendered.push_str("pub fn all_models() -> &'static [ModelMetadata] {\n");
    rendered.push_str("    ALL_MODELS\n");
    rendered.push_str("}\n\n");
    rendered
        .push_str("pub fn lookup(provider: &str, id: &str) -> Option<&'static ModelMetadata> {\n");
    rendered.push_str("    match provider {\n");
    for provider in providers.keys() {
        writeln!(
            &mut rendered,
            "        {:?} => {provider}::lookup(id),",
            provider
        )?;
    }
    rendered.push_str("        _ => None,\n");
    rendered.push_str("    }\n");
    rendered.push_str("}\n");

    Ok(rendered)
}

fn module_imports(models: &[&ModelEntry]) -> Vec<&'static str> {
    let mut imports = Vec::new();

    if models.iter().any(|model| model.completion.is_some()) {
        imports.push("CompletionMetadata");
    }
    if models
        .iter()
        .filter_map(|model| model.completion.as_ref())
        .any(|metadata| metadata.request_api.is_some())
    {
        imports.push("CompletionRequestApi");
    }
    if models.iter().any(|model| model.embedding.is_some()) {
        imports.push("EmbeddingMetadata");
    }
    if models.iter().any(|model| model.image_generation.is_some()) {
        imports.push("ImageGenerationMetadata");
    }
    if models
        .iter()
        .filter_map(|model| model.image_generation.as_ref())
        .any(|metadata| metadata.response_format_mode.is_some())
    {
        imports.push("ImageGenerationResponseFormatMode");
    }

    imports.push("ModelMetadata");
    imports.push("Operations");
    imports
}

fn needs_non_upper_case_globals_allow(models: &[&ModelEntry]) -> bool {
    models.iter().any(|model| {
        !is_upper_const_name(&model.const_name)
            || model
                .aliases
                .iter()
                .any(|alias| !is_upper_const_name(alias))
    })
}

fn is_upper_const_name(name: &str) -> bool {
    name.chars()
        .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
}

fn render_operations(model: &ModelEntry) -> String {
    let has = |name: &str| model.operations.iter().any(|operation| operation == name);
    format!(
        "Operations::new({}, {}, {}, {}, {})",
        has("completion"),
        has("embedding"),
        has("image_generation"),
        has("audio_generation"),
        has("transcription"),
    )
}

fn render_completion(metadata: Option<&CompletionEntry>) -> String {
    match metadata {
        Some(metadata) => format!(
            "Some(CompletionMetadata {{ request_api: {}, default_max_tokens: {} }})",
            render_request_api(metadata.request_api.as_deref()),
            render_option_u64(metadata.default_max_tokens),
        ),
        None => "None".to_string(),
    }
}

fn render_request_api(request_api: Option<&str>) -> &'static str {
    match request_api {
        Some("chat_completions") => "Some(CompletionRequestApi::ChatCompletions)",
        Some("responses") => "Some(CompletionRequestApi::Responses)",
        Some(other) => panic!("unsupported request_api {other}"),
        None => "None",
    }
}

fn render_embedding(metadata: Option<&EmbeddingEntry>) -> String {
    match metadata {
        Some(metadata) => format!(
            "Some(EmbeddingMetadata {{ default_dimensions: {}, supports_dimensions_override: {} }})",
            render_option_usize(metadata.default_dimensions),
            render_option_bool(metadata.supports_dimensions_override),
        ),
        None => "None".to_string(),
    }
}

fn render_image_generation(metadata: Option<&ImageGenerationEntry>) -> String {
    match metadata {
        Some(metadata) => format!(
            "Some(ImageGenerationMetadata {{ response_format_mode: {} }})",
            match metadata.response_format_mode.as_deref() {
                Some("include_b64_json") => {
                    "Some(ImageGenerationResponseFormatMode::IncludeB64Json)"
                }
                Some("omit_field") => "Some(ImageGenerationResponseFormatMode::OmitField)",
                Some(other) => panic!("unsupported response_format_mode {other}"),
                None => "None",
            }
        ),
        None => "None".to_string(),
    }
}

fn render_option_bool(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "Some(true)",
        Some(false) => "Some(false)",
        None => "None",
    }
}

fn render_option_u64(value: Option<u64>) -> String {
    value
        .map(|value| format!("Some({value})"))
        .unwrap_or_else(|| "None".to_string())
}

fn render_option_usize(value: Option<usize>) -> String {
    value
        .map(|value| format!("Some({value})"))
        .unwrap_or_else(|| "None".to_string())
}

fn verify_listers() -> Result<()> {
    let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    runtime.block_on(verify_listers_async())
}

async fn verify_listers_async() -> Result<()> {
    let catalog = load_catalog(&catalog_path())?;
    validate_catalog(&catalog)?;

    verify_provider_models(
        "openai",
        &catalog,
        rig::providers::openai::Client::from_env()
            .list_models()
            .await,
    )
    .await?;
    verify_provider_models(
        "openrouter",
        &catalog,
        rig::providers::openrouter::Client::from_env()
            .list_models()
            .await,
    )
    .await?;
    verify_provider_models(
        "anthropic",
        &catalog,
        rig::providers::anthropic::Client::from_env()
            .list_models()
            .await,
    )
    .await?;
    verify_provider_models(
        "gemini",
        &catalog,
        rig::providers::gemini::Client::from_env()
            .list_models()
            .await,
    )
    .await?;
    verify_provider_models(
        "mistral",
        &catalog,
        rig::providers::mistral::Client::from_env()
            .list_models()
            .await,
    )
    .await?;
    verify_provider_models(
        "ollama",
        &catalog,
        rig::providers::ollama::Client::from_env()
            .list_models()
            .await,
    )
    .await?;

    Ok(())
}

async fn verify_provider_models(
    provider: &str,
    catalog: &CatalogFile,
    models: std::result::Result<rig::model::ModelList, rig::model::ModelListingError>,
) -> Result<()> {
    let known_ids = catalog
        .models
        .iter()
        .filter(|model| model.provider == provider)
        .map(|model| model.id.as_str())
        .collect::<BTreeSet<_>>();

    match models {
        Ok(models) => {
            let live_ids = models
                .data
                .iter()
                .map(|model| model.id.as_str())
                .collect::<BTreeSet<_>>();
            let missing_from_catalog = live_ids.difference(&known_ids).copied().collect::<Vec<_>>();

            if !missing_from_catalog.is_empty() {
                bail!(
                    "provider `{provider}` has live models missing from the catalog: {}",
                    missing_from_catalog.join(", ")
                );
            }
        }
        Err(error) => {
            return Err(anyhow!(
                "provider `{provider}` model listing failed: {error}"
            ));
        }
    }

    Ok(())
}
