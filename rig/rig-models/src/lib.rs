//! Generated provider model catalog for Rig.

mod generated;

/// Provider operations supported by a model catalog entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Operations {
    pub completion: bool,
    pub embedding: bool,
    pub image_generation: bool,
    pub audio_generation: bool,
    pub transcription: bool,
}

impl Operations {
    pub const fn new(
        completion: bool,
        embedding: bool,
        image_generation: bool,
        audio_generation: bool,
        transcription: bool,
    ) -> Self {
        Self {
            completion,
            embedding,
            image_generation,
            audio_generation,
            transcription,
        }
    }
}

/// Request API routing metadata for completion models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionRequestApi {
    ChatCompletions,
    Responses,
}

/// Completion-specific metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CompletionMetadata {
    pub request_api: Option<CompletionRequestApi>,
    pub default_max_tokens: Option<u64>,
}

/// Embedding-specific metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EmbeddingMetadata {
    pub default_dimensions: Option<usize>,
    pub supports_dimensions_override: Option<bool>,
}

/// Image generation response formatting metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageGenerationResponseFormatMode {
    IncludeB64Json,
    OmitField,
}

/// Image-generation-specific metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ImageGenerationMetadata {
    pub response_format_mode: Option<ImageGenerationResponseFormatMode>,
}

/// Static metadata for a single provider model identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelMetadata {
    pub provider: &'static str,
    pub id: &'static str,
    pub const_name: &'static str,
    pub operations: Operations,
    pub completion: Option<CompletionMetadata>,
    pub embedding: Option<EmbeddingMetadata>,
    pub image_generation: Option<ImageGenerationMetadata>,
}

pub use generated::lookup;
pub use generated::{
    all_models, anthropic, azure, chatgpt, cohere, copilot, deepseek, galadriel, gemini, groq,
    huggingface, hyperbolic, llamafile, minimax, mistral, moonshot, ollama, openai, openrouter,
    perplexity, together, voyageai, xai, zai,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_lookup_returns_expected_constant_id() {
        let model = openai::lookup(openai::GPT_4O).expect("openai::GPT_4O should exist");
        assert_eq!(model.id, "gpt-4o");
        assert!(model.operations.completion);
    }

    #[test]
    fn openai_image_generation_metadata_tracks_response_format_mode() {
        let model =
            openai::lookup(openai::GPT_IMAGE_1_5).expect("openai::GPT_IMAGE_1_5 should exist");
        let metadata = model
            .image_generation
            .expect("gpt-image-1.5 should have image metadata");

        assert_eq!(
            metadata.response_format_mode,
            Some(ImageGenerationResponseFormatMode::OmitField)
        );
    }

    #[test]
    fn copilot_completion_metadata_tracks_request_api() {
        let model =
            copilot::lookup(copilot::GPT_5_3_CODEX).expect("copilot::GPT_5_3_CODEX should exist");
        let metadata = model
            .completion
            .expect("gpt-5.3-codex should have completion metadata");

        assert_eq!(metadata.request_api, Some(CompletionRequestApi::Responses));
    }

    #[test]
    fn embedding_dimensions_are_available_for_known_models() {
        let model =
            gemini::lookup(gemini::EMBEDDING_001).expect("gemini::EMBEDDING_001 should exist");
        let metadata = model
            .embedding
            .expect("gemini::EMBEDDING_001 should have embedding metadata");

        assert_eq!(metadata.default_dimensions, Some(3_072));
    }

    #[test]
    fn catalog_entries_are_unique() {
        use std::collections::BTreeSet;

        let mut ids = BTreeSet::new();
        let mut const_names = BTreeSet::new();

        for model in all_models() {
            assert!(
                ids.insert((model.provider, model.id)),
                "duplicate provider/id for {}::{}",
                model.provider,
                model.id
            );
            assert!(
                const_names.insert((model.provider, model.const_name)),
                "duplicate provider/const_name for {}::{}",
                model.provider,
                model.const_name
            );
        }
    }

    #[test]
    fn together_catalog_includes_multiline_constant_models() {
        let model = together::lookup(together::LLAMA_3_2_90B_VISION_INSTRUCT_TURBO)
            .expect("together vision model should exist");

        assert_eq!(model.id, "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo");
        assert!(model.operations.completion);
    }
}
