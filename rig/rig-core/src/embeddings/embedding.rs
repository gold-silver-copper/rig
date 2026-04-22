//! The module defines the [EmbeddingModel] trait, which represents an embedding model that can
//! generate embeddings for documents.
//!
//! The module also defines the [Embedding] struct, which represents a single document embedding.
//!
//! Finally, the module defines the [EmbeddingError] enum, which represents various errors that
//! can occur during embedding generation or processing.

use crate::{
    http_client,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingResponseError {
    #[error("ResponseError: response data length does not match input length")]
    MismatchedEmbeddingCount,

    #[error("ResponseError: {message}")]
    Message { message: String },
}

impl EmbeddingResponseError {
    pub fn from_message(message: impl Into<String>) -> Self {
        let message = message.into();

        match message.as_str() {
            "Response data length does not match input length"
            | "Number of returned embeddings does not match input" => {
                Self::MismatchedEmbeddingCount
            }
            _ => Self::Message { message },
        }
    }
}

impl From<String> for EmbeddingResponseError {
    fn from(message: String) -> Self {
        Self::from_message(message)
    }
}

impl From<&str> for EmbeddingResponseError {
    fn from(message: &str) -> Self {
        Self::from_message(message)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingProviderError {
    #[error("ProviderError: {message}")]
    Message { message: String },
}

impl EmbeddingProviderError {
    pub fn from_message(message: impl Into<String>) -> Self {
        Self::Message {
            message: message.into(),
        }
    }
}

impl From<String> for EmbeddingProviderError {
    fn from(message: String) -> Self {
        Self::Message { message }
    }
}

impl From<&str> for EmbeddingProviderError {
    fn from(message: &str) -> Self {
        Self::Message {
            message: message.to_owned(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingInitializationError {
    #[error("InitializationError: {message}")]
    Message { message: String },
}

impl EmbeddingInitializationError {
    pub fn from_message(message: impl Into<String>) -> Self {
        Self::Message {
            message: message.into(),
        }
    }
}

impl From<String> for EmbeddingInitializationError {
    fn from(message: String) -> Self {
        Self::Message { message }
    }
}

impl From<&str> for EmbeddingInitializationError {
    fn from(message: &str) -> Self {
        Self::Message {
            message: message.to_owned(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// Http error (e.g.: connection error, timeout, etc.)
    #[error("HttpError: {0}")]
    HttpError(#[from] http_client::Error),

    /// Json error (e.g.: serialization, deserialization)
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("UrlError: {0}")]
    UrlError(#[from] url::ParseError),

    #[cfg(not(target_family = "wasm"))]
    /// Error building the embedding request
    #[error("RequestError: {0}")]
    RequestError(Box<dyn std::error::Error + Send + Sync + 'static>),

    #[cfg(target_family = "wasm")]
    /// Error building the embedding request
    #[error("RequestError: {0}")]
    RequestError(Box<dyn std::error::Error + 'static>),

    #[cfg(not(target_family = "wasm"))]
    /// Error processing the document for embedding
    #[error("DocumentError: {0}")]
    DocumentError(Box<dyn std::error::Error + Send + Sync + 'static>),

    #[cfg(target_family = "wasm")]
    /// Error processing the document for embedding
    #[error("DocumentError: {0}")]
    DocumentError(Box<dyn std::error::Error + 'static>),

    /// Error parsing the completion response
    #[error(transparent)]
    ResponseError(EmbeddingResponseError),

    /// Error returned by the embedding model provider
    #[error(transparent)]
    ProviderError(EmbeddingProviderError),

    /// The embedding backend or local runtime could not be initialized.
    #[error(transparent)]
    InitializationError(EmbeddingInitializationError),

    /// The embedding backend returned no embeddings for a request that should produce one.
    #[error("EmptyResponse: embedding backend returned no embeddings")]
    EmptyResponse,

    /// The embedding backend omitted the embedding for a specific document.
    #[error("MissingEmbeddingForDocument: backend omitted embedding for document index {index}")]
    MissingEmbeddingForDocument { index: usize },
}

impl EmbeddingError {
    pub fn response(message: impl Into<String>) -> Self {
        Self::ResponseError(EmbeddingResponseError::from_message(message))
    }

    pub fn provider(message: impl Into<String>) -> Self {
        Self::ProviderError(EmbeddingProviderError::from_message(message))
    }

    pub fn initialization(message: impl Into<String>) -> Self {
        Self::InitializationError(EmbeddingInitializationError::from_message(message))
    }
}

/// Trait for embedding models that can generate embeddings for documents.
pub trait EmbeddingModel: WasmCompatSend + WasmCompatSync {
    /// The maximum number of documents that can be embedded in a single request.
    const MAX_DOCUMENTS: usize;

    type Client;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self;

    /// The number of dimensions in the embedding vector.
    fn ndims(&self) -> usize;

    /// Embed multiple text documents in a single request
    fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + WasmCompatSend;

    /// Embed a single text document.
    fn embed_text(
        &self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<Embedding, EmbeddingError>> + WasmCompatSend {
        async {
            self.embed_texts(vec![text.to_string()])
                .await?
                .pop()
                .ok_or(EmbeddingError::EmptyResponse)
        }
    }
}

/// Trait for embedding models that can generate embeddings for images.
pub trait ImageEmbeddingModel: Clone + WasmCompatSend + WasmCompatSync {
    /// The maximum number of images that can be embedded in a single request.
    const MAX_DOCUMENTS: usize;

    /// The number of dimensions in the embedding vector.
    fn ndims(&self) -> usize;

    /// Embed multiple images in a single request from bytes.
    fn embed_images(
        &self,
        images: impl IntoIterator<Item = Vec<u8>> + WasmCompatSend,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + Send;

    /// Embed a single image from bytes.
    fn embed_image<'a>(
        &'a self,
        bytes: &'a [u8],
    ) -> impl std::future::Future<Output = Result<Embedding, EmbeddingError>> + WasmCompatSend {
        async move {
            self.embed_images(vec![bytes.to_owned()])
                .await?
                .pop()
                .ok_or(EmbeddingError::EmptyResponse)
        }
    }
}

/// Struct that holds a single document and its embedding.
#[derive(Clone, Default, Deserialize, Serialize, Debug)]
pub struct Embedding {
    /// The document that was embedded. Used for debugging.
    pub document: String,
    /// The embedding vector
    pub vec: Vec<f64>,
}

impl PartialEq for Embedding {
    fn eq(&self, other: &Self) -> bool {
        self.document == other.document
    }
}

impl Eq for Embedding {}

#[cfg(test)]
mod tests {
    use super::{
        EmbeddingError, EmbeddingInitializationError, EmbeddingProviderError,
        EmbeddingResponseError,
    };

    #[test]
    fn embedding_response_error_maps_count_mismatches() {
        assert!(matches!(
            EmbeddingError::response("Response data length does not match input length"),
            EmbeddingError::ResponseError(EmbeddingResponseError::MismatchedEmbeddingCount)
        ));
        assert!(matches!(
            EmbeddingError::response("Number of returned embeddings does not match input"),
            EmbeddingError::ResponseError(EmbeddingResponseError::MismatchedEmbeddingCount)
        ));
    }

    #[test]
    fn embedding_error_preserves_unclassified_messages() {
        assert!(matches!(
            EmbeddingError::response("embedding payload missing vector"),
            EmbeddingError::ResponseError(EmbeddingResponseError::Message { message })
                if message == "embedding payload missing vector"
        ));
        assert!(matches!(
            EmbeddingError::provider("provider throttled request"),
            EmbeddingError::ProviderError(EmbeddingProviderError::Message { message })
                if message == "provider throttled request"
        ));
        assert!(matches!(
            EmbeddingError::initialization("tokenizer failed to load"),
            EmbeddingError::InitializationError(EmbeddingInitializationError::Message {
                message
            }) if message == "tokenizer failed to load"
        ));
    }
}
