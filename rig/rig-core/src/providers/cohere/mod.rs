//! Cohere API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::cohere;
//!
//! let client = rig::providers::cohere::Client::new("YOUR_API_KEY");
//!
//! let command_r = client.completion_model(rig::models::cohere::COMMAND_R);
//! ```

pub mod client;
pub mod completion;
pub mod embeddings;
pub mod streaming;

pub use client::{ApiErrorResponse, ApiResponse, Client};
pub use completion::CompletionModel;
pub use embeddings::EmbeddingModel;

// ================================================================
// Cohere Completion Models
// ================================================================

// ================================================================
// Cohere Embedding Models
// ================================================================

pub(crate) fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
    crate::models::cohere::lookup(identifier)
        .and_then(|model| model.embedding)
        .and_then(|metadata| metadata.default_dimensions)
}
