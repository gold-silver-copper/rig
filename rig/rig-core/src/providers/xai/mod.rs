//! xAI API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::xai;
//!
//! let client = rig::providers::xai::Client::new("YOUR_API_KEY");
//!
//! let grok = client.completion_model(rig::models::xai::GROK_3);
//! ```

mod api;
#[cfg(feature = "audio")]
pub mod audio_generation;
pub mod client;
pub mod completion;
#[cfg(feature = "image")]
pub mod image_generation;
mod streaming;

#[cfg(feature = "audio")]
pub use audio_generation::AudioGenerationModel;
pub use client::Client;
pub use completion::{CompletionModel, CompletionResponse};
#[cfg(feature = "image")]
pub use image_generation::ImageGenerationModel;
