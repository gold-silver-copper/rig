//! Everything related to core image generation abstractions in Rig.
//! Rig allows calling a number of different providers (that support image generation) using the [ImageGenerationModel] trait.
use crate::http_client;
use crate::markers::{Missing, Provided};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use http::StatusCode;
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ImageGenerationResponseError {
    #[error("ResponseError: {provider} image response contained no images")]
    MissingImages { provider: &'static str },

    #[error("ResponseError: failed to decode {provider} image payload: {source}")]
    Base64Decode {
        provider: &'static str,
        #[source]
        source: base64::DecodeError,
    },

    #[error("ResponseError: {message}")]
    Message { message: String },
}

#[derive(Debug, Error)]
pub enum ImageGenerationProviderError {
    #[error("ProviderError: image generation endpoint is not supported yet for {provider}")]
    UnsupportedEndpoint { provider: String },

    #[error("ProviderError: request failed with status {status}: {body}")]
    Status { status: StatusCode, body: String },

    #[error("ProviderError: {message}")]
    Message { message: String },
}

impl ImageGenerationProviderError {
    pub fn from_message(message: impl Into<String>) -> Self {
        Self::Message {
            message: message.into(),
        }
    }
}

impl From<String> for ImageGenerationResponseError {
    fn from(message: String) -> Self {
        Self::Message { message }
    }
}

impl From<&str> for ImageGenerationResponseError {
    fn from(message: &str) -> Self {
        Self::Message {
            message: message.to_owned(),
        }
    }
}

impl From<String> for ImageGenerationProviderError {
    fn from(message: String) -> Self {
        Self::Message { message }
    }
}

impl From<&str> for ImageGenerationProviderError {
    fn from(message: &str) -> Self {
        Self::Message {
            message: message.to_owned(),
        }
    }
}

#[derive(Debug, Error)]
pub enum ImageGenerationError {
    /// Http error (e.g.: connection error, timeout, etc.)
    #[error("HttpError: {0}")]
    HttpError(#[from] http_client::Error),

    /// Json error (e.g.: serialization, deserialization)
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Error building the transcription request
    #[error("RequestError: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    /// Error parsing the transcription response
    #[error(transparent)]
    ResponseError(ImageGenerationResponseError),

    /// Error returned by the transcription model provider
    #[error(transparent)]
    ProviderError(ImageGenerationProviderError),
}

impl ImageGenerationError {
    pub fn missing_images(provider: &'static str) -> Self {
        Self::ResponseError(ImageGenerationResponseError::MissingImages { provider })
    }

    pub fn decode_payload(provider: &'static str, source: base64::DecodeError) -> Self {
        Self::ResponseError(ImageGenerationResponseError::Base64Decode { provider, source })
    }

    pub fn response_message(message: impl Into<String>) -> Self {
        Self::ResponseError(ImageGenerationResponseError::Message {
            message: message.into(),
        })
    }

    pub fn provider(message: impl Into<String>) -> Self {
        Self::ProviderError(ImageGenerationProviderError::from_message(message))
    }

    pub fn provider_status(status: StatusCode, body: impl Into<String>) -> Self {
        Self::ProviderError(ImageGenerationProviderError::Status {
            status,
            body: body.into(),
        })
    }

    pub fn unsupported_endpoint(provider: impl Into<String>) -> Self {
        Self::ProviderError(ImageGenerationProviderError::UnsupportedEndpoint {
            provider: provider.into(),
        })
    }
}
pub trait ImageGeneration<M>
where
    M: ImageGenerationModel,
{
    /// Generates a transcription request builder for the given `file`.
    /// This function is meant to be called by the user to further customize the
    /// request at transcription time before sending it.
    ///
    /// ❗IMPORTANT: The type that implements this trait might have already
    /// populated fields in the builder (the exact fields depend on the type).
    /// For fields that have already been set by the model, calling the corresponding
    /// method on the builder will overwrite the value set by the model.
    fn image_generation(
        &self,
        prompt: &str,
        size: &(u32, u32),
    ) -> impl std::future::Future<
        Output = Result<ImageGenerationRequestBuilder<M, Provided<String>>, ImageGenerationError>,
    > + WasmCompatSend;
}

/// A unified response for a model image generation, returning both the image and the raw response.
#[derive(Debug)]
pub struct ImageGenerationResponse<T> {
    pub image: Vec<u8>,
    pub response: T,
}

pub trait ImageGenerationModel: Clone + WasmCompatSend + WasmCompatSync {
    type Response: WasmCompatSend + WasmCompatSync;

    type Client;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self;

    fn image_generation(
        &self,
        request: ImageGenerationRequest,
    ) -> impl std::future::Future<
        Output = Result<ImageGenerationResponse<Self::Response>, ImageGenerationError>,
    > + WasmCompatSend;

    fn image_generation_request(&self) -> ImageGenerationRequestBuilder<Self, Missing> {
        ImageGenerationRequestBuilder::new(self.clone())
    }
}
/// An image generation request.
#[non_exhaustive]
pub struct ImageGenerationRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub additional_params: Option<Value>,
}

/// A builder for `ImageGenerationRequest`.
/// Can be sent to a model provider.
#[non_exhaustive]
pub struct ImageGenerationRequestBuilder<M, P = Missing>
where
    M: ImageGenerationModel,
{
    model: M,
    prompt: P,
    width: u32,
    height: u32,
    additional_params: Option<Value>,
}

impl<M> ImageGenerationRequestBuilder<M, Missing>
where
    M: ImageGenerationModel,
{
    pub fn new(model: M) -> Self {
        Self {
            model,
            prompt: Missing,
            height: 256,
            width: 256,
            additional_params: None,
        }
    }
}

impl<M, P> ImageGenerationRequestBuilder<M, P>
where
    M: ImageGenerationModel,
{
    /// Sets the prompt for the image generation request
    pub fn prompt(self, prompt: &str) -> ImageGenerationRequestBuilder<M, Provided<String>> {
        ImageGenerationRequestBuilder {
            model: self.model,
            prompt: Provided(prompt.to_string()),
            width: self.width,
            height: self.height,
            additional_params: self.additional_params,
        }
    }

    /// The width of the generated image
    pub fn width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    /// The height of the generated image
    pub fn height(mut self, height: u32) -> Self {
        self.height = height;
        self
    }

    /// Adds additional parameters to the image generation request.
    pub fn additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }
}

impl<M> ImageGenerationRequestBuilder<M, Provided<String>>
where
    M: ImageGenerationModel,
{
    pub fn build(self) -> ImageGenerationRequest {
        ImageGenerationRequest {
            prompt: self.prompt.0,
            width: self.width,
            height: self.height,
            additional_params: self.additional_params,
        }
    }

    pub async fn send(self) -> Result<ImageGenerationResponse<M::Response>, ImageGenerationError> {
        let model = self.model.clone();

        model.image_generation(self.build()).await
    }
}

#[cfg(test)]
mod tests {
    use super::{ImageGenerationError, ImageGenerationProviderError, ImageGenerationResponseError};
    use http::StatusCode;

    #[test]
    fn image_generation_error_builders_preserve_structure() {
        assert!(matches!(
            ImageGenerationError::missing_images("OpenAI"),
            ImageGenerationError::ResponseError(ImageGenerationResponseError::MissingImages {
                provider: "OpenAI"
            })
        ));
        assert!(matches!(
            ImageGenerationError::provider_status(StatusCode::BAD_GATEWAY, "bad gateway"),
            ImageGenerationError::ProviderError(ImageGenerationProviderError::Status {
                status,
                body,
            }) if status == StatusCode::BAD_GATEWAY && body == "bad gateway"
        ));
        assert!(matches!(
            ImageGenerationError::unsupported_endpoint("custom-subprovider"),
            ImageGenerationError::ProviderError(
                ImageGenerationProviderError::UnsupportedEndpoint { provider }
            ) if provider == "custom-subprovider"
        ));
    }
}
