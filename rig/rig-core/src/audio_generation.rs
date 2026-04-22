//! Everything related to audio generation (ie, Text To Speech).
//! Rig abstracts over a number of different providers using the [AudioGenerationModel] trait.
use crate::markers::{Missing, Provided};
use crate::{
    http_client,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use http::StatusCode;
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AudioGenerationResponseError {
    #[error("ResponseError: failed to decode {provider} audio payload: {source}")]
    Base64Decode {
        provider: &'static str,
        #[source]
        source: base64::DecodeError,
    },

    #[error("ResponseError: {message}")]
    Message { message: String },
}

#[derive(Debug, Error)]
pub enum AudioGenerationTransportError {
    #[error("TransportError: request failed with status {status}: {body}")]
    Status { status: StatusCode, body: String },

    #[error("TransportError: {message}")]
    Message { message: String },
}

impl AudioGenerationTransportError {
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message {
            message: message.into(),
        }
    }
}

#[derive(Debug, Error)]
pub enum AudioGenerationConfigurationError {
    #[error("ConfigurationError: {message}")]
    Message { message: String },
}

#[derive(Debug, Error)]
pub enum AudioGenerationError {
    /// Http error (e.g.: connection error, timeout, etc.)
    #[error("HttpError: {0}")]
    HttpError(#[from] http_client::Error),

    /// Json error (e.g.: serialization, deserialization)
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    #[cfg(not(target_family = "wasm"))]
    /// Error building the transcription request
    #[error("RequestError: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    #[cfg(target_family = "wasm")]
    /// Error building the transcription request
    #[error("RequestError: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + 'static>),

    /// Error parsing the transcription response
    #[error(transparent)]
    ResponseError(AudioGenerationResponseError),

    /// Error returned while talking to the audio generation provider.
    #[error(transparent)]
    TransportError(AudioGenerationTransportError),

    /// Error caused by local audio generation configuration or initialization.
    #[error(transparent)]
    ConfigurationError(AudioGenerationConfigurationError),
}

impl AudioGenerationError {
    pub fn request(message: impl Into<String>) -> Self {
        Self::RequestError(Box::new(std::io::Error::other(message.into())))
    }

    pub fn decode_payload(provider: &'static str, source: base64::DecodeError) -> Self {
        Self::ResponseError(AudioGenerationResponseError::Base64Decode { provider, source })
    }

    pub fn response_message(message: impl Into<String>) -> Self {
        Self::ResponseError(AudioGenerationResponseError::Message {
            message: message.into(),
        })
    }

    pub fn transport(message: impl Into<String>) -> Self {
        Self::TransportError(AudioGenerationTransportError::message(message))
    }

    pub fn transport_status(status: StatusCode, body: impl Into<String>) -> Self {
        Self::TransportError(AudioGenerationTransportError::Status {
            status,
            body: body.into(),
        })
    }

    pub fn configuration(message: impl Into<String>) -> Self {
        Self::ConfigurationError(AudioGenerationConfigurationError::Message {
            message: message.into(),
        })
    }
}
pub trait AudioGeneration<M>
where
    M: AudioGenerationModel,
{
    /// Generates an audio generation request builder for the given `text` and `voice`.
    /// This function is meant to be called by the user to further customize the
    /// request at generation time before sending it.
    ///
    /// ❗IMPORTANT: The type that implements this trait might have already
    /// populated fields in the builder (the exact fields depend on the type).
    /// For fields that have already been set by the model, calling the corresponding
    /// method on the builder will overwrite the value set by the model.
    fn audio_generation(
        &self,
        text: &str,
        voice: &str,
    ) -> impl std::future::Future<
        Output = Result<
            AudioGenerationRequestBuilder<M, Provided<String>, Provided<String>>,
            AudioGenerationError,
        >,
    > + WasmCompatSend;
}

pub struct AudioGenerationResponse<T> {
    pub audio: Vec<u8>,
    pub response: T,
}

pub trait AudioGenerationModel: Sized + Clone + WasmCompatSend + WasmCompatSync {
    type Response: WasmCompatSend + WasmCompatSync;

    type Client;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self;

    fn audio_generation(
        &self,
        request: AudioGenerationRequest,
    ) -> impl std::future::Future<
        Output = Result<AudioGenerationResponse<Self::Response>, AudioGenerationError>,
    > + WasmCompatSend;

    fn audio_generation_request(&self) -> AudioGenerationRequestBuilder<Self, Missing, Missing> {
        AudioGenerationRequestBuilder::new(self.clone())
    }
}
#[non_exhaustive]
pub struct AudioGenerationRequest {
    pub text: String,
    pub voice: String,
    pub speed: f32,
    pub additional_params: Option<Value>,
}

#[non_exhaustive]
pub struct AudioGenerationRequestBuilder<M, T = Missing, V = Missing>
where
    M: AudioGenerationModel,
{
    model: M,
    text: T,
    voice: V,
    speed: f32,
    additional_params: Option<Value>,
}

impl<M> AudioGenerationRequestBuilder<M, Missing, Missing>
where
    M: AudioGenerationModel,
{
    pub fn new(model: M) -> Self {
        Self {
            model,
            text: Missing,
            voice: Missing,
            speed: 1.0,
            additional_params: None,
        }
    }
}

impl<M, T, V> AudioGenerationRequestBuilder<M, T, V>
where
    M: AudioGenerationModel,
{
    /// Sets the text for the audio generation request
    pub fn text(self, text: &str) -> AudioGenerationRequestBuilder<M, Provided<String>, V> {
        AudioGenerationRequestBuilder {
            model: self.model,
            text: Provided(text.to_string()),
            voice: self.voice,
            speed: self.speed,
            additional_params: self.additional_params,
        }
    }

    /// The voice of the generated audio
    pub fn voice(self, voice: &str) -> AudioGenerationRequestBuilder<M, T, Provided<String>> {
        AudioGenerationRequestBuilder {
            model: self.model,
            text: self.text,
            voice: Provided(voice.to_string()),
            speed: self.speed,
            additional_params: self.additional_params,
        }
    }

    /// The speed of the generated audio
    pub fn speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    /// Adds additional parameters to the audio generation request.
    pub fn additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }
}

impl<M> AudioGenerationRequestBuilder<M, Provided<String>, Provided<String>>
where
    M: AudioGenerationModel,
{
    pub fn build(self) -> AudioGenerationRequest {
        AudioGenerationRequest {
            text: self.text.0,
            voice: self.voice.0,
            speed: self.speed,
            additional_params: self.additional_params,
        }
    }

    pub async fn send(self) -> Result<AudioGenerationResponse<M::Response>, AudioGenerationError> {
        let model = self.model.clone();

        model.audio_generation(self.build()).await
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AudioGenerationConfigurationError, AudioGenerationError, AudioGenerationTransportError,
    };
    use http::StatusCode;

    #[test]
    fn audio_generation_error_taxonomy_is_explicit() {
        assert!(matches!(
            AudioGenerationError::transport_status(StatusCode::BAD_GATEWAY, "bad gateway"),
            AudioGenerationError::TransportError(AudioGenerationTransportError::Status {
                status,
                body,
            }) if status == StatusCode::BAD_GATEWAY && body == "bad gateway"
        ));
        assert!(matches!(
            AudioGenerationError::configuration("missing audio backend"),
            AudioGenerationError::ConfigurationError(
                AudioGenerationConfigurationError::Message { message }
            ) if message == "missing audio backend"
        ));
    }
}
