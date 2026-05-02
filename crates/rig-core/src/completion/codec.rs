//! Provider codec abstractions for completion requests and model output.
//!
//! Codecs isolate provider wire formats from Rig's normalized completion
//! model. Request codecs encode Rig requests into provider requests, response
//! codecs decode non-streaming provider responses into a [`ModelTurn`], and
//! stream codecs decode provider chunks while keeping provider-specific stream
//! state out of the agent runner.

use std::pin::Pin;

use futures::Stream;

use crate::{
    OneOrMany,
    completion::{AssistantContent, CompletionError, CompletionRequest, CompletionResponse, Usage},
    model_event::{CompletionCollector, ModelEvent, ModelEventStream},
    wasm_compat::WasmCompatSend,
};

/// A fallible stream of normalized model output events.
#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
pub type ModelEventResultStream<R> =
    Pin<Box<dyn Stream<Item = Result<ModelEvent<R>, CompletionError>> + Send>>;

/// A fallible stream of normalized model output events.
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub type ModelEventResultStream<R> =
    Pin<Box<dyn Stream<Item = Result<ModelEvent<R>, CompletionError>>>>;

/// A complete normalized model turn.
#[derive(Debug, Clone)]
pub struct ModelTurn<R> {
    /// Ordered assistant content emitted by the model.
    pub choice: OneOrMany<AssistantContent>,
    /// Normalized token usage for the turn.
    pub usage: Usage,
    /// Raw provider final response, when available.
    pub raw_response: Option<R>,
    /// Provider-assigned assistant message ID, when available.
    pub message_id: Option<String>,
    /// Provider finish reason normalized to text when available.
    pub finish_reason: Option<String>,
    /// Provider metadata that should stay observable but not interpreted by the agent loop.
    pub provider_metadata: Vec<serde_json::Value>,
}

impl<R> ModelTurn<R> {
    /// Creates a normalized model turn.
    pub fn new(choice: OneOrMany<AssistantContent>, usage: Usage) -> Self {
        Self {
            choice,
            usage,
            raw_response: None,
            message_id: None,
            finish_reason: None,
            provider_metadata: Vec::new(),
        }
    }

    /// Creates a normalized model turn from an existing completion response.
    pub fn from_completion_response(response: CompletionResponse<R>) -> Self {
        Self {
            choice: response.choice,
            usage: response.usage,
            raw_response: Some(response.raw_response),
            message_id: response.message_id,
            finish_reason: None,
            provider_metadata: Vec::new(),
        }
    }

    /// Sets the raw provider response.
    pub fn with_raw_response(mut self, raw_response: R) -> Self {
        self.raw_response = Some(raw_response);
        self
    }

    /// Sets the provider-assigned message ID.
    pub fn with_message_id(mut self, message_id: Option<String>) -> Self {
        self.message_id = message_id;
        self
    }

    /// Sets the provider finish reason.
    pub fn with_finish_reason(mut self, finish_reason: Option<String>) -> Self {
        self.finish_reason = finish_reason;
        self
    }

    /// Adds a provider metadata value.
    pub fn with_provider_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.provider_metadata.push(metadata);
        self
    }

    /// Returns the concatenated visible assistant text for this turn.
    pub fn visible_text(&self) -> String {
        self.choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<String>()
    }

    /// Converts this turn into Rig's legacy completion response shape.
    pub fn into_completion_response(self) -> Result<CompletionResponse<R>, CompletionError> {
        let raw_response = self.raw_response.ok_or_else(|| {
            CompletionError::ResponseError(
                "model turn completed without a raw provider response".to_string(),
            )
        })?;

        Ok(CompletionResponse {
            choice: self.choice,
            usage: self.usage,
            raw_response,
            message_id: self.message_id,
        })
    }

    /// Converts this turn into normalized model events.
    pub fn into_events(self) -> Result<ModelEventStream<R>, CompletionError>
    where
        R: WasmCompatSend + 'static,
    {
        let raw_response = self.raw_response.ok_or_else(|| {
            CompletionError::ResponseError(
                "model turn completed without a raw provider response".to_string(),
            )
        })?;

        Ok(crate::model_event::events_from_parts(
            raw_response,
            self.choice,
            self.usage,
            self.message_id,
        ))
    }
}

/// Collects normalized model events into a [`ModelTurn`].
#[derive(Debug)]
pub struct ModelTurnAccumulator<R> {
    collector: CompletionCollector<R>,
    finish_reason: Option<String>,
    provider_metadata: Vec<serde_json::Value>,
}

/// Creates normalized model events from assistant content and raw provider data.
pub fn events_from_parts<R, I>(
    raw_response: R,
    content: I,
    usage: Usage,
    message_id: Option<String>,
) -> ModelEventStream<R>
where
    R: WasmCompatSend + 'static,
    I: IntoIterator<Item = AssistantContent> + WasmCompatSend + 'static,
    <I as IntoIterator>::IntoIter: WasmCompatSend + 'static,
{
    let choice = OneOrMany::from_iter_optional(content)
        .unwrap_or_else(|| OneOrMany::one(AssistantContent::text("")));

    crate::model_event::events_from_parts(raw_response, choice, usage, message_id)
}

/// Creates a normalized model turn from assistant content and raw provider data.
pub fn turn_from_parts<R, I>(
    raw_response: R,
    content: I,
    usage: Usage,
    message_id: Option<String>,
) -> Result<ModelTurn<R>, CompletionError>
where
    I: IntoIterator<Item = AssistantContent>,
{
    let choice = OneOrMany::from_iter_optional(content).ok_or_else(|| {
        CompletionError::ResponseError(
            "Response contained no message or tool call (empty)".to_owned(),
        )
    })?;

    Ok(ModelTurn::new(choice, usage)
        .with_message_id(message_id)
        .with_raw_response(raw_response))
}

/// Decodes a provider response through a codec and returns normalized events.
pub fn response_events<C>(
    codec: &C,
    response: C::Response,
) -> Result<ModelEventStream<C::RawFinal>, CompletionError>
where
    C: CompletionResponseCodec,
    C::RawFinal: WasmCompatSend + 'static,
{
    codec.decode_response(response)?.into_events()
}

/// Converts a stream of fallible model events into the legacy infallible event stream.
pub fn result_stream<R, S>(stream: S) -> ModelEventStream<R>
where
    R: WasmCompatSend + 'static,
    S: Stream<Item = Result<ModelEvent<R>, CompletionError>> + WasmCompatSend + 'static,
{
    crate::model_event::result_stream(stream)
}

impl<R> Default for ModelTurnAccumulator<R> {
    fn default() -> Self {
        Self {
            collector: CompletionCollector::new(),
            finish_reason: None,
            provider_metadata: Vec::new(),
        }
    }
}

impl<R> ModelTurnAccumulator<R> {
    /// Creates an empty model turn accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the finish reason that will be attached to the completed turn.
    pub fn set_finish_reason(&mut self, finish_reason: impl Into<String>) {
        self.finish_reason = Some(finish_reason.into());
    }

    /// Applies one normalized event to this accumulator.
    pub fn push(&mut self, event: ModelEvent<R>) -> Result<(), CompletionError> {
        if let ModelEvent::ProviderMetadata { metadata } = &event {
            self.provider_metadata.push(metadata.clone());
        }

        self.collector.push(event)
    }

    /// Applies a borrowed normalized event to this accumulator.
    pub fn push_ref(&mut self, event: &ModelEvent<R>) -> Result<(), CompletionError>
    where
        R: Clone,
    {
        match event {
            ModelEvent::TextDelta { text } => {
                self.push(ModelEvent::TextDelta { text: text.clone() })
            }
            ModelEvent::ReasoningDelta { id, text } => self.push(ModelEvent::ReasoningDelta {
                id: id.clone(),
                text: text.clone(),
            }),
            ModelEvent::ReasoningDone { reasoning } => self.push(ModelEvent::ReasoningDone {
                reasoning: reasoning.clone(),
            }),
            ModelEvent::ToolCallStart {
                id,
                internal_call_id,
                call_id,
                name,
            } => self.push(ModelEvent::ToolCallStart {
                id: id.clone(),
                internal_call_id: internal_call_id.clone(),
                call_id: call_id.clone(),
                name: name.clone(),
            }),
            ModelEvent::ToolCallDelta {
                id,
                internal_call_id,
                content,
            } => self.push(ModelEvent::ToolCallDelta {
                id: id.clone(),
                internal_call_id: internal_call_id.clone(),
                content: content.clone(),
            }),
            ModelEvent::ToolCallDone {
                tool_call,
                internal_call_id,
            } => self.push(ModelEvent::ToolCallDone {
                tool_call: tool_call.clone(),
                internal_call_id: internal_call_id.clone(),
            }),
            ModelEvent::MessageDone { id } => self.push(ModelEvent::MessageDone { id: id.clone() }),
            ModelEvent::Usage { usage } => self.push(ModelEvent::Usage { usage: *usage }),
            ModelEvent::ProviderMetadata { metadata } => self.push(ModelEvent::ProviderMetadata {
                metadata: metadata.clone(),
            }),
            ModelEvent::RawResponse { response } => self.push(ModelEvent::RawResponse {
                response: response.clone(),
            }),
            ModelEvent::Error { error } => Err(CompletionError::ResponseError(error.to_string())),
            ModelEvent::Done => self.push(ModelEvent::Done),
        }
    }

    /// Finishes collection without requiring raw provider data.
    pub fn finish_optional(self) -> Result<ModelTurn<R>, CompletionError> {
        let response = self.collector.finish_optional()?;
        Ok(ModelTurn {
            choice: response.choice,
            usage: response.usage,
            raw_response: response.raw_response,
            message_id: response.message_id,
            finish_reason: self.finish_reason,
            provider_metadata: self.provider_metadata,
        })
    }

    /// Finishes collection and requires raw provider data.
    pub fn finish(self) -> Result<ModelTurn<R>, CompletionError> {
        let turn = self.finish_optional()?;
        if turn.raw_response.is_none() {
            return Err(CompletionError::ResponseError(
                "model turn completed without a raw provider response".to_string(),
            ));
        }

        Ok(turn)
    }
}

/// Encodes Rig completion requests into provider request payloads.
pub trait CompletionCodec {
    /// Provider request payload type.
    type Request;

    /// Encodes a normalized Rig request into the provider request shape.
    fn encode_request(&self, request: CompletionRequest) -> Result<Self::Request, CompletionError>;
}

/// Decodes non-streaming provider responses into normalized turns.
pub trait CompletionResponseCodec {
    /// Provider response payload type.
    type Response;
    /// Raw final response retained by the normalized turn.
    type RawFinal;

    /// Decodes a provider response into a normalized model turn.
    fn decode_response(
        &self,
        response: Self::Response,
    ) -> Result<ModelTurn<Self::RawFinal>, CompletionError>;
}

/// Decodes provider streaming chunks into normalized events and a final turn.
pub trait CompletionStreamCodec {
    /// Provider stream chunk payload type.
    type StreamChunk;
    /// Raw final response retained by the normalized turn.
    type RawFinal;

    /// Decodes one provider stream chunk into zero or more normalized events.
    fn decode_stream_chunk(
        &mut self,
        chunk: Self::StreamChunk,
    ) -> Result<Vec<ModelEvent<Self::RawFinal>>, CompletionError>;

    /// Finalizes stream state into a normalized model turn.
    fn finish_stream(self) -> Result<ModelTurn<Self::RawFinal>, CompletionError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_text_concatenates_only_text_content() {
        let choice = OneOrMany::many(vec![
            AssistantContent::text("hello "),
            AssistantContent::reasoning("hidden"),
            AssistantContent::text("world"),
        ])
        .expect("choice should be non-empty");

        let turn = ModelTurn::<()>::new(choice, Usage::new());

        assert_eq!(turn.visible_text(), "hello world");
    }

    #[test]
    fn accumulator_preserves_provider_metadata() {
        let mut accumulator = ModelTurnAccumulator::<()>::new();
        accumulator
            .push(ModelEvent::ProviderMetadata {
                metadata: serde_json::json!({"provider": "test"}),
            })
            .expect("metadata should collect");

        let turn = accumulator
            .finish_optional()
            .expect("turn should finish without raw response");

        assert_eq!(turn.provider_metadata.len(), 1);
        assert_eq!(turn.provider_metadata[0]["provider"], "test");
    }
}
