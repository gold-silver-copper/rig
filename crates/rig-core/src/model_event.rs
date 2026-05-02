//! Event-first completion model output.
//!
//! [`ModelEvent`] is the canonical representation of provider/model output. A
//! complete non-streaming response is produced by collecting these events with
//! [`CompletionCollector`].

use std::{collections::HashMap, pin::Pin};

use async_stream::stream;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::{
    OneOrMany,
    completion::{CompletionError, CompletionResponse, Usage},
    message::{AssistantContent, Reasoning, ReasoningContent, ToolCall, ToolFunction},
    wasm_compat::WasmCompatSend,
};

/// A boxed stream of normalized model events.
#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
pub type ModelEventStream<R> = Pin<Box<dyn Stream<Item = ModelEvent<R>> + Send>>;

/// A boxed stream of normalized model events.
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub type ModelEventStream<R> = Pin<Box<dyn Stream<Item = ModelEvent<R>>>>;

/// The content of a streamed tool call delta.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum ToolCallDeltaContent {
    /// A full or partial function/tool name.
    Name(String),
    /// A partial function/tool argument payload.
    Delta(String),
}

/// A normalized provider/model output event.
#[derive(Debug)]
pub enum ModelEvent<R> {
    /// A delta of visible assistant text.
    TextDelta { text: String },
    /// A delta of model reasoning text.
    ReasoningDelta { id: Option<String>, text: String },
    /// A complete reasoning block.
    ReasoningDone { reasoning: Reasoning },
    /// A tool call has started, but may not have complete arguments yet.
    ToolCallStart {
        id: String,
        internal_call_id: String,
        call_id: Option<String>,
        name: String,
    },
    /// A delta for a tool call name or argument payload.
    ToolCallDelta {
        id: String,
        internal_call_id: String,
        content: ToolCallDeltaContent,
    },
    /// A complete tool call.
    ToolCallDone {
        tool_call: ToolCall,
        internal_call_id: Option<String>,
    },
    /// The provider completed an assistant message.
    MessageDone { id: Option<String> },
    /// Token usage for this model turn. Multiple usage events are treated as
    /// replacements by the collector; providers should emit cumulative usage.
    Usage { usage: Usage },
    /// Provider-specific metadata that should be observable but not interpreted
    /// by Rig's core collector.
    ProviderMetadata { metadata: serde_json::Value },
    /// The raw provider response, when available.
    RawResponse { response: R },
    /// A provider/model error represented inside the event stream.
    Error { error: CompletionError },
    /// The model event stream has completed.
    Done,
}

impl<R> ModelEvent<R> {
    /// Returns the embedded error when this event is [`ModelEvent::Error`].
    pub fn into_error(self) -> Option<CompletionError> {
        match self {
            Self::Error { error } => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct PendingToolCall {
    id: String,
    call_id: Option<String>,
    name: String,
    arguments: String,
}

/// Collects [`ModelEvent`] values into a final [`CompletionResponse`].
#[derive(Debug)]
pub struct CompletionCollector<R> {
    assistant_items: Vec<AssistantContent>,
    text_item_index: Option<usize>,
    reasoning_item_index: Option<usize>,
    pending_tool_calls: HashMap<String, PendingToolCall>,
    usage: Usage,
    raw_response: Option<R>,
    message_id: Option<String>,
}

impl<R> Default for CompletionCollector<R> {
    fn default() -> Self {
        Self {
            assistant_items: Vec::new(),
            text_item_index: None,
            reasoning_item_index: None,
            pending_tool_calls: HashMap::new(),
            usage: Usage::new(),
            raw_response: None,
            message_id: None,
        }
    }
}

impl<R> CompletionCollector<R> {
    /// Creates an empty collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies one model event to the collector.
    pub fn push(&mut self, event: ModelEvent<R>) -> Result<(), CompletionError> {
        match event {
            ModelEvent::TextDelta { text } => self.append_text_chunk(&text),
            ModelEvent::ReasoningDelta { id, text } => self.append_reasoning_chunk(&id, &text),
            ModelEvent::ReasoningDone { reasoning } => self.append_reasoning(reasoning),
            ModelEvent::ToolCallStart {
                id,
                internal_call_id,
                call_id,
                name,
            } => self.start_tool_call(id, internal_call_id, call_id, name),
            ModelEvent::ToolCallDelta {
                id,
                internal_call_id,
                content,
            } => self.append_tool_call_delta(id, internal_call_id, content),
            ModelEvent::ToolCallDone { tool_call, .. } => self.append_tool_call(tool_call),
            ModelEvent::MessageDone { id } => {
                if id.is_some() {
                    self.message_id = id;
                }
            }
            ModelEvent::Usage { usage } => self.usage = usage,
            ModelEvent::ProviderMetadata { .. } => {}
            ModelEvent::RawResponse { response } => self.raw_response = Some(response),
            ModelEvent::Error { error } => return Err(error),
            ModelEvent::Done => self.flush_pending_tool_calls()?,
        }

        Ok(())
    }

    /// Finishes collection and returns a response with optional raw provider data.
    pub fn finish_optional(mut self) -> Result<CompletionResponse<Option<R>>, CompletionError> {
        self.flush_pending_tool_calls()?;

        if self.assistant_items.is_empty() {
            self.assistant_items.push(AssistantContent::text(""));
        }

        let choice = OneOrMany::from_iter_optional(self.assistant_items)
            .unwrap_or_else(|| OneOrMany::one(AssistantContent::text("")));

        Ok(CompletionResponse {
            choice,
            usage: self.usage,
            raw_response: self.raw_response,
            message_id: self.message_id,
        })
    }

    /// Finishes collection and requires a raw provider response to be present.
    pub fn finish(self) -> Result<CompletionResponse<R>, CompletionError> {
        let response = self.finish_optional()?;
        let raw_response = response.raw_response.ok_or_else(|| {
            CompletionError::ResponseError(
                "model event stream completed without a raw provider response".to_string(),
            )
        })?;

        Ok(CompletionResponse {
            choice: response.choice,
            usage: response.usage,
            raw_response,
            message_id: response.message_id,
        })
    }

    fn append_text_chunk(&mut self, text: &str) {
        self.reasoning_item_index = None;

        if let Some(index) = self.text_item_index
            && let Some(AssistantContent::Text(existing_text)) = self.assistant_items.get_mut(index)
        {
            existing_text.text.push_str(text);
            return;
        }

        self.assistant_items
            .push(AssistantContent::text(text.to_owned()));
        self.text_item_index = Some(self.assistant_items.len() - 1);
    }

    fn append_reasoning_chunk(&mut self, id: &Option<String>, text: &str) {
        self.text_item_index = None;

        if let Some(index) = self.reasoning_item_index
            && let Some(AssistantContent::Reasoning(existing)) = self.assistant_items.get_mut(index)
            && let Some(ReasoningContent::Text {
                text: existing_text,
                ..
            }) = existing.content.last_mut()
        {
            existing_text.push_str(text);
            return;
        }

        self.assistant_items
            .push(AssistantContent::Reasoning(Reasoning {
                id: id.clone(),
                content: vec![ReasoningContent::Text {
                    text: text.to_string(),
                    signature: None,
                }],
            }));
        self.reasoning_item_index = Some(self.assistant_items.len() - 1);
    }

    fn append_reasoning(&mut self, reasoning: Reasoning) {
        self.text_item_index = None;
        self.reasoning_item_index = None;
        self.assistant_items
            .push(AssistantContent::Reasoning(reasoning));
    }

    fn start_tool_call(
        &mut self,
        id: String,
        internal_call_id: String,
        call_id: Option<String>,
        name: String,
    ) {
        self.text_item_index = None;
        self.reasoning_item_index = None;
        self.pending_tool_calls.insert(
            internal_call_id,
            PendingToolCall {
                id,
                call_id,
                name,
                arguments: String::new(),
            },
        );
    }

    fn append_tool_call_delta(
        &mut self,
        id: String,
        internal_call_id: String,
        content: ToolCallDeltaContent,
    ) {
        self.text_item_index = None;
        self.reasoning_item_index = None;

        let pending = self
            .pending_tool_calls
            .entry(internal_call_id)
            .or_insert_with(|| PendingToolCall {
                id,
                call_id: None,
                name: String::new(),
                arguments: String::new(),
            });

        match content {
            ToolCallDeltaContent::Name(name) => pending.name.push_str(&name),
            ToolCallDeltaContent::Delta(delta) => pending.arguments.push_str(&delta),
        }
    }

    fn append_tool_call(&mut self, tool_call: ToolCall) {
        self.text_item_index = None;
        self.reasoning_item_index = None;
        self.assistant_items
            .push(AssistantContent::ToolCall(tool_call));
    }

    fn flush_pending_tool_calls(&mut self) -> Result<(), CompletionError> {
        let pending = std::mem::take(&mut self.pending_tool_calls);
        for tool_call in pending.into_values() {
            let arguments = if tool_call.arguments.trim().is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::from_str(&tool_call.arguments).map_err(|err| {
                    CompletionError::ResponseError(format!(
                        "failed to parse streamed tool call arguments: {err}"
                    ))
                })?
            };

            self.append_tool_call(ToolCall {
                id: tool_call.id,
                call_id: tool_call.call_id,
                function: ToolFunction {
                    name: tool_call.name,
                    arguments,
                },
                signature: None,
                additional_params: None,
            });
        }

        Ok(())
    }
}

/// Collects a model event stream into a completion response with optional raw provider data.
pub async fn collect_optional<R>(
    mut events: ModelEventStream<R>,
) -> Result<CompletionResponse<Option<R>>, CompletionError> {
    let mut collector = CompletionCollector::new();
    while let Some(event) = events.next().await {
        collector.push(event)?;
    }
    collector.finish_optional()
}

/// Collects a model event stream into a completion response with required raw provider data.
pub async fn collect<R>(
    events: ModelEventStream<R>,
) -> Result<CompletionResponse<R>, CompletionError> {
    collect_optional(events).await.and_then(|response| {
        let raw_response = response.raw_response.ok_or_else(|| {
            CompletionError::ResponseError(
                "model event stream completed without a raw provider response".to_string(),
            )
        })?;

        Ok(CompletionResponse {
            choice: response.choice,
            usage: response.usage,
            raw_response,
            message_id: response.message_id,
        })
    })
}

/// Creates model events from already-normalized assistant content and raw provider data.
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
    Box::pin(stream! {
        for content in content {
            for event in events_from_assistant_content(content) {
                yield event;
            }
        }

        yield ModelEvent::Usage { usage };
        if message_id.is_some() {
            yield ModelEvent::MessageDone { id: message_id };
        }
        yield ModelEvent::RawResponse {
            response: raw_response,
        };
        yield ModelEvent::Done;
    })
}

/// Converts a stream of fallible model events into the canonical infallible event stream.
pub fn result_stream<R, S>(stream: S) -> ModelEventStream<R>
where
    R: WasmCompatSend + 'static,
    S: Stream<Item = Result<ModelEvent<R>, CompletionError>> + WasmCompatSend + 'static,
{
    Box::pin(stream.map(|event| match event {
        Ok(event) => event,
        Err(error) => ModelEvent::Error { error },
    }))
}

fn events_from_assistant_content<R>(content: AssistantContent) -> Vec<ModelEvent<R>> {
    match content {
        AssistantContent::Text(text) => vec![ModelEvent::TextDelta { text: text.text }],
        AssistantContent::ToolCall(tool_call) => vec![ModelEvent::ToolCallDone {
            tool_call,
            internal_call_id: None,
        }],
        AssistantContent::Reasoning(reasoning) => {
            vec![ModelEvent::ReasoningDone { reasoning }]
        }
        AssistantContent::Image(image) => {
            vec![ModelEvent::ProviderMetadata {
                metadata: serde_json::json!({
                    "assistant_content": "image",
                    "image": image,
                }),
            }]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ReasoningContent, Text};

    #[tokio::test]
    async fn collects_text_deltas_into_completion_response() {
        let events = Box::pin(stream! {
            yield ModelEvent::<()>::TextDelta { text: "hello ".to_string() };
            yield ModelEvent::<()>::TextDelta { text: "world".to_string() };
            yield ModelEvent::RawResponse { response: () };
            yield ModelEvent::Done;
        });

        let response = collect(events).await.expect("events should collect");
        let items = response.choice.into_iter().collect::<Vec<_>>();

        assert_eq!(
            items,
            vec![AssistantContent::Text(Text {
                text: "hello world".to_string()
            })]
        );
    }

    #[tokio::test]
    async fn collects_reasoning_and_text_in_arrival_order() {
        let events = Box::pin(stream! {
            yield ModelEvent::<()>::ReasoningDelta {
                id: Some("rs_1".to_string()),
                text: "think".to_string(),
            };
            yield ModelEvent::<()>::TextDelta { text: "answer".to_string() };
            yield ModelEvent::RawResponse { response: () };
            yield ModelEvent::Done;
        });

        let response = collect(events).await.expect("events should collect");
        let items = response.choice.into_iter().collect::<Vec<_>>();

        assert!(matches!(
            items.first(),
            Some(AssistantContent::Reasoning(Reasoning {
                id: Some(id),
                content,
            })) if id == "rs_1"
                && matches!(
                    content.first(),
                    Some(ReasoningContent::Text { text, signature: None }) if text == "think"
                )
        ));
        assert!(matches!(
            items.get(1),
            Some(AssistantContent::Text(Text { text })) if text == "answer"
        ));
    }

    #[tokio::test]
    async fn collects_tool_call_delta_when_done_is_missing() {
        let events = Box::pin(stream! {
            yield ModelEvent::<()>::ToolCallStart {
                id: "call_1".to_string(),
                internal_call_id: "internal_1".to_string(),
                call_id: None,
                name: "lookup".to_string(),
            };
            yield ModelEvent::<()>::ToolCallDelta {
                id: "call_1".to_string(),
                internal_call_id: "internal_1".to_string(),
                content: ToolCallDeltaContent::Delta("{\"q\":\"rig\"}".to_string()),
            };
            yield ModelEvent::RawResponse { response: () };
            yield ModelEvent::Done;
        });

        let response = collect(events).await.expect("events should collect");
        let items = response.choice.into_iter().collect::<Vec<_>>();

        assert!(matches!(
            items.first(),
            Some(AssistantContent::ToolCall(ToolCall { id, function, .. }))
                if id == "call_1"
                    && function.name == "lookup"
                    && function.arguments == serde_json::json!({"q": "rig"})
        ));
    }
}
