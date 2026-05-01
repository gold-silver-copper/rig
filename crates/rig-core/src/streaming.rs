//! This module provides functionality for working with streaming completion models.
//! It provides traits and types for generating streaming completion requests and
//! handling streaming completion responses.
//!
//! The main traits defined in this module are:
//! - [StreamingPrompt]: Defines a high-level streaming LLM one-shot prompt interface
//! - [StreamingChat]: Defines a high-level streaming LLM chat interface with history
//! - [StreamingCompletion]: Defines a low-level streaming LLM completion interface
//!

use crate::OneOrMany;
use crate::agent::Agent;
use crate::agent::prompt_request::hooks::PromptHook;
use crate::agent::prompt_request::streaming::StreamingPromptRequest;
use crate::completion::{
    CompletionError, CompletionModel, CompletionRequestBuilder, CompletionResponse, GetTokenUsage,
    Message, Usage,
};
use crate::message::{
    AssistantContent, Reasoning, ReasoningContent, ToolCall, ToolFunction, ToolResult,
};
use crate::model_event::{IntoModelEvents, ModelEvent};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use futures::stream::{AbortHandle, Abortable};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::watch;

/// Control for pausing and resuming a streaming response
pub struct PauseControl {
    pub(crate) paused_tx: watch::Sender<bool>,
    pub(crate) paused_rx: watch::Receiver<bool>,
}

impl PauseControl {
    pub fn new() -> Self {
        let (paused_tx, paused_rx) = watch::channel(false);
        Self {
            paused_tx,
            paused_rx,
        }
    }

    pub fn pause(&self) {
        let _ = self.paused_tx.send(true);
    }

    pub fn resume(&self) {
        let _ = self.paused_tx.send(false);
    }

    pub fn is_paused(&self) -> bool {
        *self.paused_rx.borrow()
    }
}

impl Default for PauseControl {
    fn default() -> Self {
        Self::new()
    }
}

/// The content of a tool call delta - either the tool name or argument data
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum ToolCallDeltaContent {
    Name(String),
    Delta(String),
}

/// Describes a streaming tool call response (in its entirety)
#[derive(Debug, Clone)]
pub struct RawStreamingToolCall {
    /// Provider-supplied tool call ID.
    pub id: String,
    /// Rig-generated unique identifier for this tool call.
    pub internal_call_id: String,
    pub call_id: Option<String>,
    pub name: String,
    pub arguments: serde_json::Value,
    pub signature: Option<String>,
    pub additional_params: Option<serde_json::Value>,
}

impl RawStreamingToolCall {
    pub fn empty() -> Self {
        Self {
            id: String::new(),
            internal_call_id: nanoid::nanoid!(),
            call_id: None,
            name: String::new(),
            arguments: serde_json::Value::Null,
            signature: None,
            additional_params: None,
        }
    }

    pub fn new(id: String, name: String, arguments: serde_json::Value) -> Self {
        Self {
            id,
            internal_call_id: nanoid::nanoid!(),
            call_id: None,
            name,
            arguments,
            signature: None,
            additional_params: None,
        }
    }

    pub fn with_internal_call_id(mut self, internal_call_id: String) -> Self {
        self.internal_call_id = internal_call_id;
        self
    }

    pub fn with_call_id(mut self, call_id: String) -> Self {
        self.call_id = Some(call_id);
        self
    }

    pub fn with_signature(mut self, signature: Option<String>) -> Self {
        self.signature = signature;
        self
    }

    pub fn with_additional_params(mut self, additional_params: Option<serde_json::Value>) -> Self {
        self.additional_params = additional_params;
        self
    }
}

impl From<RawStreamingToolCall> for ToolCall {
    fn from(tool_call: RawStreamingToolCall) -> Self {
        ToolCall {
            id: tool_call.id,
            call_id: tool_call.call_id,
            function: ToolFunction {
                name: tool_call.name,
                arguments: tool_call.arguments,
            },
            signature: tool_call.signature,
            additional_params: tool_call.additional_params,
        }
    }
}

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
pub type StreamingResult<R> = Pin<Box<dyn Stream<Item = ModelEvent<R>> + Send>>;

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub type StreamingResult<R> = Pin<Box<dyn Stream<Item = ModelEvent<R>>>>;

/// The response from a streaming completion request;
/// message and response are populated at the end of the
/// `inner` stream.
pub struct StreamingCompletionResponse<R>
where
    R: Clone + Unpin + GetTokenUsage,
{
    pub(crate) inner: Abortable<StreamingResult<R>>,
    pub(crate) abort_handle: AbortHandle,
    pub(crate) pause_control: PauseControl,
    assistant_items: Vec<AssistantContent>,
    text_item_index: Option<usize>,
    reasoning_item_index: Option<usize>,
    /// The final aggregated message from the stream
    /// contains all text and tool calls generated
    pub choice: OneOrMany<AssistantContent>,
    /// The final response from the stream, may be `None`
    /// if the provider didn't yield it during the stream
    pub response: Option<R>,
    pub usage: Usage,
    /// Provider-assigned message ID (e.g. OpenAI Responses API `msg_` ID).
    pub message_id: Option<String>,
}

impl<R> StreamingCompletionResponse<R>
where
    R: Clone + Unpin + GetTokenUsage + WasmCompatSend + 'static,
{
    pub fn stream<S, E>(inner: S) -> StreamingCompletionResponse<R>
    where
        S: Stream<Item = E> + WasmCompatSend + 'static,
        E: IntoModelEvents<R> + WasmCompatSend + 'static,
    {
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        let inner = inner.flat_map(|item| futures::stream::iter(item.into_model_events()));
        let inner: StreamingResult<R> = Box::pin(inner);
        let abortable_stream = Abortable::new(inner, abort_registration);
        let pause_control = PauseControl::new();
        Self {
            inner: abortable_stream,
            abort_handle,
            pause_control,
            assistant_items: vec![],
            text_item_index: None,
            reasoning_item_index: None,
            choice: OneOrMany::one(AssistantContent::text("")),
            response: None,
            usage: Usage::new(),
            message_id: None,
        }
    }

    pub fn cancel(&self) {
        self.abort_handle.abort();
    }

    pub fn pause(&self) {
        self.pause_control.pause();
    }

    pub fn resume(&self) {
        self.pause_control.resume();
    }

    pub fn is_paused(&self) -> bool {
        self.pause_control.is_paused()
    }

    fn append_text_chunk(&mut self, text: &str) {
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

    /// Accumulate streaming reasoning delta text into assistant_items.
    /// Providers that only emit ReasoningDelta (not full Reasoning blocks)
    /// need this so the aggregated response includes reasoning content.
    fn append_reasoning_chunk(&mut self, id: &Option<String>, text: &str) {
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

    fn record_event(&mut self, event: &ModelEvent<R>) {
        match event {
            ModelEvent::TextDelta { text } => {
                self.reasoning_item_index = None;
                self.append_text_chunk(text);
            }
            ModelEvent::ReasoningDone { reasoning } => {
                self.text_item_index = None;
                self.reasoning_item_index = None;
                self.assistant_items
                    .push(AssistantContent::Reasoning(reasoning.clone()));
            }
            ModelEvent::ReasoningDelta { id, text } => {
                self.text_item_index = None;
                self.append_reasoning_chunk(id, text);
            }
            ModelEvent::ToolCallDone { tool_call, .. } => {
                self.text_item_index = None;
                self.reasoning_item_index = None;
                self.assistant_items
                    .push(AssistantContent::ToolCall(tool_call.clone()));
            }
            ModelEvent::MessageDone { id } => {
                if id.is_some() {
                    self.message_id = id.clone();
                }
            }
            ModelEvent::Usage { usage } => {
                self.usage = *usage;
            }
            ModelEvent::RawResponse { response } => {
                self.response = Some(response.clone());
            }
            ModelEvent::ToolCallStart { .. }
            | ModelEvent::ToolCallDelta { .. }
            | ModelEvent::ProviderMetadata { .. }
            | ModelEvent::Error { .. }
            | ModelEvent::Done => {}
        }
    }
}

impl<R> From<StreamingCompletionResponse<R>> for CompletionResponse<Option<R>>
where
    R: Clone + Unpin + GetTokenUsage,
{
    fn from(value: StreamingCompletionResponse<R>) -> CompletionResponse<Option<R>> {
        CompletionResponse {
            choice: value.choice,
            usage: value.usage,
            raw_response: value.response,
            message_id: value.message_id,
        }
    }
}

impl<R> Stream for StreamingCompletionResponse<R>
where
    R: Clone + Unpin + GetTokenUsage + WasmCompatSend + 'static,
{
    type Item = ModelEvent<R>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let stream = self.get_mut();

        if stream.is_paused() {
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }

        match Pin::new(&mut stream.inner).poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                // This is run at the end of the inner stream to collect all tokens into
                // a single unified `Message`.
                if stream.assistant_items.is_empty() {
                    stream.assistant_items.push(AssistantContent::text(""));
                }

                if let Some(choice) =
                    OneOrMany::from_iter_optional(std::mem::take(&mut stream.assistant_items))
                {
                    stream.choice = choice;
                }

                Poll::Ready(None)
            }
            Poll::Ready(Some(event)) => {
                if let ModelEvent::Error { error } = &event
                    && matches!(error, CompletionError::ProviderError(e) if e.contains("aborted"))
                {
                    return Poll::Ready(None);
                }

                stream.record_event(&event);
                Poll::Ready(Some(event))
            }
        }
    }
}

/// Trait for high-level streaming prompt interface.
///
/// This trait provides a simple interface for streaming prompts to a completion model.
/// Implementations can optionally support prompt hooks for observing and controlling
/// the agent's execution lifecycle.
pub trait StreamingPrompt<M, R>
where
    M: CompletionModel + 'static,
    <M as CompletionModel>::StreamingResponse: WasmCompatSend,
    R: Clone + Unpin + GetTokenUsage,
{
    /// The hook type used by this streaming prompt implementation.
    ///
    /// If your implementation does not need prompt hooks, use `()` as the hook type:
    ///
    /// ```ignore
    /// impl<M, R> StreamingPrompt<M, R> for MyType<M>
    /// where
    ///     M: CompletionModel + 'static,
    ///     // ... other bounds ...
    /// {
    ///     type Hook = ();
    ///
    ///     fn stream_prompt(&self, prompt: impl Into<Message>) -> StreamingPromptRequest<M, ()> {
    ///         // ...
    ///     }
    /// }
    /// ```
    type Hook: PromptHook<M>;

    /// Stream a simple prompt to the model
    fn stream_prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> StreamingPromptRequest<M, Self::Hook>;
}

/// Trait for high-level streaming chat interface with conversation history.
///
/// This trait provides an interface for streaming chat completions with support
/// for maintaining conversation history. Implementations can optionally support
/// prompt hooks for observing and controlling the agent's execution lifecycle.
pub trait StreamingChat<M, R>: WasmCompatSend + WasmCompatSync
where
    M: CompletionModel + 'static,
    <M as CompletionModel>::StreamingResponse: WasmCompatSend,
    R: Clone + Unpin + GetTokenUsage,
{
    /// The hook type used by this streaming chat implementation.
    ///
    /// If your implementation does not need prompt hooks, use `()` as the hook type:
    ///
    /// ```ignore
    /// impl<M, R> StreamingChat<M, R> for MyType<M>
    /// where
    ///     M: CompletionModel + 'static,
    ///     // ... other bounds ...
    /// {
    ///     type Hook = ();
    ///
    ///     fn stream_chat(
    ///         &self,
    ///         prompt: impl Into<Message>,
    ///         chat_history: Vec<Message>,
    ///     ) -> StreamingPromptRequest<M, ()> {
    ///         // ...
    ///     }
    /// }
    /// ```
    type Hook: PromptHook<M>;

    /// Stream a chat with history to the model.
    ///
    /// The messages returned by the model can be accessed via `FinalResponse::history()`
    ///
    /// You are responsible for managing history, a simple linear solution could look like:
    /// ```ignore
    ///  let mut history = vec![];
    ///
    ///  loop {
    ///      let prompt = "Create GPT-67, make no mistakes";
    ///      let mut stream = agent.stream_chat(prompt, &history).await;
    ///
    ///      while let Some(msg) = stream.next().await {
    ///         match msg {
    ///              Ok(MultiTurnStreamItem::FinalResponse(fin)) => {
    ///                  history.extend_from_slice(fin.history().unwrap_or_default());
    ///                  break;
    ///             }
    ///             Ok(_other) => { /* Do something with this chunk */ }
    ///             Err(e) => return Err(e.into()),
    ///         }
    ///     }
    /// }
    /// ```
    fn stream_chat<I, T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: I,
    ) -> StreamingPromptRequest<M, Self::Hook>
    where
        I: IntoIterator<Item = T> + WasmCompatSend,
        T: Into<Message>;
}

/// Trait for low-level streaming completion interface
pub trait StreamingCompletion<M: CompletionModel> {
    /// Generate a streaming completion from a request
    fn stream_completion<I, T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: I,
    ) -> impl Future<Output = Result<CompletionRequestBuilder<M>, CompletionError>>
    where
        I: IntoIterator<Item = T> + WasmCompatSend,
        T: Into<Message>;
}

/// A helper function to stream a completion request to stdout.
/// Tool call deltas are ignored as tool calls are generally much easier to handle when received in their entirety rather than using deltas.
pub async fn stream_to_stdout<M>(
    agent: &'static Agent<M>,
    stream: &mut StreamingCompletionResponse<M::StreamingResponse>,
) -> Result<(), std::io::Error>
where
    M: CompletionModel,
{
    let mut is_reasoning = false;
    print!("Response: ");
    while let Some(chunk) = stream.next().await {
        match chunk {
            ModelEvent::TextDelta { text } => {
                if is_reasoning {
                    is_reasoning = false;
                    println!("\n---\n");
                }
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            ModelEvent::ToolCallDone {
                tool_call,
                internal_call_id: _,
            } => {
                let res = agent
                    .tool_server_handle
                    .call_tool(
                        &tool_call.function.name,
                        &tool_call.function.arguments.to_string(),
                    )
                    .await
                    .map_err(|x| std::io::Error::other(x.to_string()))?;
                println!("\nResult: {res}");
            }
            ModelEvent::RawResponse { response } => {
                if let Ok(json_res) = serde_json::to_string_pretty(&response) {
                    println!();
                    tracing::info!("Final result: {json_res}");
                }
            }
            ModelEvent::ReasoningDone { reasoning } => {
                if !is_reasoning {
                    is_reasoning = true;
                    println!();
                    println!("Thinking: ");
                }
                let reasoning = reasoning.display_text();

                print!("{reasoning}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            ModelEvent::ReasoningDelta { text, .. } => {
                if !is_reasoning {
                    is_reasoning = true;
                    println!();
                    println!("Thinking: ");
                }

                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            ModelEvent::Error { error } => {
                let e = error;
                if e.to_string().contains("aborted") {
                    println!("\nStream cancelled.");
                    break;
                }
                eprintln!("Error: {e}");
                break;
            }
            ModelEvent::ToolCallStart { .. }
            | ModelEvent::ToolCallDelta { .. }
            | ModelEvent::MessageDone { .. }
            | ModelEvent::Usage { .. }
            | ModelEvent::ProviderMetadata { .. }
            | ModelEvent::Done => {}
        }
    }

    println!(); // New line after streaming completes

    Ok(())
}

// Test module
#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::message::Text;
    use async_stream::stream;
    use tokio::time::sleep;

    #[derive(Debug, Clone)]
    pub struct MockResponse {
        #[allow(dead_code)]
        token_count: u32,
    }

    impl GetTokenUsage for MockResponse {
        fn token_usage(&self) -> Option<crate::completion::Usage> {
            let mut usage = Usage::new();
            usage.total_tokens = 15;
            Some(usage)
        }
    }

    fn to_stream_result(
        stream: impl futures::Stream<Item = Result<ModelEvent<MockResponse>, CompletionError>> + 'static,
    ) -> impl futures::Stream<Item = Result<ModelEvent<MockResponse>, CompletionError>> + 'static
    {
        stream
    }

    fn create_mock_stream() -> StreamingCompletionResponse<MockResponse> {
        let stream = stream! {
            yield Ok(ModelEvent::TextDelta { text: "hello 1".to_string() });
            sleep(Duration::from_millis(100)).await;
            yield Ok(ModelEvent::TextDelta { text: "hello 2".to_string() });
            sleep(Duration::from_millis(100)).await;
            yield Ok(ModelEvent::TextDelta { text: "hello 3".to_string() });
            sleep(Duration::from_millis(100)).await;
            let response = MockResponse { token_count: 15 };
            if let Some(usage) = response.token_usage() {
                yield Ok(ModelEvent::Usage { usage });
            }
            yield Ok(ModelEvent::RawResponse { response });
            yield Ok(ModelEvent::Done);
        };

        StreamingCompletionResponse::stream(to_stream_result(stream))
    }

    fn create_reasoning_stream() -> StreamingCompletionResponse<MockResponse> {
        let stream = stream! {
            yield Ok(ModelEvent::ReasoningDone {
                reasoning: Reasoning {
                    id: Some("rs_1".to_string()),
                    content: vec![ReasoningContent::Text {
                    text: "step one".to_string(),
                    signature: Some("sig_1".to_string()),
                    }],
                }
            });
            yield Ok(ModelEvent::TextDelta { text: "final answer".to_string() });
            let response = MockResponse { token_count: 5 };
            if let Some(usage) = response.token_usage() {
                yield Ok(ModelEvent::Usage { usage });
            }
            yield Ok(ModelEvent::RawResponse { response });
            yield Ok(ModelEvent::Done);
        };

        StreamingCompletionResponse::stream(to_stream_result(stream))
    }

    fn create_reasoning_only_stream() -> StreamingCompletionResponse<MockResponse> {
        let stream = stream! {
            yield Ok(ModelEvent::ReasoningDone {
                reasoning: Reasoning {
                    id: Some("rs_only".to_string()),
                    content: vec![ReasoningContent::Summary("hidden summary".to_string())],
                }
            });
            let response = MockResponse { token_count: 2 };
            if let Some(usage) = response.token_usage() {
                yield Ok(ModelEvent::Usage { usage });
            }
            yield Ok(ModelEvent::RawResponse { response });
            yield Ok(ModelEvent::Done);
        };

        StreamingCompletionResponse::stream(to_stream_result(stream))
    }

    fn create_interleaved_stream() -> StreamingCompletionResponse<MockResponse> {
        let stream = stream! {
            yield Ok(ModelEvent::ReasoningDone {
                reasoning: Reasoning {
                    id: Some("rs_interleaved".to_string()),
                    content: vec![ReasoningContent::Text {
                    text: "chain-of-thought".to_string(),
                    signature: None,
                    }],
                }
            });
            yield Ok(ModelEvent::TextDelta { text: "final-text".to_string() });
            yield Ok(ModelEvent::from(
                RawStreamingToolCall::new(
                    "tool_1".to_string(),
                    "mock_tool".to_string(),
                    serde_json::json!({"arg": 1}),
                ),
            ));
            let response = MockResponse { token_count: 3 };
            if let Some(usage) = response.token_usage() {
                yield Ok(ModelEvent::Usage { usage });
            }
            yield Ok(ModelEvent::RawResponse { response });
            yield Ok(ModelEvent::Done);
        };

        StreamingCompletionResponse::stream(to_stream_result(stream))
    }

    fn create_text_tool_text_stream() -> StreamingCompletionResponse<MockResponse> {
        let stream = stream! {
            yield Ok(ModelEvent::TextDelta { text: "first".to_string() });
            yield Ok(ModelEvent::from(
                RawStreamingToolCall::new(
                    "tool_split".to_string(),
                    "mock_tool".to_string(),
                    serde_json::json!({"arg": "x"}),
                ),
            ));
            yield Ok(ModelEvent::TextDelta { text: "second".to_string() });
            let response = MockResponse { token_count: 3 };
            if let Some(usage) = response.token_usage() {
                yield Ok(ModelEvent::Usage { usage });
            }
            yield Ok(ModelEvent::RawResponse { response });
            yield Ok(ModelEvent::Done);
        };

        StreamingCompletionResponse::stream(to_stream_result(stream))
    }

    #[tokio::test]
    async fn test_stream_cancellation() {
        let mut stream = create_mock_stream();

        println!("Response: ");
        let mut chunk_count = 0;
        while let Some(chunk) = stream.next().await {
            match chunk {
                ModelEvent::TextDelta { text } => {
                    print!("{text}");
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                    chunk_count += 1;
                }
                ModelEvent::ToolCallDone {
                    tool_call,
                    internal_call_id,
                } => {
                    println!("\nTool Call: {tool_call:?}, internal_call_id={internal_call_id:?}");
                    chunk_count += 1;
                }
                ModelEvent::ToolCallDelta {
                    id,
                    internal_call_id,
                    content,
                } => {
                    println!(
                        "\nTool Call delta: id={id:?}, internal_call_id={internal_call_id:?}, content={content:?}"
                    );
                    chunk_count += 1;
                }
                ModelEvent::RawResponse { response } => {
                    println!("\nFinal response: {response:?}");
                }
                ModelEvent::ReasoningDone { reasoning } => {
                    let reasoning = reasoning.display_text();
                    print!("{reasoning}");
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                }
                ModelEvent::ReasoningDelta { text, .. } => {
                    println!("Reasoning delta: {text}");
                    chunk_count += 1;
                }
                ModelEvent::Error { error } => {
                    eprintln!("Error: {error:?}");
                    break;
                }
                ModelEvent::ToolCallStart { .. }
                | ModelEvent::MessageDone { .. }
                | ModelEvent::Usage { .. }
                | ModelEvent::ProviderMetadata { .. }
                | ModelEvent::Done => {}
            }

            if chunk_count >= 2 {
                println!("\nCancelling stream...");
                stream.cancel();
                println!("Stream cancelled.");
                break;
            }
        }

        let next_chunk = stream.next().await;
        assert!(
            next_chunk.is_none(),
            "Expected no further chunks after cancellation, got {next_chunk:?}"
        );
    }

    #[tokio::test]
    async fn test_stream_pause_resume() {
        let stream = create_mock_stream();

        // Test pause
        stream.pause();
        assert!(stream.is_paused());

        // Test resume
        stream.resume();
        assert!(!stream.is_paused());
    }

    #[tokio::test]
    async fn test_stream_aggregates_reasoning_content() {
        let mut stream = create_reasoning_stream();
        while stream.next().await.is_some() {}

        let choice_items: Vec<AssistantContent> = stream.choice.clone().into_iter().collect();

        assert!(choice_items.iter().any(|item| matches!(
            item,
            AssistantContent::Reasoning(Reasoning {
                id: Some(id),
                content
            }) if id == "rs_1"
                && matches!(
                    content.first(),
                    Some(ReasoningContent::Text {
                        text,
                        signature: Some(signature)
                    }) if text == "step one" && signature == "sig_1"
                )
        )));
    }

    #[tokio::test]
    async fn test_stream_reasoning_only_does_not_inject_empty_text() {
        let mut stream = create_reasoning_only_stream();
        while stream.next().await.is_some() {}

        let choice_items: Vec<AssistantContent> = stream.choice.clone().into_iter().collect();
        assert_eq!(choice_items.len(), 1);
        assert!(matches!(
            choice_items.first(),
            Some(AssistantContent::Reasoning(Reasoning { id: Some(id), .. })) if id == "rs_only"
        ));
    }

    #[tokio::test]
    async fn test_stream_aggregates_assistant_items_in_arrival_order() {
        let mut stream = create_interleaved_stream();
        while stream.next().await.is_some() {}

        let choice_items: Vec<AssistantContent> = stream.choice.clone().into_iter().collect();
        assert_eq!(choice_items.len(), 3);
        assert!(matches!(
            choice_items.first(),
            Some(AssistantContent::Reasoning(Reasoning { id: Some(id), .. })) if id == "rs_interleaved"
        ));
        assert!(matches!(
            choice_items.get(1),
            Some(AssistantContent::Text(Text { text })) if text == "final-text"
        ));
        assert!(matches!(
            choice_items.get(2),
            Some(AssistantContent::ToolCall(ToolCall { id, .. })) if id == "tool_1"
        ));
    }

    #[tokio::test]
    async fn test_stream_keeps_non_contiguous_text_chunks_split_by_tool_call() {
        let mut stream = create_text_tool_text_stream();
        while stream.next().await.is_some() {}

        let choice_items: Vec<AssistantContent> = stream.choice.clone().into_iter().collect();
        assert_eq!(choice_items.len(), 3);
        assert!(matches!(
            choice_items.first(),
            Some(AssistantContent::Text(Text { text })) if text == "first"
        ));
        assert!(matches!(
            choice_items.get(1),
            Some(AssistantContent::ToolCall(ToolCall { id, .. })) if id == "tool_split"
        ));
        assert!(matches!(
            choice_items.get(2),
            Some(AssistantContent::Text(Text { text })) if text == "second"
        ));
    }
}

/// Streamed user content. This content is primarily used to represent tool results from tool calls made during a multi-turn/step agent prompt.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum StreamedUserContent {
    ToolResult {
        tool_result: ToolResult,
        /// Rig-generated unique identifier for the tool call this result
        /// belongs to. Use this to correlate with the originating
        /// [`ModelEvent::ToolCallDone`] internal call id.
        internal_call_id: String,
    },
}

impl StreamedUserContent {
    pub fn tool_result(tool_result: ToolResult, internal_call_id: String) -> Self {
        Self::ToolResult {
            tool_result,
            internal_call_id,
        }
    }
}
