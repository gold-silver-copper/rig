//! Integration tests for `PromptResponse.messages` using mock models.
//! Exercises the real agent loop code path with mocked LLM responses.

use anyhow::{Context, Result, anyhow};
use rig::OneOrMany;
use rig::agent::AgentBuilder;
use rig::completion::{
    CompletionError, CompletionModel, CompletionRequest, CompletionResponse, Message, Prompt, Usage,
};
use rig::message::{AssistantContent, Text, ToolCall, ToolFunction, UserContent};
use rig::streaming::{StreamingCompletionResponse, StreamingResult};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// Mock model infrastructure
// ---------------------------------------------------------------------------

/// A mock model that returns a fixed text response on every call.
#[derive(Clone)]
struct SimpleTextModel;

#[allow(refining_impl_trait)]
impl CompletionModel for SimpleTextModel {
    type Response = ();
    type StreamingResponse = ();
    type Client = ();

    fn make(_: &Self::Client, _: impl Into<String>) -> Self {
        Self
    }

    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        Ok(CompletionResponse {
            choice: OneOrMany::one(AssistantContent::Text(Text {
                text: "hello from mock".to_string(),
            })),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
            raw_response: (),
            message_id: Some("msg_mock_1".to_string()),
        })
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let stream: StreamingResult<()> = Box::pin(futures::stream::empty());
        Ok(StreamingCompletionResponse::stream(stream))
    }
}

/// A mock model that returns a tool call on the first turn, then a text response.
/// This exercises the multi-turn agent loop.
#[derive(Clone)]
struct ToolThenTextModel {
    turn: Arc<AtomicUsize>,
}

impl ToolThenTextModel {
    fn new() -> Self {
        Self {
            turn: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[allow(refining_impl_trait)]
impl CompletionModel for ToolThenTextModel {
    type Response = ();
    type StreamingResponse = ();
    type Client = ();

    fn make(_: &Self::Client, _: impl Into<String>) -> Self {
        Self::new()
    }

    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let turn = self.turn.fetch_add(1, Ordering::SeqCst);

        if turn == 0 {
            // First turn: return a tool call
            Ok(CompletionResponse {
                choice: OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
                    "tc_1".to_string(),
                    ToolFunction::new(
                        "calculator".to_string(),
                        serde_json::json!({"op": "add", "a": 2, "b": 3}),
                    ),
                ))),
                usage: Usage {
                    input_tokens: 15,
                    output_tokens: 8,
                    total_tokens: 23,
                    cached_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                },
                raw_response: (),
                message_id: Some("msg_tool".to_string()),
            })
        } else {
            // Second turn: return a text response
            Ok(CompletionResponse {
                choice: OneOrMany::one(AssistantContent::Text(Text {
                    text: "The answer is 5".to_string(),
                })),
                usage: Usage {
                    input_tokens: 20,
                    output_tokens: 4,
                    total_tokens: 24,
                    cached_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                },
                raw_response: (),
                message_id: Some("msg_text".to_string()),
            })
        }
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let stream: StreamingResult<()> = Box::pin(futures::stream::empty());
        Ok(StreamingCompletionResponse::stream(stream))
    }
}

/// A mock model that always returns tool calls, never text.
/// Used to test the MaxTurnsError path.
#[derive(Clone)]
struct AlwaysToolCallModel;

#[allow(refining_impl_trait)]
impl CompletionModel for AlwaysToolCallModel {
    type Response = ();
    type StreamingResponse = ();
    type Client = ();

    fn make(_: &Self::Client, _: impl Into<String>) -> Self {
        Self
    }

    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        Ok(CompletionResponse {
            choice: OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
                "tc_loop".to_string(),
                ToolFunction::new("infinite_tool".to_string(), serde_json::json!({"x": 1})),
            ))),
            usage: Usage::new(),
            raw_response: (),
            message_id: None,
        })
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let stream: StreamingResult<()> = Box::pin(futures::stream::empty());
        Ok(StreamingCompletionResponse::stream(stream))
    }
}

fn user_text(message: &Message) -> Result<String> {
    match message {
        Message::User { content } => match content.first() {
            UserContent::Text(text) => Ok(text.text.clone()),
            other => Err(anyhow!("expected text user content, got: {other:?}")),
        },
        other => Err(anyhow!("expected User message, got: {other:?}")),
    }
}

fn assistant_text(message: &Message) -> Result<String> {
    match message {
        Message::Assistant { content, .. } => match content.first() {
            AssistantContent::Text(text) => Ok(text.text.clone()),
            other => Err(anyhow!("expected text assistant content, got: {other:?}")),
        },
        other => Err(anyhow!("expected Assistant message, got: {other:?}")),
    }
}

fn assert_assistant_tool_call(message: &Message) -> Result<()> {
    match message {
        Message::Assistant { content, .. } => {
            assert!(
                matches!(content.first(), AssistantContent::ToolCall(_)),
                "expected tool call, got: {content:?}"
            );
            Ok(())
        }
        other => Err(anyhow!("expected Assistant with tool call, got: {other:?}")),
    }
}

fn assert_user_tool_result(message: &Message) -> Result<()> {
    match message {
        Message::User { content } => {
            assert!(
                matches!(content.first(), UserContent::ToolResult(_)),
                "expected tool result, got: {content:?}"
            );
            Ok(())
        }
        other => Err(anyhow!("expected User with tool result, got: {other:?}")),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Test 1: Standard path still returns a plain String (backward compat).
#[tokio::test]
async fn standard_prompt_returns_string() -> Result<()> {
    let agent = AgentBuilder::new(SimpleTextModel).build();

    let result: String = agent.prompt("hi").await?;

    assert_eq!(result, "hello from mock");
    Ok(())
}

/// Test 2: `extended_details()` returns a `PromptResponse` with `messages: Some(...)`.
#[tokio::test]
async fn extended_details_populates_messages() -> Result<()> {
    let agent = AgentBuilder::new(SimpleTextModel).build();

    let resp = agent.prompt("hi").extended_details().await?;

    assert_eq!(resp.output, "hello from mock");
    assert_eq!(resp.usage.input_tokens, 10);
    assert_eq!(resp.usage.output_tokens, 5);

    // Messages should be populated
    let messages = resp
        .messages
        .context("messages should be Some for extended_details")?;

    // Should contain: [User("hi"), Assistant("hello from mock")]
    assert_eq!(messages.len(), 2);

    // First message: User
    assert_eq!(
        user_text(messages.first().context("missing user message")?)?,
        "hi"
    );

    // Second message: Assistant
    assert_eq!(
        assistant_text(messages.get(1).context("missing assistant message")?)?,
        "hello from mock"
    );
    Ok(())
}

/// Test 3: `with_history()` + `extended_details()` — the response messages
/// should contain the full conversation including any provided history.
#[tokio::test]
async fn extended_with_history_both_populated() -> Result<()> {
    let agent = AgentBuilder::new(SimpleTextModel).build();

    let initial_history: Vec<Message> = Vec::new();

    let resp = agent
        .prompt("hello")
        .with_history(&initial_history)
        .extended_details()
        .await?;

    let response_messages = resp.messages.context("messages should be Some")?;

    // Response should contain the full conversation (User + Assistant)
    assert_eq!(response_messages.len(), 2);

    // First message: User
    assert!(matches!(
        response_messages.first(),
        Some(Message::User { .. })
    ));

    // Second message: Assistant
    assert!(matches!(
        response_messages.get(1),
        Some(Message::Assistant { .. })
    ));
    Ok(())
}

/// Test 4: Standard path with `with_history()` — verify the API works with
/// an immutable history reference.
#[tokio::test]
async fn standard_with_history_works() -> Result<()> {
    let agent = AgentBuilder::new(SimpleTextModel).build();

    let history: Vec<Message> = Vec::new();

    let result = agent.prompt("test").with_history(&history).await?;

    assert_eq!(result, "hello from mock");

    // Note: The input history is not mutated. To get the updated history,
    // use `.extended_details()` and access `response.messages`.
    Ok(())
}

/// Test 5: Multi-turn agent loop with tool calls — messages should contain the
/// full conversation: User → Assistant(tool_call) → User(tool_result) → Assistant(text).
#[tokio::test]
async fn multi_turn_messages_include_tool_calls() -> Result<()> {
    let agent = AgentBuilder::new(ToolThenTextModel::new()).build();

    let resp = agent
        .prompt("What is 2 + 3?")
        .max_turns(5)
        .extended_details()
        .await?;

    assert_eq!(resp.output, "The answer is 5");

    let messages = resp.messages.context("messages should be Some")?;

    // Expected sequence:
    // [0] User: "What is 2 + 3?"
    // [1] Assistant: ToolCall(calculator)
    // [2] User: ToolResult (error since calculator tool isn't registered, but that's fine)
    // [3] Assistant: "The answer is 5"
    assert_eq!(messages.len(), 4, "expected 4 messages, got: {messages:#?}");

    // [0] User prompt
    assert!(matches!(messages.first(), Some(Message::User { .. })));

    // [1] Assistant with tool call
    assert_assistant_tool_call(messages.get(1).context("missing assistant tool call")?)?;

    // [2] User with tool result
    assert_user_tool_result(messages.get(2).context("missing tool result")?)?;

    // [3] Assistant with text
    assert_eq!(
        assistant_text(messages.get(3).context("missing final assistant text")?)?,
        "The answer is 5"
    );

    // Usage should be aggregated across both turns
    assert_eq!(resp.usage.input_tokens, 35); // 15 + 20
    assert_eq!(resp.usage.output_tokens, 12); // 8 + 4
    Ok(())
}

/// Test 6: `PromptResponse::new()` backward compatibility — 2-argument constructor
/// should still work, and `messages` should be `None`.
#[tokio::test]
async fn prompt_response_new_backward_compat() {
    use rig::agent::PromptResponse;

    let resp = PromptResponse::new("output text", Usage::new());

    assert_eq!(resp.output, "output text");
    assert!(resp.messages.is_none());
}

/// Test 6b: `PromptResponse` implements `Display`, delegating to `output`.
#[tokio::test]
async fn prompt_response_display_shows_output() {
    use rig::agent::PromptResponse;

    let resp = PromptResponse::new("the answer is 42", Usage::new());

    assert_eq!(format!("{resp}"), "the answer is 42");
    // Also works with format args
    assert_eq!(resp.to_string(), "the answer is 42");
}

/// Test 7: `PromptResponse::with_messages()` builder works correctly.
#[tokio::test]
async fn prompt_response_with_messages_builder() -> Result<()> {
    use rig::agent::PromptResponse;

    let messages = vec![Message::user("hello"), Message::assistant("world")];

    let resp = PromptResponse::new("output", Usage::new()).with_messages(messages.clone());

    assert!(resp.messages.is_some());
    assert_eq!(
        resp.messages
            .as_ref()
            .context("messages should be present")?
            .len(),
        2
    );
    Ok(())
}

/// Test 8: MaxTurnsError still works — the error should contain the chat history.
/// This verifies the error path isn't broken by our changes.
#[tokio::test]
async fn max_turns_error_still_contains_history() -> Result<()> {
    use rig::completion::PromptError;

    let agent = AgentBuilder::new(AlwaysToolCallModel).build();

    let result = agent
        .prompt("do something")
        .max_turns(2)
        .extended_details()
        .await;

    let error = result.err().context("expected MaxTurnsError, got Ok")?;
    match error {
        PromptError::MaxTurnsError {
            max_turns,
            chat_history,
            ..
        } => {
            assert_eq!(max_turns, 2);
            assert!(
                !chat_history.is_empty(),
                "chat_history in error should not be empty"
            );
        }
        other => return Err(anyhow!("expected MaxTurnsError, got: {other:?}")),
    }
    Ok(())
}

/// Test 9: Extended details without `with_history()` — messages should still
/// be populated (this is the core feature: no need for &mut borrow).
#[tokio::test]
async fn extended_details_works_without_with_history() -> Result<()> {
    let agent = AgentBuilder::new(ToolThenTextModel::new()).build();

    // Note: NO .with_history() call — this is the new use case
    let resp = agent
        .prompt("compute 2+3")
        .max_turns(5)
        .extended_details()
        .await?;

    let messages = resp
        .messages
        .context("messages should be Some even without with_history()")?;

    // Should have full multi-turn history
    assert_eq!(messages.len(), 4);
    assert_eq!(resp.output, "The answer is 5");
    Ok(())
}

/// Test 10: Multiple sequential prompts each return independent message histories.
#[tokio::test]
async fn sequential_prompts_have_independent_histories() -> Result<()> {
    let agent = AgentBuilder::new(SimpleTextModel).build();

    let resp1 = agent.prompt("first").extended_details().await?;

    let resp2 = agent.prompt("second").extended_details().await?;

    let msgs1 = resp1.messages.context("messages should be Some")?;
    let msgs2 = resp2.messages.context("messages should be Some")?;

    // Each should have exactly 2 messages (user + assistant)
    assert_eq!(msgs1.len(), 2);
    assert_eq!(msgs2.len(), 2);

    // First prompt's user message should be "first"
    assert_eq!(
        user_text(msgs1.first().context("missing first prompt user message")?)?,
        "first"
    );

    // Second prompt's user message should be "second"
    assert_eq!(
        user_text(
            msgs2
                .first()
                .context("missing second prompt user message")?
        )?,
        "second"
    );
    Ok(())
}
