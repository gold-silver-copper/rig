//! This module provides functionality for working with streaming completion models.
//! It provides traits and types for generating streaming completion requests and
//! handling streaming completion responses.
//!
//! The main traits defined in this module are:
//! - [StreamingPrompt]: Defines a high-level streaming LLM one-shot prompt interface
//! - [StreamingChat]: Defines a high-level streaming LLM chat interface with history
//! - [StreamingCompletion]: Defines a low-level streaming LLM completion interface
//!

use crate::agent::Agent;
use crate::agent::prompt_request::hooks::PromptHook;
use crate::agent::prompt_request::streaming::StreamingPromptRequest;
use crate::completion::{
    CompletionError, CompletionModel, CompletionRequestBuilder, GetTokenUsage, Message,
};
use crate::model_event::{ModelEvent, ModelEventStream};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use futures::StreamExt;
use std::future::Future;

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
    ///              Ok(AgentEvent::FinalResponse(fin)) => {
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
    stream: &mut ModelEventStream<M::StreamingResponse>,
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
