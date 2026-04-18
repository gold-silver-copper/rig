use std::collections::HashMap;

use async_stream::stream;
use futures::StreamExt;
use http::Request;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::info_span;
use tracing_futures::Instrument;

use crate::completion::{CompletionError, CompletionRequest, GetTokenUsage};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::json_utils;
use crate::providers::openai::completion::{
    CompatibleStreamingToolCall, ToolCallConflictPolicy, apply_compatible_tool_call_deltas,
    take_finalized_tool_calls, take_tool_calls,
};
use crate::providers::openrouter::{
    OpenRouterRequestParams, OpenrouterCompletionRequest, ReasoningDetails,
};
use crate::streaming;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct StreamingCompletionResponse {
    pub usage: Usage,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();

        usage.input_tokens = self.usage.prompt_tokens as u64;
        usage.output_tokens = self.usage.completion_tokens as u64;
        usage.total_tokens = self.usage.total_tokens as u64;

        Some(usage)
    }
}

fn map_finish_reason(reason: &FinishReason) -> crate::completion::StopReason {
    match reason {
        FinishReason::ToolCalls => crate::completion::StopReason::ToolCalls,
        FinishReason::Stop => crate::completion::StopReason::Stop,
        FinishReason::Error => crate::completion::StopReason::Other("error".to_owned()),
        FinishReason::ContentFilter => crate::completion::StopReason::ContentFilter,
        FinishReason::Length => crate::completion::StopReason::MaxTokens,
        FinishReason::Other(other) => crate::completion::StopReason::Other(other.clone()),
    }
}

#[derive(Deserialize, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    ToolCalls,
    Stop,
    Error,
    ContentFilter,
    Length,
    #[serde(untagged)]
    Other(String),
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct StreamingChoice {
    pub finish_reason: Option<FinishReason>,
    pub native_finish_reason: Option<String>,
    pub logprobs: Option<Value>,
    pub index: usize,
    pub delta: StreamingDelta,
}

#[derive(Deserialize, Debug)]
struct StreamingFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct StreamingToolCall {
    pub index: usize,
    pub id: Option<String>,
    pub r#type: Option<String>,
    pub function: StreamingFunction,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct ErrorResponse {
    pub code: i32,
    pub message: String,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct StreamingDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    pub tool_calls: Vec<StreamingToolCall>,
    pub reasoning: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    pub reasoning_details: Vec<ReasoningDetails>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct StreamingCompletionChunk {
    id: String,
    model: String,
    choices: Vec<StreamingChoice>,
    usage: Option<Usage>,
    error: Option<ErrorResponse>,
}

impl<T> super::CompletionModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let request_model = completion_request
            .model
            .clone()
            .unwrap_or_else(|| self.model.clone());
        let preamble = completion_request.preamble.clone();
        let mut request = OpenrouterCompletionRequest::try_from(OpenRouterRequestParams {
            model: request_model.as_ref(),
            request: completion_request,
            strict_tools: self.strict_tools,
        })?;

        let params = json_utils::merge(
            request.additional_params.unwrap_or(serde_json::json!({})),
            serde_json::json!({"stream": true }),
        );

        request.additional_params = Some(params);

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/chat/completions")?
            .body(body)
            .map_err(|x| CompletionError::HttpError(x.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "openrouter",
                gen_ai.request.model = &request_model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cached_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        tracing::Instrument::instrument(
            send_compatible_streaming_request(self.client.clone(), req),
            span,
        )
        .await
    }
}

pub async fn send_compatible_streaming_request<T>(
    http_client: T,
    req: Request<Vec<u8>>,
) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
{
    let span = tracing::Span::current();
    // Build the request with proper headers for SSE
    let mut event_source = GenericEventSource::new(http_client, req);

    let stream = stream! {
        // Accumulate tool calls by index while streaming
        let mut tool_calls: HashMap<usize, streaming::RawStreamingToolCall> = HashMap::new();
        let mut final_usage = None;

        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => {
                    tracing::trace!("SSE connection opened");
                    continue;
                }

                Ok(Event::Message(message)) => {
                    if message.data.trim().is_empty() || message.data == "[DONE]" {
                        continue;
                    }

                    let data = match serde_json::from_str::<StreamingCompletionChunk>(&message.data) {
                        Ok(data) => data,
                        Err(error) => {
                            tracing::error!(?error, message = message.data, "Failed to parse SSE message");
                            continue;
                        }
                    };

                    // Expect at least one choice
                     let Some(choice) = data.choices.first() else {
                        tracing::debug!("There is no choice");
                        continue;
                    };
                    let delta = &choice.delta;

                    if !delta.tool_calls.is_empty() {
                        for event in apply_compatible_tool_call_deltas(
                            &mut tool_calls,
                            delta.tool_calls.iter().map(|tool_call| CompatibleStreamingToolCall {
                                index: tool_call.index,
                                id: tool_call.id.as_deref(),
                                name: tool_call.function.name.as_deref(),
                                arguments: tool_call.function.arguments.as_deref(),
                            }),
                            ToolCallConflictPolicy::KeepIndex,
                        ) {
                            yield Ok(event);
                        }

                        // Update the signature and the additional params of the tool call if present
                        for reasoning_detail in &delta.reasoning_details {
                            if let ReasoningDetails::Encrypted { id, data, .. } = reasoning_detail
                                && let Some(id) = id
                                && let Some(tool_call) = tool_calls.values_mut().find(|tool_call| tool_call.id.eq(id))
                                && let Ok(additional_params) = serde_json::to_value(reasoning_detail) {
                                tool_call.signature = Some(data.clone());
                                tool_call.additional_params = Some(additional_params);
                            }
                        }
                    }

                    // Streamed reasoning content
                    if let Some(reasoning) = &delta.reasoning && !reasoning.is_empty() {
                        yield Ok(streaming::RawStreamingChoice::ReasoningDelta {
                            reasoning: reasoning.clone(),
                            id: None,
                        });
                    }

                    // Streamed text content
                    if let Some(content) = &delta.content && !content.is_empty() {
                        yield Ok(streaming::RawStreamingChoice::Message(content.clone()));
                    }

                    // Usage updates
                    if let Some(usage) = data.usage {
                        final_usage = Some(usage);
                    }

                    // Finish reason
                    if let Some(finish_reason) = &choice.finish_reason && *finish_reason == FinishReason::ToolCalls {
                        for tool_call in take_finalized_tool_calls(&mut tool_calls) {
                            yield Ok(tool_call);
                        }
                    }

                    if let Some(finish_reason) = &choice.finish_reason {
                        yield Ok(streaming::RawStreamingChoice::StopReason(map_finish_reason(
                            finish_reason,
                        )));
                    }
                }
                Err(crate::http_client::Error::StreamEnded) => {
                    break;
                }
                Err(error) => {
                    tracing::error!(?error, "SSE error");
                    yield Err(CompletionError::ProviderError(error.to_string()));
                    break;
                }
            }
        }

        // Ensure event source is closed when stream ends
        event_source.close();

        // Flush any accumulated tool calls (that weren't emitted as ToolCall earlier)
        for tool_call in take_tool_calls(&mut tool_calls) {
            yield Ok(tool_call);
        }

        // Final response with usage
        yield Ok(streaming::RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
            usage: final_usage.unwrap_or_default(),
        }));
    }.instrument(span);

    Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
        stream,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_streaming_completion_response_deserialization() {
        let json = json!({
            "id": "gen-abc123",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Hello"
                }
            }],
            "created": 1234567890u64,
            "model": "gpt-3.5-turbo",
            "object": "chat.completion.chunk"
        });

        let response: StreamingCompletionChunk = serde_json::from_value(json).unwrap();
        assert_eq!(response.id, "gen-abc123");
        assert_eq!(response.model, "gpt-3.5-turbo");
        assert_eq!(response.choices.len(), 1);
    }

    #[test]
    fn test_delta_with_content() {
        let json = json!({
            "role": "assistant",
            "content": "Hello, world!"
        });

        let delta: StreamingDelta = serde_json::from_value(json).unwrap();
        assert_eq!(delta.role, Some("assistant".to_string()));
        assert_eq!(delta.content, Some("Hello, world!".to_string()));
    }

    #[test]
    fn test_delta_with_tool_call() {
        let json = json!({
            "role": "assistant",
            "tool_calls": [{
                "index": 0,
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\":"
                }
            }]
        });

        let delta: StreamingDelta = serde_json::from_value(json).unwrap();
        assert_eq!(delta.tool_calls.len(), 1);
        assert_eq!(delta.tool_calls[0].index, 0);
        assert_eq!(delta.tool_calls[0].id, Some("call_abc".to_string()));
    }

    #[test]
    fn test_tool_call_with_partial_arguments() {
        let json = json!({
            "index": 0,
            "id": null,
            "type": null,
            "function": {
                "name": null,
                "arguments": "Paris"
            }
        });

        let tool_call: StreamingToolCall = serde_json::from_value(json).unwrap();
        assert_eq!(tool_call.index, 0);
        assert!(tool_call.id.is_none());
        assert_eq!(tool_call.function.arguments, Some("Paris".to_string()));
    }

    #[test]
    fn test_streaming_with_usage() {
        let json = json!({
            "id": "gen-xyz",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": null
                }
            }],
            "created": 1234567890u64,
            "model": "gpt-4",
            "object": "chat.completion.chunk",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        });

        let response: StreamingCompletionChunk = serde_json::from_value(json).unwrap();
        assert!(response.usage.is_some());
        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_multiple_tool_call_deltas() {
        // Simulates the sequence of deltas for a tool call with arguments
        let start_json = json!({
            "id": "gen-1",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": ""
                        }
                    }]
                }
            }],
            "created": 1234567890u64,
            "model": "gpt-4",
            "object": "chat.completion.chunk"
        });

        let delta1_json = json!({
            "id": "gen-2",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": "{\"query\":"
                        }
                    }]
                }
            }],
            "created": 1234567890u64,
            "model": "gpt-4",
            "object": "chat.completion.chunk"
        });

        let delta2_json = json!({
            "id": "gen-3",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": "\"Rust programming\"}"
                        }
                    }]
                }
            }],
            "created": 1234567890u64,
            "model": "gpt-4",
            "object": "chat.completion.chunk"
        });

        // Verify all chunks deserialize
        let start: StreamingCompletionChunk = serde_json::from_value(start_json).unwrap();
        assert_eq!(
            start.choices[0].delta.tool_calls[0].id,
            Some("call_123".to_string())
        );

        let delta1: StreamingCompletionChunk = serde_json::from_value(delta1_json).unwrap();
        assert_eq!(
            delta1.choices[0].delta.tool_calls[0].function.arguments,
            Some("{\"query\":".to_string())
        );

        let delta2: StreamingCompletionChunk = serde_json::from_value(delta2_json).unwrap();
        assert_eq!(
            delta2.choices[0].delta.tool_calls[0].function.arguments,
            Some("\"Rust programming\"}".to_string())
        );
    }

    #[test]
    fn test_response_with_error() {
        let json = json!({
            "id": "cmpl-abc123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-3.5-turbo",
            "provider": "openai",
            "error": { "code": 500, "message": "Provider disconnected" },
            "choices": [
                { "index": 0, "delta": { "content": "" }, "finish_reason": "error" }
            ]
        });

        let response: StreamingCompletionChunk = serde_json::from_value(json).unwrap();
        assert!(response.error.is_some());
        let error = response.error.as_ref().unwrap();
        assert_eq!(error.code, 500);
        assert_eq!(error.message, "Provider disconnected");
    }

    #[tokio::test]
    async fn test_stream_captures_stop_reason() {
        use crate::http_client::mock::MockStreamingClient;
        use bytes::Bytes;
        use futures::StreamExt;

        let sse = concat!(
            "data: {\"id\":\"gen-1\",\"model\":\"gpt-4\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":\"stop\"}],\"usage\":null}\n\n",
            "data: {\"id\":\"gen-2\",\"model\":\"gpt-4\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
            "data: [DONE]\n\n",
        );

        let client = MockStreamingClient {
            sse_bytes: Bytes::from(sse),
        };
        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .unwrap();

        let mut stream = send_compatible_streaming_request(client, req)
            .await
            .expect("stream should start");

        while let Some(chunk) = stream.next().await {
            chunk.expect("stream chunk should deserialize");
        }

        assert_eq!(
            stream.stop_reason,
            Some(crate::completion::StopReason::Stop)
        );
    }
}
