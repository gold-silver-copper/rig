use bytes::Bytes;
use serde_json::json;

use super::*;
use crate::{
    completion::{self, CompletionError},
    http_client::mock::MockStreamingClient,
    providers::conformance::{
        BoxFuture, Fixture, Harness, NormalizedItem, Outcome, StopReason, Turn, drain_stream,
        normalize_completion_response, provider_conformance_tests,
    },
};

struct OpenAiChatHarness;

impl Harness for OpenAiChatHarness {
    fn family_name() -> &'static str {
        "openai-chat"
    }

    fn expected(case: Fixture) -> Outcome<Turn> {
        match case {
            Fixture::EmptyAssistantTurnAfterToolResult => Outcome::Supported(Turn {
                items: vec![],
                message_id: None,
                stop_reason: Some(StopReason::Stop),
            }),
            Fixture::ToolOnlyTurn => Outcome::Supported(Turn {
                items: vec![NormalizedItem::ToolCall {
                    id: "call_lookup".to_string(),
                    name: "lookup_weather".to_string(),
                    arguments: json!({"city": "Paris"}),
                }],
                message_id: None,
                stop_reason: Some(StopReason::ToolCalls),
            }),
            Fixture::TextAndToolCallTurn => Outcome::Supported(Turn {
                items: vec![
                    NormalizedItem::Text("Need weather data first.".to_string()),
                    NormalizedItem::ToolCall {
                        id: "call_lookup".to_string(),
                        name: "lookup_weather".to_string(),
                        arguments: json!({"city": "Paris"}),
                    },
                ],
                message_id: None,
                stop_reason: Some(StopReason::ToolCalls),
            }),
            Fixture::EmptyTextBlocks => Outcome::Supported(Turn {
                items: vec![],
                message_id: None,
                stop_reason: Some(StopReason::Stop),
            }),
            Fixture::ReasoningOnlyTurn => Outcome::Unsupported(
                "OpenAI chat completions do not expose normalized reasoning blocks",
            ),
            Fixture::MessageIdPreservation => {
                Outcome::Unsupported("OpenAI chat completions do not expose message IDs")
            }
            Fixture::StopReasonMapping => Outcome::Supported(Turn {
                items: vec![NormalizedItem::Text("Truncated response".to_string())],
                message_id: None,
                stop_reason: Some(StopReason::MaxTokens),
            }),
        }
    }

    fn non_stream(case: Fixture) -> Result<Outcome<Turn>, CompletionError> {
        match case {
            Fixture::ReasoningOnlyTurn => Ok(Self::expected(case)),
            Fixture::MessageIdPreservation => Ok(Self::expected(case)),
            _ => {
                let raw = non_stream_response(case);
                let response: completion::CompletionResponse<CompletionResponse> =
                    raw.try_into()?;
                Ok(Outcome::Supported(normalize_completion_response(&response)))
            }
        }
    }

    fn stream(case: Fixture) -> BoxFuture<Result<Outcome<Turn>, CompletionError>> {
        Box::pin(async move {
            match case {
                Fixture::ReasoningOnlyTurn => Ok(Self::expected(case)),
                Fixture::MessageIdPreservation => Ok(Self::expected(case)),
                _ => {
                    let client = MockStreamingClient {
                        sse_bytes: Bytes::from(streaming_sse(case)),
                    };
                    let request = http::Request::builder()
                        .method("POST")
                        .uri("http://localhost/v1/chat/completions")
                        .body(Vec::new())
                        .expect("request should build");
                    let stream =
                        streaming::send_compatible_streaming_request(client, request).await?;
                    let response = drain_stream(stream).await?;
                    Ok(Outcome::Supported(normalize_completion_response(&response)))
                }
            }
        })
    }
}

fn non_stream_response(case: Fixture) -> CompletionResponse {
    match case {
        Fixture::EmptyAssistantTurnAfterToolResult => response_with_message(
            Message::Assistant {
                content: vec![],
                refusal: None,
                audio: None,
                name: None,
                tool_calls: vec![],
            },
            "stop",
        ),
        Fixture::ToolOnlyTurn => response_with_message(
            Message::Assistant {
                content: vec![],
                refusal: None,
                audio: None,
                name: None,
                tool_calls: vec![tool_call(
                    "call_lookup",
                    "lookup_weather",
                    json!({"city": "Paris"}),
                )],
            },
            "tool_calls",
        ),
        Fixture::TextAndToolCallTurn => response_with_message(
            Message::Assistant {
                content: vec![AssistantContent::Text {
                    text: "Need weather data first.".to_string(),
                }],
                refusal: None,
                audio: None,
                name: None,
                tool_calls: vec![tool_call(
                    "call_lookup",
                    "lookup_weather",
                    json!({"city": "Paris"}),
                )],
            },
            "tool_calls",
        ),
        Fixture::EmptyTextBlocks => response_with_message(
            Message::Assistant {
                content: vec![AssistantContent::Text {
                    text: String::new(),
                }],
                refusal: None,
                audio: None,
                name: None,
                tool_calls: vec![],
            },
            "stop",
        ),
        Fixture::StopReasonMapping => response_with_message(
            Message::Assistant {
                content: vec![AssistantContent::Text {
                    text: "Truncated response".to_string(),
                }],
                refusal: None,
                audio: None,
                name: None,
                tool_calls: vec![],
            },
            "length",
        ),
        Fixture::ReasoningOnlyTurn | Fixture::MessageIdPreservation => {
            unreachable!("unsupported cases are handled before construction")
        }
    }
}

fn response_with_message(message: Message, finish_reason: &str) -> CompletionResponse {
    CompletionResponse {
        id: "chatcmpl_123".to_string(),
        object: "chat.completion".to_string(),
        created: 0,
        model: "gpt-4o-mini".to_string(),
        system_fingerprint: None,
        choices: vec![Choice {
            index: 0,
            message,
            logprobs: None,
            finish_reason: finish_reason.to_string(),
        }],
        usage: Some(Usage {
            prompt_tokens: 10,
            total_tokens: 15,
            prompt_tokens_details: None,
        }),
    }
}

fn tool_call(id: &str, name: &str, arguments: serde_json::Value) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        r#type: ToolType::Function,
        function: Function {
            name: name.to_string(),
            arguments,
        },
    }
}

fn streaming_sse(case: Fixture) -> String {
    match case {
        Fixture::EmptyAssistantTurnAfterToolResult => concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"\",\"tool_calls\":[]},\"finish_reason\":\"stop\"}],\"usage\":null}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":0,\"total_tokens\":10}}\n\n",
            "data: [DONE]\n\n",
        )
        .to_string(),
        Fixture::ToolOnlyTurn => concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_lookup\",\"function\":{\"name\":\"lookup_weather\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}]},\"finish_reason\":null}],\"usage\":null}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
            "data: [DONE]\n\n",
        )
        .to_string(),
        Fixture::TextAndToolCallTurn => concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Need weather data first.\",\"tool_calls\":[]},\"finish_reason\":null}],\"usage\":null}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_lookup\",\"function\":{\"name\":\"lookup_weather\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}]},\"finish_reason\":null}],\"usage\":null}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":7,\"total_tokens\":17}}\n\n",
            "data: [DONE]\n\n",
        )
        .to_string(),
        Fixture::EmptyTextBlocks => concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"\",\"tool_calls\":[]},\"finish_reason\":\"stop\"}],\"usage\":null}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":0,\"total_tokens\":10}}\n\n",
            "data: [DONE]\n\n",
        )
        .to_string(),
        Fixture::StopReasonMapping => concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Truncated response\",\"tool_calls\":[]},\"finish_reason\":\"length\"}],\"usage\":null}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12}}\n\n",
            "data: [DONE]\n\n",
        )
        .to_string(),
        Fixture::ReasoningOnlyTurn | Fixture::MessageIdPreservation => unreachable!(),
    }
}

provider_conformance_tests!(OpenAiChatHarness);
