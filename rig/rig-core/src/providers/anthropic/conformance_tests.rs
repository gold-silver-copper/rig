use bytes::Bytes;
use serde_json::json;

use super::*;
use crate::{
    OneOrMany,
    completion::{self, CompletionError},
    http_client::mock::MockStreamingClient,
    providers::conformance::{
        BoxFuture, Fixture, Harness, NormalizedItem, Outcome, StopReason, Turn, drain_stream,
        normalize_completion_response, provider_conformance_tests,
    },
};

struct AnthropicHarness;

impl Harness for AnthropicHarness {
    fn family_name() -> &'static str {
        "anthropic-messages"
    }

    fn expected(case: Fixture) -> Outcome<Turn> {
        match case {
            Fixture::EmptyAssistantTurnAfterToolResult => Outcome::Supported(Turn {
                items: vec![],
                message_id: None,
                stop_reason: Some(StopReason::EndTurn),
            }),
            Fixture::ToolOnlyTurn => Outcome::Supported(Turn {
                items: vec![NormalizedItem::ToolCall {
                    id: "toolu_lookup".to_string(),
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
                        id: "toolu_lookup".to_string(),
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
                stop_reason: Some(StopReason::EndTurn),
            }),
            Fixture::ReasoningOnlyTurn => Outcome::Supported(Turn {
                items: vec![NormalizedItem::Reasoning(
                    "Need to reason about the tool result.".to_string(),
                )],
                message_id: None,
                stop_reason: Some(StopReason::EndTurn),
            }),
            Fixture::MessageIdPreservation => {
                Outcome::Unsupported("Anthropic Messages responses do not expose message IDs")
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
            Fixture::MessageIdPreservation => Ok(Self::expected(case)),
            _ => {
                let raw = non_stream_response(case);
                let stop_reason = raw.stop_reason.as_deref().map(map_stop_reason);
                let response: completion::CompletionResponse<CompletionResponse> =
                    raw.try_into()?;
                Ok(Outcome::Supported(normalize_completion_response(
                    &response,
                    stop_reason,
                )))
            }
        }
    }

    fn stream(case: Fixture) -> BoxFuture<Result<Outcome<Turn>, CompletionError>> {
        Box::pin(async move {
            match case {
                Fixture::MessageIdPreservation => Ok(Self::expected(case)),
                _ => {
                    let client = crate::providers::anthropic::Client::builder()
                        .http_client(MockStreamingClient {
                            sse_bytes: Bytes::from(streaming_sse(case)),
                        })
                        .api_key("test-key")
                        .build()
                        .expect("client should build");
                    let model = CompletionModel::new(client, "claude-test");
                    let stream = model.stream(stream_request()).await?;
                    let response = drain_stream(stream).await?;
                    Ok(Outcome::Supported(normalize_completion_response(
                        &response, None,
                    )))
                }
            }
        })
    }
}

fn non_stream_response(case: Fixture) -> CompletionResponse {
    match case {
        Fixture::EmptyAssistantTurnAfterToolResult => response_with_content(vec![], "end_turn"),
        Fixture::ToolOnlyTurn => response_with_content(
            vec![Content::ToolUse {
                id: "toolu_lookup".to_string(),
                name: "lookup_weather".to_string(),
                input: json!({"city": "Paris"}),
            }],
            "tool_use",
        ),
        Fixture::TextAndToolCallTurn => response_with_content(
            vec![
                Content::Text {
                    text: "Need weather data first.".to_string(),
                    cache_control: None,
                },
                Content::ToolUse {
                    id: "toolu_lookup".to_string(),
                    name: "lookup_weather".to_string(),
                    input: json!({"city": "Paris"}),
                },
            ],
            "tool_use",
        ),
        Fixture::EmptyTextBlocks => response_with_content(
            vec![Content::Text {
                text: String::new(),
                cache_control: None,
            }],
            "end_turn",
        ),
        Fixture::ReasoningOnlyTurn => response_with_content(
            vec![Content::Thinking {
                thinking: "Need to reason about the tool result.".to_string(),
                signature: Some("sig_1".to_string()),
            }],
            "end_turn",
        ),
        Fixture::StopReasonMapping => response_with_content(
            vec![Content::Text {
                text: "Truncated response".to_string(),
                cache_control: None,
            }],
            "max_tokens",
        ),
        Fixture::MessageIdPreservation => unreachable!(),
    }
}

fn response_with_content(content: Vec<Content>, stop_reason: &str) -> CompletionResponse {
    CompletionResponse {
        content,
        id: "msg_123".to_string(),
        model: "claude-test".to_string(),
        role: "assistant".to_string(),
        stop_reason: Some(stop_reason.to_string()),
        stop_sequence: None,
        usage: Usage {
            input_tokens: 10,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            output_tokens: 5,
        },
    }
}

fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolCalls,
        "max_tokens" => StopReason::MaxTokens,
        other => StopReason::Other(other.to_string()),
    }
}

fn stream_request() -> completion::CompletionRequest {
    completion::CompletionRequest {
        model: None,
        preamble: None,
        chat_history: OneOrMany::one(completion::Message::user("hello")),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: Some(32),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
    }
}

fn streaming_sse(case: Fixture) -> String {
    match case {
        Fixture::EmptyAssistantTurnAfterToolResult => concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-test\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":null,\"cache_creation_input_tokens\":null,\"output_tokens\":0}}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":0}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        )
        .to_string(),
        Fixture::ToolOnlyTurn => concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-test\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":null,\"cache_creation_input_tokens\":null,\"output_tokens\":0}}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_lookup\",\"name\":\"lookup_weather\",\"input\":{\"city\":\"Paris\"}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":5}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        )
        .to_string(),
        Fixture::TextAndToolCallTurn => concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-test\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":null,\"cache_creation_input_tokens\":null,\"output_tokens\":0}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Need weather data first.\"}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_lookup\",\"name\":\"lookup_weather\",\"input\":{\"city\":\"Paris\"}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":7}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        )
        .to_string(),
        Fixture::EmptyTextBlocks => concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-test\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":null,\"cache_creation_input_tokens\":null,\"output_tokens\":0}}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":0}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        )
        .to_string(),
        Fixture::ReasoningOnlyTurn => concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-test\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":null,\"cache_creation_input_tokens\":null,\"output_tokens\":0}}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":null}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Need to reason about the tool result.\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_1\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":5}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        )
        .to_string(),
        Fixture::StopReasonMapping => concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-test\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":null,\"cache_creation_input_tokens\":null,\"output_tokens\":0}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Truncated response\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":2}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        )
        .to_string(),
        Fixture::MessageIdPreservation => unreachable!(),
    }
}

provider_conformance_tests!(AnthropicHarness);
