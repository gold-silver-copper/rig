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

use super::gemini_api_types::{Content, ContentCandidate, FinishReason, Part, PartKind, Role};

struct GeminiHarness;

impl Harness for GeminiHarness {
    fn family_name() -> &'static str {
        "gemini-generate-content"
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
                    id: "lookup_weather".to_string(),
                    name: "lookup_weather".to_string(),
                    arguments: json!({"city": "Paris"}),
                }],
                message_id: None,
                stop_reason: Some(StopReason::Stop),
            }),
            Fixture::TextAndToolCallTurn => Outcome::Supported(Turn {
                items: vec![
                    NormalizedItem::Text("Need weather data first.".to_string()),
                    NormalizedItem::ToolCall {
                        id: "lookup_weather".to_string(),
                        name: "lookup_weather".to_string(),
                        arguments: json!({"city": "Paris"}),
                    },
                ],
                message_id: None,
                stop_reason: Some(StopReason::Stop),
            }),
            Fixture::EmptyTextBlocks => Outcome::Supported(Turn {
                items: vec![],
                message_id: None,
                stop_reason: Some(StopReason::Stop),
            }),
            Fixture::ReasoningOnlyTurn => Outcome::Supported(Turn {
                items: vec![NormalizedItem::Reasoning(
                    "Need to reason about the tool result.".to_string(),
                )],
                message_id: None,
                stop_reason: Some(StopReason::Stop),
            }),
            Fixture::MessageIdPreservation => {
                Outcome::Unsupported("Gemini GenerateContent responses do not expose message IDs")
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
                let response: completion::CompletionResponse<GenerateContentResponse> =
                    raw.try_into()?;
                Ok(Outcome::Supported(normalize_completion_response(&response)))
            }
        }
    }

    fn stream(case: Fixture) -> BoxFuture<Result<Outcome<Turn>, CompletionError>> {
        Box::pin(async move {
            match case {
                Fixture::MessageIdPreservation => Ok(Self::expected(case)),
                _ => {
                    let client = crate::providers::gemini::Client::builder()
                        .http_client(MockStreamingClient {
                            sse_bytes: Bytes::from(streaming_sse(case)),
                        })
                        .api_key("test-key")
                        .build()
                        .expect("client should build");
                    let model = CompletionModel::new(client, "gemini-test");
                    let stream = model.stream(stream_request()).await?;
                    let response = drain_stream(stream).await?;
                    Ok(Outcome::Supported(normalize_completion_response(&response)))
                }
            }
        })
    }
}

fn non_stream_response(case: Fixture) -> GenerateContentResponse {
    let candidate = match case {
        Fixture::EmptyAssistantTurnAfterToolResult => candidate(vec![], FinishReason::Stop),
        Fixture::ToolOnlyTurn => candidate(
            vec![Part {
                thought: None,
                thought_signature: None,
                part: PartKind::FunctionCall(gemini_api_types::FunctionCall {
                    name: "lookup_weather".to_string(),
                    args: json!({"city": "Paris"}),
                }),
                additional_params: None,
            }],
            FinishReason::Stop,
        ),
        Fixture::TextAndToolCallTurn => candidate(
            vec![
                Part {
                    thought: Some(false),
                    thought_signature: None,
                    part: PartKind::Text("Need weather data first.".to_string()),
                    additional_params: None,
                },
                Part {
                    thought: None,
                    thought_signature: None,
                    part: PartKind::FunctionCall(gemini_api_types::FunctionCall {
                        name: "lookup_weather".to_string(),
                        args: json!({"city": "Paris"}),
                    }),
                    additional_params: None,
                },
            ],
            FinishReason::Stop,
        ),
        Fixture::EmptyTextBlocks => candidate(
            vec![Part {
                thought: Some(false),
                thought_signature: None,
                part: PartKind::Text(String::new()),
                additional_params: None,
            }],
            FinishReason::Stop,
        ),
        Fixture::ReasoningOnlyTurn => candidate(
            vec![Part {
                thought: Some(true),
                thought_signature: Some("sig_1".to_string()),
                part: PartKind::Text("Need to reason about the tool result.".to_string()),
                additional_params: None,
            }],
            FinishReason::Stop,
        ),
        Fixture::StopReasonMapping => candidate(
            vec![Part {
                thought: Some(false),
                thought_signature: None,
                part: PartKind::Text("Truncated response".to_string()),
                additional_params: None,
            }],
            FinishReason::MaxTokens,
        ),
        Fixture::MessageIdPreservation => unreachable!(),
    };

    GenerateContentResponse {
        response_id: "resp_123".to_string(),
        candidates: vec![candidate],
        prompt_feedback: None,
        usage_metadata: None,
        model_version: None,
    }
}

fn candidate(parts: Vec<Part>, finish_reason: FinishReason) -> ContentCandidate {
    ContentCandidate {
        content: Some(Content {
            parts,
            role: Some(Role::Model),
        }),
        finish_reason: Some(finish_reason),
        safety_ratings: None,
        citation_metadata: None,
        token_count: None,
        avg_logprobs: None,
        logprobs_result: None,
        index: Some(0),
        finish_message: None,
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
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
    }
}

fn streaming_sse(case: Fixture) -> String {
    match case {
        Fixture::EmptyAssistantTurnAfterToolResult => concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[],\"role\":\"model\"},\"finishReason\":\"STOP\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":0,\"totalTokenCount\":10}}\n\n",
        )
        .to_string(),
        Fixture::ToolOnlyTurn => concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"lookup_weather\",\"args\":{\"city\":\"Paris\"}}}],\"role\":\"model\"},\"finishReason\":\"STOP\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":5,\"totalTokenCount\":15}}\n\n",
        )
        .to_string(),
        Fixture::TextAndToolCallTurn => concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Need weather data first.\"},{\"functionCall\":{\"name\":\"lookup_weather\",\"args\":{\"city\":\"Paris\"}}}],\"role\":\"model\"},\"finishReason\":\"STOP\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":7,\"totalTokenCount\":17}}\n\n",
        )
        .to_string(),
        Fixture::EmptyTextBlocks => concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"\"}],\"role\":\"model\"},\"finishReason\":\"STOP\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":0,\"totalTokenCount\":10}}\n\n",
        )
        .to_string(),
        Fixture::ReasoningOnlyTurn => concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Need to reason about the tool result.\",\"thought\":true,\"thoughtSignature\":\"sig_1\"}],\"role\":\"model\"},\"finishReason\":\"STOP\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":5,\"thoughtsTokenCount\":3,\"totalTokenCount\":18}}\n\n",
        )
        .to_string(),
        Fixture::StopReasonMapping => concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Truncated response\"}],\"role\":\"model\"},\"finishReason\":\"MAX_TOKENS\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":2,\"totalTokenCount\":12}}\n\n",
        )
        .to_string(),
        Fixture::MessageIdPreservation => unreachable!(),
    }
}

provider_conformance_tests!(GeminiHarness);
