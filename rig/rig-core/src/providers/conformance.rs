use std::{future::Future, pin::Pin};

use futures::StreamExt;
use serde_json::Value;

use crate::{
    OneOrMany,
    completion::{self, CompletionError, GetTokenUsage},
    message::AssistantContent,
};

#[derive(Debug, Clone, Copy)]
pub(crate) enum Fixture {
    EmptyAssistantTurnAfterToolResult,
    ToolOnlyTurn,
    TextAndToolCallTurn,
    EmptyTextBlocks,
    ReasoningOnlyTurn,
    MessageIdPreservation,
    StopReasonMapping,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum NormalizedItem {
    Text(String),
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
    Reasoning(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StopReason {
    Stop,
    ToolCalls,
    EndTurn,
    MaxTokens,
    ContentFilter,
    Safety,
    Other(String),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Turn {
    pub(crate) items: Vec<NormalizedItem>,
    pub(crate) message_id: Option<String>,
    pub(crate) stop_reason: Option<StopReason>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Outcome<T> {
    Supported(T),
    Unsupported(&'static str),
}

pub(crate) type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

pub(crate) trait Harness {
    fn family_name() -> &'static str;

    fn expected(case: Fixture) -> Outcome<Turn>;

    fn non_stream(case: Fixture) -> Result<Outcome<Turn>, CompletionError>;

    fn stream(case: Fixture) -> BoxFuture<Result<Outcome<Turn>, CompletionError>>;
}

pub(crate) fn normalize_turn(
    choice: &OneOrMany<AssistantContent>,
    message_id: Option<String>,
    stop_reason: Option<StopReason>,
) -> Turn {
    let mut items = Vec::new();

    for item in choice.iter() {
        let normalized = match item {
            AssistantContent::Text(text) if !text.text.is_empty() => {
                Some(NormalizedItem::Text(text.text.clone()))
            }
            AssistantContent::ToolCall(tool_call) => Some(NormalizedItem::ToolCall {
                id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                arguments: tool_call.function.arguments.clone(),
            }),
            AssistantContent::Reasoning(reasoning) => {
                let text = reasoning.display_text();
                if text.is_empty() {
                    None
                } else {
                    Some(NormalizedItem::Reasoning(text))
                }
            }
            _ => None,
        };

        if let Some(normalized) = normalized {
            let duplicate_reasoning = matches!(
                (&normalized, items.last()),
                (NormalizedItem::Reasoning(current), Some(NormalizedItem::Reasoning(previous)))
                    if current == previous
            );

            if !duplicate_reasoning {
                items.push(normalized);
            }
        }
    }

    Turn {
        items,
        message_id,
        stop_reason,
    }
}

pub(crate) fn normalize_completion_response<T>(
    response: &completion::CompletionResponse<T>,
    stop_reason: Option<StopReason>,
) -> Turn {
    normalize_turn(&response.choice, response.message_id.clone(), stop_reason)
}

pub(crate) async fn drain_stream<R>(
    mut stream: crate::streaming::StreamingCompletionResponse<R>,
) -> Result<completion::CompletionResponse<Option<R>>, CompletionError>
where
    R: Clone + Unpin + GetTokenUsage,
{
    while let Some(item) = stream.next().await {
        item?;
    }

    Ok(stream.into())
}

pub(crate) fn assert_non_stream_case<H: Harness>(case: Fixture) {
    let expected = H::expected(case);
    let actual = H::non_stream(case)
        .unwrap_or_else(|err| panic!("{} non-stream {:?} failed: {err}", H::family_name(), case));

    assert_eq!(
        actual,
        expected,
        "{} non-stream {:?} mismatch",
        H::family_name(),
        case
    );
}

pub(crate) async fn assert_stream_matches_non_stream<H: Harness>(case: Fixture) {
    let non_stream = H::non_stream(case)
        .unwrap_or_else(|err| panic!("{} non-stream {:?} failed: {err}", H::family_name(), case));
    let stream = H::stream(case)
        .await
        .unwrap_or_else(|err| panic!("{} stream {:?} failed: {err}", H::family_name(), case));

    match (non_stream, stream) {
        (Outcome::Supported(expected), Outcome::Supported(actual)) => {
            assert_eq!(
                actual.items,
                expected.items,
                "{} stream {:?} items diverged",
                H::family_name(),
                case
            );
            assert_eq!(
                actual.message_id,
                expected.message_id,
                "{} stream {:?} message_id diverged",
                H::family_name(),
                case
            );
        }
        (Outcome::Unsupported(_), Outcome::Unsupported(_)) => {}
        (expected, actual) => panic!(
            "{} stream/non-stream support mismatch for {:?}: non-stream={expected:?}, stream={actual:?}",
            H::family_name(),
            case
        ),
    }
}

macro_rules! provider_conformance_tests {
    ($harness:ty) => {
        #[test]
        fn conformance_empty_assistant_turn_after_tool_result_non_stream() {
            crate::providers::conformance::assert_non_stream_case::<$harness>(
                crate::providers::conformance::Fixture::EmptyAssistantTurnAfterToolResult,
            );
        }

        #[test]
        fn conformance_tool_only_turn_non_stream() {
            crate::providers::conformance::assert_non_stream_case::<$harness>(
                crate::providers::conformance::Fixture::ToolOnlyTurn,
            );
        }

        #[test]
        fn conformance_text_and_tool_call_turn_non_stream() {
            crate::providers::conformance::assert_non_stream_case::<$harness>(
                crate::providers::conformance::Fixture::TextAndToolCallTurn,
            );
        }

        #[test]
        fn conformance_empty_text_blocks_non_stream() {
            crate::providers::conformance::assert_non_stream_case::<$harness>(
                crate::providers::conformance::Fixture::EmptyTextBlocks,
            );
        }

        #[test]
        fn conformance_reasoning_only_turn_non_stream() {
            crate::providers::conformance::assert_non_stream_case::<$harness>(
                crate::providers::conformance::Fixture::ReasoningOnlyTurn,
            );
        }

        #[test]
        fn conformance_message_id_preservation_non_stream() {
            crate::providers::conformance::assert_non_stream_case::<$harness>(
                crate::providers::conformance::Fixture::MessageIdPreservation,
            );
        }

        #[test]
        fn conformance_stop_reason_mapping_non_stream() {
            crate::providers::conformance::assert_non_stream_case::<$harness>(
                crate::providers::conformance::Fixture::StopReasonMapping,
            );
        }

        #[tokio::test]
        async fn conformance_empty_assistant_turn_after_tool_result_stream_equivalence() {
            crate::providers::conformance::assert_stream_matches_non_stream::<$harness>(
                crate::providers::conformance::Fixture::EmptyAssistantTurnAfterToolResult,
            )
            .await;
        }

        #[tokio::test]
        async fn conformance_tool_only_turn_stream_equivalence() {
            crate::providers::conformance::assert_stream_matches_non_stream::<$harness>(
                crate::providers::conformance::Fixture::ToolOnlyTurn,
            )
            .await;
        }

        #[tokio::test]
        async fn conformance_text_and_tool_call_turn_stream_equivalence() {
            crate::providers::conformance::assert_stream_matches_non_stream::<$harness>(
                crate::providers::conformance::Fixture::TextAndToolCallTurn,
            )
            .await;
        }

        #[tokio::test]
        async fn conformance_empty_text_blocks_stream_equivalence() {
            crate::providers::conformance::assert_stream_matches_non_stream::<$harness>(
                crate::providers::conformance::Fixture::EmptyTextBlocks,
            )
            .await;
        }

        #[tokio::test]
        async fn conformance_reasoning_only_turn_stream_equivalence() {
            crate::providers::conformance::assert_stream_matches_non_stream::<$harness>(
                crate::providers::conformance::Fixture::ReasoningOnlyTurn,
            )
            .await;
        }

        #[tokio::test]
        async fn conformance_message_id_preservation_stream_equivalence() {
            crate::providers::conformance::assert_stream_matches_non_stream::<$harness>(
                crate::providers::conformance::Fixture::MessageIdPreservation,
            )
            .await;
        }
    };
}

pub(crate) use provider_conformance_tests;
