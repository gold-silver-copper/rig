use futures::StreamExt;

use crate::completion::{self, CompletionError, GetTokenUsage};

pub(crate) use crate::completion::normalized::{
    NormalizedItem, NormalizedTurn as Turn, StopReason,
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
pub(crate) enum Outcome<T> {
    Supported(T),
    Unsupported(&'static str),
}

pub(crate) type BoxFuture<T> = crate::wasm_compat::WasmBoxedFuture<'static, T>;

pub(crate) trait Harness {
    fn family_name() -> &'static str;

    fn expected(case: Fixture) -> Outcome<Turn>;

    fn non_stream(case: Fixture) -> Result<Outcome<Turn>, CompletionError>;

    fn stream(case: Fixture) -> BoxFuture<Result<Outcome<Turn>, CompletionError>>;
}

pub(crate) fn normalize_completion_response<T>(
    response: &completion::CompletionResponse<T>,
) -> Turn {
    Turn::from_completion_response(response)
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
            assert_eq!(
                actual.stop_reason,
                expected.stop_reason,
                "{} stream {:?} stop_reason diverged",
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
