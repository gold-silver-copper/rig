use crate::completion::{self, CompletionError};

pub(crate) fn first_choice<T>(choices: &[T]) -> Result<&T, CompletionError> {
    choices
        .first()
        .ok_or_else(|| CompletionError::ResponseError("Response contained no choices".to_owned()))
}

pub(crate) fn map_finish_reason(reason: &str) -> completion::StopReason {
    match reason {
        "stop" => completion::StopReason::Stop,
        "tool_calls" => completion::StopReason::ToolCalls,
        "content_filter" => completion::StopReason::ContentFilter,
        "length" => completion::StopReason::MaxTokens,
        other => completion::StopReason::Other(other.to_string()),
    }
}

pub(crate) fn non_empty_text(text: impl AsRef<str>) -> Option<completion::AssistantContent> {
    let text = text.as_ref();
    if text.is_empty() {
        None
    } else {
        Some(completion::AssistantContent::text(text))
    }
}

pub(crate) fn build_completion_response<R, C>(
    raw_response: R,
    usage: completion::Usage,
    message_id: Option<String>,
    stop_reason: Option<completion::StopReason>,
    choice: C,
) -> completion::CompletionResponse<R>
where
    C: Into<completion::AssistantChoice>,
{
    completion::CompletionResponse {
        choice: choice.into(),
        usage,
        raw_response,
        message_id,
        stop_reason,
    }
}
