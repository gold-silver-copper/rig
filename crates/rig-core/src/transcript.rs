//! Conversation validation, repair of unanswered tool calls, and constructors
//! for real or synthetic tool results.
//!
//! ```
//! use rig_core::{message::Message, transcript::validate_canonical};
//!
//! validate_canonical(&[Message::user("Hello"), Message::assistant("Hi")])?;
//! # Ok::<(), rig_core::transcript::TranscriptError>(())
//! ```

use std::collections::BTreeSet;

use crate::message::{
    AssistantContent, CallId, Message, ToolCall, ToolName, ToolResultContent, UserContent,
};
use crate::tool::ToolOutput;

/// Why a history is not a canonical transcript. See [`validate_canonical`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TranscriptError {
    /// Two assistant messages in a row (index of the second).
    #[error("consecutive assistant messages at index {index}")]
    ConsecutiveAssistant {
        /// Index of the offending (second) assistant message.
        index: usize,
    },
    /// An assistant tool call whose result is not in the next message.
    #[error("tool call `{call_id}` at index {index} has no result in the following message")]
    UnansweredToolCall {
        /// Index of the assistant message carrying the call.
        index: usize,
        /// The unanswered call id.
        call_id: CallId,
    },
    /// A tool result that answers no call from the immediately preceding
    /// assistant message.
    #[error(
        "tool result `{call_id}` at index {index} answers no call from the preceding assistant message"
    )]
    OrphanToolResult {
        /// Index of the user message carrying the result.
        index: usize,
        /// The orphan result's call id.
        call_id: CallId,
    },
}

/// Rejects consecutive assistant messages, unanswered tool-call IDs, and results
/// without a pending call. Each pending ID must be answered once in the next user
/// message, before another assistant message or the end of history.
/// System messages reset the consecutive-assistant check but retain pending calls.
/// Duplicate call IDs are treated as one pending ID.
pub fn validate_canonical(messages: &[Message]) -> Result<(), TranscriptError> {
    let mut prev_assistant_calls: Option<BTreeSet<CallId>> = None;
    let mut prev_was_assistant = false;
    for (index, message) in messages.iter().enumerate() {
        match message {
            Message::Assistant { content, .. } => {
                if prev_was_assistant {
                    return Err(TranscriptError::ConsecutiveAssistant { index });
                }
                if let Some(call_id) = prev_assistant_calls
                    .take()
                    .and_then(|pending| pending.into_iter().next())
                {
                    return Err(TranscriptError::UnansweredToolCall {
                        index: index - 1,
                        call_id,
                    });
                }
                let calls: BTreeSet<CallId> = content
                    .iter()
                    .filter_map(|c| match c {
                        AssistantContent::ToolCall(call) => Some(call.id.clone()),
                        _ => None,
                    })
                    .collect();
                prev_assistant_calls = (!calls.is_empty()).then_some(calls);
                prev_was_assistant = true;
            }
            Message::User { content } => {
                let mut pending = prev_assistant_calls.take().unwrap_or_default();
                for item in content.iter() {
                    if let UserContent::ToolResult(result) = item {
                        let id = result.call.clone();
                        if !pending.remove(&id) {
                            return Err(TranscriptError::OrphanToolResult { index, call_id: id });
                        }
                    }
                }
                if let Some(call_id) = pending.into_iter().next() {
                    return Err(TranscriptError::UnansweredToolCall {
                        index: index.saturating_sub(1),
                        call_id,
                    });
                }
                prev_was_assistant = false;
            }
            Message::System { .. } => {
                prev_was_assistant = false;
            }
        }
    }
    if let Some(call_id) = prev_assistant_calls.and_then(|pending| pending.into_iter().next()) {
        return Err(TranscriptError::UnansweredToolCall {
            index: messages.len().saturating_sub(1),
            call_id,
        });
    }
    Ok(())
}

/// Answers every tool call that has no result in the message after it, so a
/// history cut short mid-turn (a crash, a restart, a cancelled run) becomes
/// canonical again. `reason` supplies the synthetic result text for each
/// unanswered call. Missing results join the tool results of the following
/// user message, or a new user message when the next message is not one.
/// Returns how many results were added. Orphan results are left alone.
///
/// ```
/// use rig_core::message::{AssistantContent, Message, ToolName};
/// use rig_core::transcript::{answer_unanswered, validate_canonical};
///
/// let call = AssistantContent::tool_call("c1", ToolName::new("bash")?, serde_json::json!({}));
/// let mut history = vec![
///     Message::user("run it"),
///     Message::Assistant { id: None, content: vec![call] },
/// ];
/// assert_eq!(answer_unanswered(&mut history, |_| "interrupted".into()), 1);
/// validate_canonical(&history)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn answer_unanswered<F>(messages: &mut Vec<Message>, mut reason: F) -> usize
where
    F: FnMut(&ToolCall) -> String,
{
    let mut added = 0;
    let mut index = 0;
    while let Some(message) = messages.get(index) {
        let mut calls: Vec<ToolCall> = Vec::new();
        if let Message::Assistant { content, .. } = message {
            for part in content {
                if let AssistantContent::ToolCall(call) = part
                    && !calls.iter().any(|seen| seen.id == call.id)
                {
                    calls.push(call.clone());
                }
            }
        }
        index += 1;
        if calls.is_empty() {
            continue;
        }
        let answered: BTreeSet<CallId> = match messages.get(index) {
            Some(Message::User { content }) => content
                .iter()
                .filter_map(|part| match part {
                    UserContent::ToolResult(result) => Some(result.call.clone()),
                    _ => None,
                })
                .collect(),
            _ => BTreeSet::new(),
        };
        let missing: Vec<UserContent> = calls
            .iter()
            .filter(|call| !answered.contains(&call.id))
            .map(|call| {
                tool_result_message(call.id.clone(), call.function.name.clone(), reason(call))
            })
            .collect();
        if missing.is_empty() {
            continue;
        }
        added += missing.len();
        match messages.get_mut(index) {
            // Results lead the message: some providers reject text before them.
            Some(Message::User { content }) => {
                let results = content
                    .iter()
                    .take_while(|part| matches!(part, UserContent::ToolResult(_)))
                    .count();
                content.splice(results..results, missing);
            }
            _ => messages.insert(index, Message::User { content: missing }),
        }
    }
    added
}

/// Shape a canonical real tool output as a tool result without reparsing text.
pub fn tool_result_output(call: CallId, name: ToolName, output: ToolOutput) -> UserContent {
    UserContent::tool_result(call, name, output.into_content())
}

/// Constructs a synthetic tool result containing verbatim text, such as recovery
/// feedback or a skip reason. JSON-shaped text is not reinterpreted as structured
/// or multimodal output.
pub fn tool_result_message(call: CallId, name: ToolName, message: String) -> UserContent {
    UserContent::tool_result(call, name, vec![ToolResultContent::text(message)])
}

#[cfg(test)]
mod validator_tests;
