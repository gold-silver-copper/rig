//! Internal normalization helpers for provider responses.
//!
//! Providers often expose different stop-reason enums and content block shapes.
//! This module defines a small shared semantic representation that can be used by
//! higher-level code, test harnesses, and future provider adapters without
//! depending on provider-specific wire types.

use serde_json::Value;

use super::{AssistantChoice, CompletionResponse};
use crate::message::AssistantContent;

/// Provider-agnostic assistant content used by internal normalization flows.
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

/// Provider-agnostic reasons why a provider stopped generating a turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// The model reached a natural stop point.
    Stop,
    /// The model stopped in order to request tool execution.
    ToolCalls,
    /// The provider explicitly ended the assistant turn.
    EndTurn,
    /// The model hit the configured output-token limit.
    MaxTokens,
    /// The provider filtered the output content.
    ContentFilter,
    /// The provider stopped for a safety-related reason.
    Safety,
    /// The provider reported a stop reason Rig does not normalize yet.
    Other(String),
}

/// Provider-agnostic normalized assistant turn.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct NormalizedTurn {
    pub(crate) items: Vec<NormalizedItem>,
    pub(crate) message_id: Option<String>,
    pub(crate) stop_reason: Option<StopReason>,
}

impl NormalizedTurn {
    /// Build a normalized turn from a normalized assistant choice plus metadata.
    pub(crate) fn from_choice(
        choice: &AssistantChoice,
        message_id: Option<String>,
        stop_reason: Option<StopReason>,
    ) -> Self {
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

        Self {
            items,
            message_id,
            stop_reason,
        }
    }

    /// Build a normalized turn from a completion response.
    pub(crate) fn from_completion_response<T>(response: &CompletionResponse<T>) -> Self {
        Self::from_choice(
            &response.choice,
            response.message_id.clone(),
            response.stop_reason.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{NormalizedItem, NormalizedTurn, StopReason};
    use crate::completion::AssistantChoice;
    use crate::message::{AssistantContent, Reasoning};

    #[test]
    fn normalized_turn_ignores_empty_text_blocks() {
        let choice = AssistantChoice::many(vec![
            AssistantContent::text("visible"),
            AssistantContent::text(""),
        ]);

        let turn = NormalizedTurn::from_choice(&choice, Some("msg_1".to_string()), None);

        assert_eq!(
            turn,
            NormalizedTurn {
                items: vec![NormalizedItem::Text("visible".to_string())],
                message_id: Some("msg_1".to_string()),
                stop_reason: None,
            }
        );
    }

    #[test]
    fn normalized_turn_deduplicates_adjacent_reasoning_blocks() {
        let choice = AssistantChoice::many(vec![
            AssistantContent::Reasoning(Reasoning::new("step one")),
            AssistantContent::Reasoning(Reasoning::new("step one")),
        ]);

        let turn = NormalizedTurn::from_choice(&choice, None, Some(StopReason::EndTurn));

        assert_eq!(
            turn.items,
            vec![NormalizedItem::Reasoning("step one".to_string())]
        );
        assert_eq!(turn.stop_reason, Some(StopReason::EndTurn));
    }
}
