//! Shared helpers for deriving user-visible assistant text across agent turns.

use crate::completion::normalized::NormalizedTurn;
use crate::completion::{AssistantChoice, CompletionResponse, normalized::NormalizedItem};
use crate::streaming::StreamingCompletionResponse;

/// Summary of the visible assistant text in a normalized turn.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct AssistantTurnSummary {
    visible_text_blocks: Vec<String>,
}

impl AssistantTurnSummary {
    fn from_turn(turn: NormalizedTurn) -> Self {
        let visible_text_blocks = turn
            .items
            .into_iter()
            .filter_map(|item| match item {
                NormalizedItem::Text(text) => Some(text),
                _ => None,
            })
            .collect();

        Self {
            visible_text_blocks,
        }
    }

    /// Extract non-empty text blocks from a normalized assistant choice.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_choice(choice: &AssistantChoice) -> Self {
        Self::from_turn(NormalizedTurn::from_choice(choice, None, None))
    }

    /// Extract non-empty text blocks from a normalized completion response.
    pub(crate) fn from_response<T>(response: &CompletionResponse<T>) -> Self {
        Self::from_turn(NormalizedTurn::from_completion_response(response))
    }

    /// Extract non-empty text blocks from an aggregated streaming response.
    pub(crate) fn from_stream_response<R>(response: &StreamingCompletionResponse<R>) -> Self
    where
        R: Clone + Unpin + crate::completion::GetTokenUsage,
    {
        Self::from_turn(NormalizedTurn::from_choice(
            &response.choice,
            response.message_id.clone(),
            response.stop_reason.clone(),
        ))
    }

    /// Render the visible text blocks using the caller's preferred separator.
    pub(crate) fn visible_text(&self, separator: &str) -> String {
        self.visible_text_blocks.join(separator)
    }
}

/// Tracks non-empty assistant text from earlier turns so a textless final turn
/// can still return the last user-visible answer the model produced.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct AssistantTextAccumulator {
    prior_turn_texts: Vec<String>,
}

impl AssistantTextAccumulator {
    /// Record a turn's visible text when it is not empty.
    pub(crate) fn observe(&mut self, turn_text: &str) {
        if !turn_text.is_empty() {
            self.prior_turn_texts.push(turn_text.to_owned());
        }
    }

    /// Prefer the current turn's visible text and otherwise fall back to the
    /// accumulated text from earlier turns in the same request.
    pub(crate) fn final_output(&self, current_turn_text: &str) -> String {
        if current_turn_text.is_empty() {
            self.prior_turn_texts.join("\n")
        } else {
            current_turn_text.to_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AssistantTextAccumulator, AssistantTurnSummary};
    use crate::{completion::AssistantChoice, message::AssistantContent};

    #[test]
    fn summary_ignores_empty_text_blocks() {
        let choice = AssistantChoice::many(vec![
            AssistantContent::text("visible"),
            AssistantContent::text(""),
        ]);

        let summary = AssistantTurnSummary::from_choice(&choice);

        assert_eq!(summary.visible_text("\n"), "visible");
    }

    #[test]
    fn accumulator_falls_back_when_terminal_turn_is_empty() {
        let mut accumulator = AssistantTextAccumulator::default();
        accumulator.observe("first turn");

        assert_eq!(accumulator.final_output(""), "first turn");
    }
}
