//! Shared helpers for deriving user-visible assistant text across agent turns.

use crate::{OneOrMany, message::AssistantContent};

/// Summary of the visible assistant text in a normalized turn.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct AssistantTurnSummary {
    visible_text_blocks: Vec<String>,
}

impl AssistantTurnSummary {
    /// Extract non-empty text blocks from a normalized assistant choice.
    pub(crate) fn from_choice(choice: &OneOrMany<AssistantContent>) -> Self {
        let visible_text_blocks = choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) if !text.text.is_empty() => Some(text.text.clone()),
                _ => None,
            })
            .collect();

        Self {
            visible_text_blocks,
        }
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
    use crate::{OneOrMany, message::AssistantContent};

    #[test]
    fn summary_ignores_empty_text_blocks() {
        let choice = OneOrMany::many(vec![
            AssistantContent::text("visible"),
            AssistantContent::text(""),
        ])
        .expect("non-empty assistant choice");

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
