use crate::{
    OneOrMany,
    message::{AssistantContent, Reasoning, ReasoningContent, ToolCall},
};

use super::{RawStreamingChoice, StreamedAssistantContent};

pub(super) enum AggregationOutcome<R> {
    Emit(StreamedAssistantContent<R>),
    FinalResponse(R),
    MessageId(String),
}

pub(super) struct StreamingAggregation {
    assistant_items: Vec<AssistantContent>,
    text_item_index: Option<usize>,
    reasoning_item_index: Option<usize>,
}

impl StreamingAggregation {
    pub(super) fn new() -> Self {
        Self {
            assistant_items: Vec::new(),
            text_item_index: None,
            reasoning_item_index: None,
        }
    }

    pub(super) fn push<R>(&mut self, choice: RawStreamingChoice<R>) -> AggregationOutcome<R>
    where
        R: Clone + Unpin,
    {
        match choice {
            RawStreamingChoice::Message(text) => {
                self.reasoning_item_index = None;
                self.append_text_chunk(&text);
                AggregationOutcome::Emit(StreamedAssistantContent::text(&text))
            }
            RawStreamingChoice::ToolCallDelta {
                id,
                internal_call_id,
                content,
            } => AggregationOutcome::Emit(StreamedAssistantContent::ToolCallDelta {
                id,
                internal_call_id,
                content,
            }),
            RawStreamingChoice::Reasoning { id, content } => {
                let reasoning = Reasoning {
                    id,
                    content: vec![content],
                };
                self.text_item_index = None;
                self.reasoning_item_index = None;
                self.assistant_items
                    .push(AssistantContent::Reasoning(reasoning.clone()));
                AggregationOutcome::Emit(StreamedAssistantContent::Reasoning(reasoning))
            }
            RawStreamingChoice::ReasoningDelta { id, reasoning } => {
                self.text_item_index = None;
                self.append_reasoning_chunk(&id, &reasoning);
                AggregationOutcome::Emit(StreamedAssistantContent::ReasoningDelta { id, reasoning })
            }
            RawStreamingChoice::ToolCall(raw_tool_call) => {
                let internal_call_id = raw_tool_call.internal_call_id.clone();
                let tool_call: ToolCall = raw_tool_call.into();
                self.text_item_index = None;
                self.reasoning_item_index = None;
                self.assistant_items
                    .push(AssistantContent::ToolCall(tool_call.clone()));
                AggregationOutcome::Emit(StreamedAssistantContent::ToolCall {
                    tool_call,
                    internal_call_id,
                })
            }
            RawStreamingChoice::FinalResponse(response) => {
                AggregationOutcome::FinalResponse(response)
            }
            RawStreamingChoice::MessageId(id) => AggregationOutcome::MessageId(id),
        }
    }

    pub(super) fn finish(&mut self) -> OneOrMany<AssistantContent> {
        if self.assistant_items.is_empty() {
            self.assistant_items.push(AssistantContent::text(""));
        }

        OneOrMany::from_iter_optional(std::mem::take(&mut self.assistant_items))
            .unwrap_or_else(|| OneOrMany::one(AssistantContent::text("")))
    }

    fn append_text_chunk(&mut self, text: &str) {
        if let Some(index) = self.text_item_index
            && let Some(AssistantContent::Text(existing_text)) = self.assistant_items.get_mut(index)
        {
            existing_text.text.push_str(text);
            return;
        }

        self.assistant_items
            .push(AssistantContent::text(text.to_owned()));
        self.text_item_index = Some(self.assistant_items.len() - 1);
    }

    fn append_reasoning_chunk(&mut self, id: &Option<String>, text: &str) {
        if let Some(index) = self.reasoning_item_index
            && let Some(AssistantContent::Reasoning(existing)) = self.assistant_items.get_mut(index)
            && let Some(ReasoningContent::Text {
                text: existing_text,
                ..
            }) = existing.content.last_mut()
        {
            existing_text.push_str(text);
            return;
        }

        self.assistant_items
            .push(AssistantContent::Reasoning(Reasoning {
                id: id.clone(),
                content: vec![ReasoningContent::Text {
                    text: text.to_string(),
                    signature: None,
                }],
            }));
        self.reasoning_item_index = Some(self.assistant_items.len() - 1);
    }
}

#[cfg(test)]
mod tests {
    use crate::message::{Text, ToolFunction};

    use super::*;
    use crate::streaming::{RawStreamingToolCall, ToolCallDeltaContent};

    #[derive(Clone, Debug, PartialEq)]
    struct MockResponse;

    fn finish_items(aggregation: &mut StreamingAggregation) -> Vec<AssistantContent> {
        aggregation.finish().into_iter().collect()
    }

    #[test]
    fn adjacent_text_deltas_merge() {
        let mut aggregation = StreamingAggregation::new();

        let _: AggregationOutcome<MockResponse> =
            aggregation.push(RawStreamingChoice::Message("hello".to_string()));
        let _: AggregationOutcome<MockResponse> =
            aggregation.push(RawStreamingChoice::Message(" world".to_string()));

        assert_eq!(
            finish_items(&mut aggregation),
            vec![AssistantContent::Text(Text {
                text: "hello world".to_string()
            })]
        );
    }

    #[test]
    fn tool_calls_split_text_runs() {
        let mut aggregation = StreamingAggregation::new();

        let _: AggregationOutcome<MockResponse> =
            aggregation.push(RawStreamingChoice::Message("first".to_string()));
        let _: AggregationOutcome<MockResponse> =
            aggregation.push(RawStreamingChoice::ToolCall(RawStreamingToolCall::new(
                "call_1".to_string(),
                "search".to_string(),
                serde_json::json!({"q": "rust"}),
            )));
        let _: AggregationOutcome<MockResponse> =
            aggregation.push(RawStreamingChoice::Message("second".to_string()));

        let items = finish_items(&mut aggregation);
        assert!(matches!(
            items.as_slice(),
            [
                AssistantContent::Text(Text { text: first }),
                AssistantContent::ToolCall(ToolCall {
                    function: ToolFunction { name, .. },
                    ..
                }),
                AssistantContent::Text(Text { text: second })
            ] if first == "first" && name == "search" && second == "second"
        ));
    }

    #[test]
    fn reasoning_deltas_merge_until_text_arrives() {
        let mut aggregation = StreamingAggregation::new();

        let _: AggregationOutcome<MockResponse> =
            aggregation.push(RawStreamingChoice::ReasoningDelta {
                id: Some("rs_1".to_string()),
                reasoning: "step ".to_string(),
            });
        let _: AggregationOutcome<MockResponse> =
            aggregation.push(RawStreamingChoice::ReasoningDelta {
                id: Some("rs_1".to_string()),
                reasoning: "one".to_string(),
            });
        let _: AggregationOutcome<MockResponse> =
            aggregation.push(RawStreamingChoice::Message("answer".to_string()));

        let items = finish_items(&mut aggregation);
        assert_eq!(items.len(), 2);
        assert!(matches!(
            &items[0],
            AssistantContent::Reasoning(Reasoning { content, .. })
                if matches!(
                    content.as_slice(),
                    [ReasoningContent::Text { text, .. }] if text == "step one"
                )
        ));
        assert!(matches!(
            &items[1],
            AssistantContent::Text(Text { text }) if text == "answer"
        ));
    }

    #[test]
    fn empty_stream_finishes_with_empty_text() {
        let mut aggregation = StreamingAggregation::new();

        assert_eq!(
            finish_items(&mut aggregation),
            vec![AssistantContent::Text(Text {
                text: String::new()
            })]
        );
    }

    #[test]
    fn control_events_do_not_change_final_items() {
        let mut aggregation = StreamingAggregation::new();

        assert!(matches!(
            aggregation.push(RawStreamingChoice::MessageId::<MockResponse>("msg_1".to_string())),
            AggregationOutcome::MessageId(id) if id == "msg_1"
        ));
        assert!(matches!(
            aggregation.push(RawStreamingChoice::FinalResponse(MockResponse)),
            AggregationOutcome::FinalResponse(MockResponse)
        ));

        assert_eq!(
            finish_items(&mut aggregation),
            vec![AssistantContent::Text(Text {
                text: String::new()
            })]
        );
    }

    #[test]
    fn tool_call_delta_is_emitted_without_final_aggregation() {
        let mut aggregation = StreamingAggregation::new();

        assert!(matches!(
            aggregation.push(RawStreamingChoice::ToolCallDelta::<MockResponse> {
                id: "provider_call".to_string(),
                internal_call_id: "internal_call".to_string(),
                content: ToolCallDeltaContent::Name("lookup".to_string()),
            }),
            AggregationOutcome::Emit(StreamedAssistantContent::ToolCallDelta {
                id,
                internal_call_id,
                content: ToolCallDeltaContent::Name(name),
            }) if id == "provider_call"
                && internal_call_id == "internal_call"
                && name == "lookup"
        ));

        assert_eq!(
            finish_items(&mut aggregation),
            vec![AssistantContent::Text(Text {
                text: String::new()
            })]
        );
    }
}
