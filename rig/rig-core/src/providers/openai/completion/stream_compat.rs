use std::collections::HashMap;

use crate::streaming::{RawStreamingChoice, RawStreamingToolCall, ToolCallDeltaContent};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolCallConflictPolicy {
    KeepIndex,
    EvictDistinctIdAndName,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CompatibleStreamingToolCall<'a> {
    pub index: usize,
    pub id: Option<&'a str>,
    pub name: Option<&'a str>,
    pub arguments: Option<&'a str>,
}

pub(crate) fn apply_compatible_tool_call_deltas<'a, R>(
    tool_calls: &mut HashMap<usize, RawStreamingToolCall>,
    incoming: impl IntoIterator<Item = CompatibleStreamingToolCall<'a>>,
    conflict_policy: ToolCallConflictPolicy,
) -> Vec<RawStreamingChoice<R>>
where
    R: Clone,
{
    let mut events = Vec::new();

    for tool_call in incoming {
        let index = tool_call.index;

        if conflict_policy == ToolCallConflictPolicy::EvictDistinctIdAndName
            && should_evict_existing_tool_call(tool_calls.get(&index), tool_call.id, tool_call.name)
            && let Some(evicted) = tool_calls.remove(&index)
        {
            events.push(RawStreamingChoice::ToolCall(
                finalize_completed_streaming_tool_call(evicted),
            ));
        }

        let existing_tool_call = tool_calls
            .entry(index)
            .or_insert_with(RawStreamingToolCall::empty);

        if let Some(id) = tool_call.id
            && !id.is_empty()
        {
            existing_tool_call.id = id.to_owned();
        }

        if let Some(name) = tool_call.name
            && !name.is_empty()
        {
            existing_tool_call.name = name.to_owned();
            events.push(RawStreamingChoice::ToolCallDelta {
                id: existing_tool_call.id.clone(),
                internal_call_id: existing_tool_call.internal_call_id.clone(),
                content: ToolCallDeltaContent::Name(name.to_owned()),
            });
        }

        if let Some(chunk) = tool_call.arguments
            && !chunk.is_empty()
        {
            append_tool_call_arguments(&mut existing_tool_call.arguments, chunk);
            events.push(RawStreamingChoice::ToolCallDelta {
                id: existing_tool_call.id.clone(),
                internal_call_id: existing_tool_call.internal_call_id.clone(),
                content: ToolCallDeltaContent::Delta(chunk.to_owned()),
            });
        }
    }

    events
}

pub(crate) fn take_finalized_tool_calls<R>(
    tool_calls: &mut HashMap<usize, RawStreamingToolCall>,
) -> Vec<RawStreamingChoice<R>>
where
    R: Clone,
{
    std::mem::take(tool_calls)
        .into_values()
        .map(|tool_call| {
            RawStreamingChoice::ToolCall(finalize_completed_streaming_tool_call(tool_call))
        })
        .collect()
}

pub(crate) fn take_tool_calls<R>(
    tool_calls: &mut HashMap<usize, RawStreamingToolCall>,
) -> Vec<RawStreamingChoice<R>>
where
    R: Clone,
{
    std::mem::take(tool_calls)
        .into_values()
        .map(RawStreamingChoice::ToolCall)
        .collect()
}

pub(crate) fn finalize_completed_streaming_tool_call(
    mut tool_call: RawStreamingToolCall,
) -> RawStreamingToolCall {
    if tool_call.arguments.is_null() {
        tool_call.arguments = serde_json::Value::Object(serde_json::Map::new());
    }

    tool_call
}

fn should_evict_existing_tool_call(
    existing: Option<&RawStreamingToolCall>,
    new_id: Option<&str>,
    new_name: Option<&str>,
) -> bool {
    let Some(existing) = existing else {
        return false;
    };

    let Some(new_id) = new_id.filter(|id| !id.is_empty()) else {
        return false;
    };
    let Some(new_name) = new_name.filter(|name| !name.is_empty()) else {
        return false;
    };

    !existing.id.is_empty()
        && existing.id != new_id
        && !existing.name.is_empty()
        && existing.name != new_name
}

fn append_tool_call_arguments(arguments: &mut serde_json::Value, chunk: &str) {
    let current_arguments = match arguments {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(value) => value.clone(),
        ref value => value.to_string(),
    };
    let combined = format!("{current_arguments}{chunk}");

    if combined.trim_start().starts_with('{') && combined.trim_end().ends_with('}') {
        match serde_json::from_str(&combined) {
            Ok(parsed) => *arguments = parsed,
            Err(_) => *arguments = serde_json::Value::String(combined),
        }
    } else {
        *arguments = serde_json::Value::String(combined);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CompatibleStreamingToolCall, ToolCallConflictPolicy, apply_compatible_tool_call_deltas,
        finalize_completed_streaming_tool_call, take_finalized_tool_calls,
    };
    use crate::streaming::{RawStreamingChoice, RawStreamingToolCall};
    use std::collections::HashMap;

    #[test]
    fn evicts_distinct_tool_calls_that_reuse_the_same_index() {
        let mut tool_calls = HashMap::from([(
            0,
            RawStreamingToolCall {
                id: "call_1".to_owned(),
                internal_call_id: "internal_1".to_owned(),
                call_id: None,
                name: "weather".to_owned(),
                arguments: serde_json::json!({"city":"Paris"}),
                signature: None,
                additional_params: None,
            },
        )]);

        let events = apply_compatible_tool_call_deltas::<()>(
            &mut tool_calls,
            [CompatibleStreamingToolCall {
                index: 0,
                id: Some("call_2"),
                name: Some("time"),
                arguments: Some("{"),
            }],
            ToolCallConflictPolicy::EvictDistinctIdAndName,
        );

        assert!(
            matches!(events.first(), Some(RawStreamingChoice::ToolCall(tool_call)) if tool_call.id == "call_1")
        );
        assert_eq!(
            tool_calls.get(&0).map(|tool_call| tool_call.id.as_str()),
            Some("call_2")
        );
        assert_eq!(
            tool_calls.get(&0).map(|tool_call| tool_call.name.as_str()),
            Some("time")
        );
    }

    #[test]
    fn finalizes_null_arguments_into_empty_objects() {
        let finalized = finalize_completed_streaming_tool_call(RawStreamingToolCall::empty());
        assert_eq!(finalized.arguments, serde_json::json!({}));
    }

    #[test]
    fn drains_finalized_tool_calls() {
        let mut tool_calls = HashMap::from([(0, RawStreamingToolCall::empty())]);

        let events = take_finalized_tool_calls::<()>(&mut tool_calls);

        assert!(tool_calls.is_empty());
        assert!(
            matches!(events.as_slice(), [RawStreamingChoice::ToolCall(tool_call)] if tool_call.arguments == serde_json::json!({}))
        );
    }
}
