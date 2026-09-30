use super::*;
use crate::message::{ToolCall, ToolFunction, ToolResult};

fn call(id: &str) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall {
        id: crate::message::CallId::from_wire(id),
        function: ToolFunction {
            name: crate::message::ToolName::new("add").expect("tool name"),
            arguments: serde_json::json!({}),
        },
        additional_params: None,
        signature: None,
    })
}
fn result(id: &str) -> UserContent {
    UserContent::ToolResult(ToolResult {
        call: crate::message::CallId::from_wire(id),
        name: crate::message::ToolName::new("add").expect("tool name"),
        content: vec![ToolResultContent::text("3")],
    })
}
fn assistant(content: Vec<AssistantContent>) -> Message {
    Message::Assistant { id: None, content }
}

#[test]
fn canonical_transcripts_pass() {
    let history = vec![
        Message::user("hi"),
        assistant(vec![call("c1")]),
        Message::User {
            content: vec![result("c1")],
        },
        assistant(vec![AssistantContent::text("done")]),
        Message::user("thanks"),
    ];
    assert_eq!(validate_canonical(&history), Ok(()));
    assert!(validate_canonical(&[]).is_ok());
}

#[test]
fn consecutive_assistant_is_rejected() {
    let history = vec![
        assistant(vec![AssistantContent::text("a")]),
        assistant(vec![AssistantContent::text("b")]),
    ];
    assert_eq!(
        validate_canonical(&history),
        Err(TranscriptError::ConsecutiveAssistant { index: 1 })
    );
}

#[test]
fn unanswered_and_orphan_results_are_rejected() {
    let unanswered = vec![assistant(vec![call("c1")]), Message::user("no result")];
    assert!(matches!(
        validate_canonical(&unanswered),
        Err(TranscriptError::UnansweredToolCall { .. })
    ));
    let orphan = vec![
        Message::user("hi"),
        Message::User {
            content: vec![result("ghost")],
        },
    ];
    assert!(matches!(
        validate_canonical(&orphan),
        Err(TranscriptError::OrphanToolResult { .. })
    ));
    let trailing = vec![assistant(vec![call("c1")])];
    assert!(matches!(
        validate_canonical(&trailing),
        Err(TranscriptError::UnansweredToolCall { .. })
    ));
}

/// A result answers only the call whose id it carries.
#[test]
fn a_result_answers_only_its_own_call() {
    let (first, second) = (
        crate::message::CallId::from_wire(""),
        crate::message::CallId::from_wire(""),
    );
    let tool = crate::message::ToolName::new("add").expect("tool name");
    let call_for = |id: crate::message::CallId| {
        AssistantContent::ToolCall(ToolCall::new(
            id,
            ToolFunction::new(tool.clone(), serde_json::json!({})),
        ))
    };
    let result_for = |id: crate::message::CallId| {
        UserContent::tool_result(id, tool.clone(), vec![ToolResultContent::text("3")])
    };
    let history = vec![
        assistant(vec![call_for(first.clone()), call_for(second.clone())]),
        Message::User {
            content: vec![result_for(second.clone()), result_for(first.clone())],
        },
    ];
    assert_eq!(validate_canonical(&history), Ok(()));
    let mismatched = vec![
        assistant(vec![call_for(first)]),
        Message::User {
            content: vec![result_for(second)],
        },
    ];
    assert!(matches!(
        validate_canonical(&mismatched),
        Err(TranscriptError::OrphanToolResult { .. })
    ));
}

#[test]
fn unanswered_calls_at_the_end_get_a_new_results_message() {
    let mut history = vec![Message::user("hi"), assistant(vec![call("c1"), call("c2")])];
    let added = answer_unanswered(&mut history, |call| format!("lost {}", call.id));
    assert_eq!(added, 2);
    assert_eq!(history.len(), 3);
    validate_canonical(&history).expect("repaired history is canonical");
}

#[test]
fn partly_answered_calls_are_completed_in_place_ahead_of_text() {
    let mut history = vec![
        Message::user("hi"),
        assistant(vec![call("c1"), call("c2")]),
        Message::User {
            content: vec![result("c1"), UserContent::text("and then?")],
        },
        assistant(vec![AssistantContent::text("done")]),
    ];
    assert_eq!(answer_unanswered(&mut history, |_| "interrupted".into()), 1);
    assert_eq!(history.len(), 4);
    let Some(Message::User { content }) = history.get(2) else {
        panic!("the results message stays in place");
    };
    assert!(
        matches!(content.get(1), Some(UserContent::ToolResult(r)) if r.call == crate::message::CallId::from_wire("c2"))
    );
    assert!(matches!(content.get(2), Some(UserContent::Text(_))));
    validate_canonical(&history).expect("repaired history is canonical");
}

#[test]
fn canonical_histories_are_left_alone() {
    let mut history = vec![
        Message::user("hi"),
        assistant(vec![call("c1")]),
        Message::User {
            content: vec![result("c1")],
        },
    ];
    let before = history.clone();
    assert_eq!(answer_unanswered(&mut history, |_| unreachable!()), 0);
    assert_eq!(history, before);
}

#[test]
fn a_call_followed_by_an_assistant_message_is_answered_between_them() {
    let mut history = vec![
        assistant(vec![call("c1")]),
        assistant(vec![AssistantContent::text("again")]),
    ];
    assert_eq!(answer_unanswered(&mut history, |_| "interrupted".into()), 1);
    assert!(matches!(history.get(1), Some(Message::User { .. })));
    validate_canonical(&history).expect("repaired history is canonical");
}
