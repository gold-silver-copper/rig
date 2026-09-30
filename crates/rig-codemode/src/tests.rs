use std::time::Duration;

use rig_core::tool::{DynamicTool, ToolExecutionError, ToolOutput};
use serde_json::json;

use super::{CodeMode, CodeModeError, NestedCall, identifier};

fn add() -> DynamicTool {
    DynamicTool::new("add", "Add a and b.", json!({"type": "object"}), |args| {
        Box::pin(async move {
            let sum = args["a"].as_i64().unwrap_or(0) + args["b"].as_i64().unwrap_or(0);
            Ok(ToolOutput::json(json!(sum)))
        })
    })
}

fn shout() -> DynamicTool {
    DynamicTool::new(
        "shout-it",
        "Upper-case text.",
        json!({"type": "object"}),
        |args| {
            Box::pin(async move {
                Ok(ToolOutput::text(
                    args["text"].as_str().unwrap_or_default().to_uppercase(),
                ))
            })
        },
    )
}

fn broken() -> DynamicTool {
    DynamicTool::new("broken", "Always fails.", json!({"type": "object"}), |_| {
        Box::pin(async { Err(ToolExecutionError::other("disk on fire")) })
    })
}

#[tokio::test]
async fn scripts_call_tools_and_return_values() {
    let run = CodeMode::new([add()])
        .run("const x = await tools.add({a: 2, b: 3}); return x * 10;")
        .await;
    assert_eq!(run.result, Ok(Some(json!(50))));
    assert_eq!(
        run.calls,
        vec![NestedCall {
            name: "add".into(),
            arguments: json!({"a": 2, "b": 3}),
            error: None
        }]
    );
}

#[tokio::test]
async fn text_results_are_strings_and_names_become_identifiers() {
    let run = CodeMode::new([shout()])
        .run(r#"return [tools.shout_it({text: "a"}), await tools["shout-it"]({text: "b"})];"#)
        .await;
    assert_eq!(run.result, Ok(Some(json!(["A", "B"]))));
    assert_eq!(identifier("mcp.server/tool-1"), "mcp_server_tool_1");
    assert_eq!(identifier("1x"), "_1x");
}

#[tokio::test]
async fn output_comes_from_text_and_console_in_order() {
    let run = CodeMode::new([])
        .run(r#"text("one"); console.log("two", {n: 3}); text(4); return "done";"#)
        .await;
    assert_eq!(run.output, vec!["one", r#"two {"n":3}"#, "4"]);
    assert_eq!(run.render(), "one\ntwo {\"n\":3}\n4\ndone");
}

#[tokio::test]
async fn tool_errors_throw_and_can_be_caught() {
    let run = CodeMode::new([broken()])
        .run("try { await tools.broken({}); } catch (e) { return 'caught: ' + e.message; }")
        .await;
    assert_eq!(run.result, Ok(Some(json!("caught: disk on fire"))));
    assert_eq!(run.calls[0].error.as_deref(), Some("disk on fire"));

    let uncaught = CodeMode::new([broken()])
        .run("await tools.broken({});")
        .await;
    assert_eq!(
        uncaught.result,
        Err(CodeModeError::Script("Error: disk on fire".into()))
    );
}

#[tokio::test]
async fn the_sandbox_has_no_ambient_capabilities() {
    let run = CodeMode::new([])
        .run(
            "return [typeof fetch, typeof require, typeof setTimeout, typeof process, \
             typeof std, typeof os, typeof tools.nothing];",
        )
        .await;
    assert_eq!(run.result, Ok(Some(json!(vec!["undefined"; 7]))));
}

#[tokio::test]
async fn a_runaway_script_times_out() {
    let run = CodeMode::new([])
        .with_timeout(Duration::from_millis(100))
        .run("while (true) {}")
        .await;
    assert_eq!(
        run.result,
        Err(CodeModeError::Timeout(Duration::from_millis(100)))
    );
}

#[tokio::test]
async fn the_heap_is_capped() {
    let run = CodeMode::new([])
        .with_memory_limit(8 * 1024 * 1024)
        .run("const a = []; while (true) a.push('x'.repeat(1024));")
        .await;
    let Err(CodeModeError::Script(message)) = run.result else {
        panic!("expected an out-of-memory error, got {:?}", run.result);
    };
    assert!(message.contains("out of memory"), "{message}");
}

#[tokio::test]
async fn syntax_errors_and_stalls_are_reported() {
    let syntax = CodeMode::new([]).run("return )").await;
    assert!(
        matches!(&syntax.result, Err(CodeModeError::Script(m)) if m.starts_with("SyntaxError")),
        "{:?}",
        syntax.result
    );
    let stalled = CodeMode::new([]).run("await new Promise(() => {});").await;
    assert_eq!(stalled.result, Err(CodeModeError::Stalled));
}

#[tokio::test]
async fn as_a_tool_it_runs_code_and_lists_the_nested_tools() {
    let tool = CodeMode::new([add(), shout()]).tool();
    let definition = tool.definition();
    assert_eq!(definition.name, "codemode");
    assert!(
        definition
            .description
            .contains("tools.add(args): Add a and b.")
    );
    assert!(definition.description.contains("tools.shout_it(args)"));

    let output = tool
        .execute(json!({"code": "text('sum'); return await tools.add({a: 1, b: 1});"}))
        .await
        .expect("script succeeds");
    assert_eq!(output.as_text(), Some("sum\n2"));

    let error = tool
        .execute(json!({"code": "text('partial'); throw new TypeError('nope');"}))
        .await
        .expect_err("script fails");
    assert_eq!(error.message(), "TypeError: nope");
    assert_eq!(
        error.model_output().as_text(),
        Some("partial\nError: TypeError: nope")
    );
}
