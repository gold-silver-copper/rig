use super::*;

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: format!("The {name} tool"),
        parameters: json!({
            "type": "object",
            "properties": {"path": {"type": "string", "description": "A path"}, "n": {"type": "integer"}},
            "required": ["path"]
        }),
    }
}

fn echo(name: &str, arguments: Value) -> Result<Value, String> {
    Ok(json!({"tool": name, "args": arguments}))
}

#[test]
fn tools_chain_and_only_output_is_kept() {
    let sandbox = CodeMode::new([tool("read"), tool("my-tool")]);
    let execution = sandbox.execute(
        r#"
        const a = await tools.read({path: "a"});
        const b = await tools.my_tool({path: a.args.path + "b"});
        const both = await Promise.all([tools["my-tool"]({path: "x"}), tools.read({path: "y"})]);
        console.log("got", b.args.path, both.length);
        return {done: true};
        "#,
        echo,
    );
    assert_eq!(execution.outcome, Ok(()));
    assert_eq!(execution.output, ["got ab 2", r#"{"done":true}"#]);
    let names: Vec<&str> = execution
        .calls
        .iter()
        .map(|call| call.tool.as_str())
        .collect();
    assert_eq!(names, ["read", "my-tool", "my-tool", "read"]);
}

#[test]
fn a_failing_tool_rejects_and_the_script_can_catch_it() {
    let sandbox = CodeMode::new([tool("read")]);
    let execution = sandbox.execute(
        r#"try { await tools.read({path: "missing"}); } catch (e) { text("caught: " + e.message); }"#,
        |_, _| Err("no such file".into()),
    );
    assert_eq!(execution.outcome, Ok(()));
    assert_eq!(execution.output, ["caught: no such file"]);
    assert_eq!(execution.calls[0].result, Err("no such file".into()));
}

#[test]
fn uncaught_errors_fail_the_execution_and_keep_earlier_output() {
    let sandbox = CodeMode::new([]);
    let execution = sandbox.execute(r#"text("before"); throw new Error("boom");"#, echo);
    assert_eq!(execution.output, ["before"]);
    let error = execution.outcome.clone().unwrap_err();
    assert!(error.contains("boom"), "{error}");
    assert!(execution.render().contains("Error:"));
}

#[test]
fn exit_ends_successfully() {
    let execution = CodeMode::new([]).execute(r#"text("a"); exit(); text("b");"#, echo);
    assert_eq!(execution.outcome, Ok(()));
    assert_eq!(execution.output, ["a"]);
}

#[test]
fn the_sandbox_has_no_host_access() {
    let execution = CodeMode::new([]).execute(
        r#"text([typeof require, typeof process, typeof fetch, typeof setTimeout, typeof std, typeof os].join(","));"#,
        echo,
    );
    assert_eq!(
        execution.output,
        ["undefined,undefined,undefined,undefined,undefined,undefined"]
    );
}

#[test]
fn runaway_scripts_time_out() {
    let execution = CodeMode::new([])
        .with_timeout(Duration::from_millis(200))
        .execute("while (true) {}", echo);
    assert!(execution.outcome.is_err());
}

#[test]
fn a_promise_nothing_can_settle_fails_instead_of_hanging() {
    let execution = CodeMode::new([]).execute("await new Promise(() => {});", echo);
    let error = execution.outcome.unwrap_err();
    assert!(error.contains("nothing can settle"), "{error}");
}

#[test]
fn the_definition_declares_every_tool() {
    let definition = CodeMode::new([tool("read"), tool("mcp.server/tool")]).definition();
    assert_eq!(definition.name, TOOL_NAME);
    assert!(definition.description.contains("read(args: {"));
    assert!(definition.description.contains("path: string;"));
    assert!(definition.description.contains("n?: number;"));
    assert!(definition.description.contains("mcp_server_tool(args"));
    assert_eq!(definition.parameters["required"], json!(["code"]));
}

#[tokio::test]
async fn into_tool_runs_dynamic_tools_on_the_callers_executor() {
    let double = DynamicTool::new(
        "double",
        "Double n",
        json!({"type": "object", "properties": {"n": {"type": "number"}}}),
        |args: Value| {
            Box::pin(async move {
                tokio::task::yield_now().await;
                Ok(ToolOutput::json(json!(
                    args["n"].as_f64().unwrap_or(0.0) * 2.0
                )))
            })
        },
    );
    let codemode = CodeMode::new([double.definition()]).into_tool(vec![double]);
    let output = codemode
        .execute(json!({"code": "let n = 1; for (let i = 0; i < 3; i++) n = await tools.double({n}); return n;"}))
        .await
        .expect("the script runs");
    assert_eq!(output.render(), "8");
}
