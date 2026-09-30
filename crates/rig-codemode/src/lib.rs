#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! Code mode: the model writes JavaScript, and the script's only capability
//! is calling tools. A script can chain, loop over, and filter tool calls in
//! one model turn, and only its output reaches the model's context.
//!
//! Scripts run in a fresh QuickJS interpreter per execution, with no file
//! system, network, timers, or modules. [`CodeMode::execute`] runs a script
//! and routes each `tools.<name>(args)` call to a callback, so a host decides
//! how tools run. [`CodeMode::into_tool`] wraps a set of
//! [`DynamicTool`]s into one `codemode` tool that runs them itself.
//!
//! ```
//! use rig_codemode::CodeMode;
//! use rig_core::completion::ToolDefinition;
//! use serde_json::json;
//!
//! let add = ToolDefinition {
//!     name: "add".into(),
//!     description: "Add two numbers".into(),
//!     parameters: json!({"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}),
//! };
//! let code_mode = CodeMode::new([add]);
//! let execution = code_mode.execute(
//!     "const sum = await tools.add({a: 2, b: 3}); text(`sum is ${sum}`);",
//!     |_, args| Ok(json!(args["a"].as_f64().unwrap_or(0.0) + args["b"].as_f64().unwrap_or(0.0))),
//! );
//! assert_eq!(execution.output, ["sum is 5"]);
//! ```

#[cfg(target_family = "wasm")]
compile_error!("rig-codemode is native-only: it embeds the QuickJS interpreter, a C library.");

use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use rig_core::completion::ToolDefinition;
use rig_core::tool::{DynamicTool, ToolExecutionError, ToolOutput};
use rquickjs::{CatchResultExt, Context, Function, Promise, Runtime};
use serde_json::{Value, json};

/// The name of the tool [`CodeMode::definition`] describes.
pub const TOOL_NAME: &str = "codemode";

const DEFAULT_MEMORY_LIMIT: usize = 256 * 1024 * 1024;

/// A code-mode sandbox over a set of tool definitions.
#[derive(Debug, Clone)]
pub struct CodeMode {
    tools: Vec<ToolDefinition>,
    timeout: Option<Duration>,
    memory_limit: usize,
}

/// One nested tool call a script made.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    /// The tool's name.
    pub tool: String,
    /// The arguments the script passed.
    pub arguments: Value,
    /// What the tool returned, or its error text.
    pub result: Result<Value, String>,
}

/// What an execution produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Execution {
    /// Text items from `text()`, `console.*` and the returned value, in order.
    pub output: Vec<String>,
    /// Every nested tool call, in order.
    pub calls: Vec<Call>,
    /// `Err` with the error text when the script threw, timed out, or ran
    /// out of memory. Output and calls made before the failure are kept.
    pub outcome: Result<(), String>,
}

impl Execution {
    /// The execution as text for the model: the output, then the error if
    /// the script failed.
    pub fn render(&self) -> String {
        let mut text = self.output.join("\n");
        if let Err(error) = &self.outcome {
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(&format!("Error: {error}"));
        }
        if text.is_empty() {
            text.push_str("(no output)");
        }
        text
    }
}

impl CodeMode {
    /// A sandbox whose scripts may call `tools`, with no timeout and a
    /// 256 MiB memory limit.
    pub fn new(tools: impl IntoIterator<Item = ToolDefinition>) -> Self {
        Self {
            tools: tools.into_iter().collect(),
            timeout: None,
            memory_limit: DEFAULT_MEMORY_LIMIT,
        }
    }

    /// Stop scripts that run longer than `timeout`, nested calls included.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Cap the interpreter's heap at `bytes`.
    pub fn with_memory_limit(mut self, bytes: usize) -> Self {
        self.memory_limit = bytes;
        self
    }

    /// The tools scripts may call.
    pub fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    /// The `codemode` tool: one `code` string argument, and a description
    /// with the script environment and a declaration for every tool.
    pub fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: TOOL_NAME.into(),
            description: format!("{INTRO}\n\n{}", self.declarations()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Raw JavaScript source, the body of an async function. Top-level await and return work."
                    }
                },
                "required": ["code"]
            }),
        }
    }

    /// TypeScript-style declarations of the `tools` object scripts see.
    pub fn declarations(&self) -> String {
        let mut text = String::from("declare const tools: {\n");
        for tool in &self.tools {
            let description = tool.description.replace("*/", "* /");
            text.push_str(&format!(
                "  /** {} */\n  {}(args: {}): Promise<unknown>;\n",
                description.trim(),
                identifier(&tool.name),
                schema_type(&tool.parameters, 2)
            ));
        }
        text.push_str("};");
        text
    }

    /// Run `code` as the body of an async function on this thread. Each
    /// `tools.<name>(args)` call invokes `call` with the tool's name and
    /// arguments; `Ok` resolves the script's promise with the value and
    /// `Err` rejects it with an `Error` carrying the text. The call blocks
    /// the script until `call` returns.
    pub fn execute<F>(&self, code: &str, call: F) -> Execution
    where
        F: FnMut(&str, Value) -> Result<Value, String> + 'static,
    {
        let state = Rc::new(RefCell::new(State {
            output: Vec::new(),
            calls: Vec::new(),
            call: Box::new(call),
        }));
        let outcome = self.run(code, &state);
        let State { output, calls, .. } = Rc::try_unwrap(state)
            .map(RefCell::into_inner)
            .unwrap_or_else(|shared| {
                let mut state = shared.borrow_mut();
                State {
                    output: std::mem::take(&mut state.output),
                    calls: std::mem::take(&mut state.calls),
                    call: Box::new(|_, _| Err(String::new())),
                }
            });
        Execution {
            output,
            calls,
            outcome,
        }
    }

    fn run(&self, code: &str, state: &Rc<RefCell<State>>) -> Result<(), String> {
        let runtime = Runtime::new().map_err(|error| error.to_string())?;
        runtime.set_memory_limit(self.memory_limit);
        if let Some(timeout) = self.timeout {
            let deadline = Instant::now() + timeout;
            runtime.set_interrupt_handler(Some(Box::new(move || Instant::now() > deadline)));
        }
        let context = Context::full(&runtime).map_err(|error| error.to_string())?;
        let catalog: Vec<Value> = self
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "name": tool.name,
                    "id": identifier(&tool.name),
                    "description": tool.description,
                })
            })
            .collect();
        let program = format!(
            "{PRELUDE}\n__install({});\n(async () => {{ try {{ const __value = await (async () => {{\n{code}\n}})(); if (__value !== undefined) text(__value); }} catch (e) {{ if (e !== __EXIT) throw e; }} }})()",
            Value::Array(catalog)
        );
        context.with(|ctx| {
            let globals = ctx.globals();
            let emit_state = state.clone();
            let emit = Function::new(ctx.clone(), move |text: String| {
                emit_state.borrow_mut().output.push(text);
            })
            .map_err(|error| error.to_string())?;
            globals
                .set("__emit", emit)
                .map_err(|error| error.to_string())?;
            let call_state = state.clone();
            let call = Function::new(ctx.clone(), move |name: String, arguments: String| {
                let arguments: Value = serde_json::from_str(&arguments).unwrap_or(Value::Null);
                let result = {
                    let mut state = call_state.borrow_mut();
                    (state.call)(&name, arguments.clone())
                };
                call_state.borrow_mut().calls.push(Call {
                    tool: name,
                    arguments,
                    result: result.clone(),
                });
                match result {
                    Ok(value) => json!({ "ok": value }).to_string(),
                    Err(error) => json!({ "error": error }).to_string(),
                }
            })
            .map_err(|error| error.to_string())?;
            globals
                .set("__call", call)
                .map_err(|error| error.to_string())?;

            let promise: Promise = ctx
                .eval(program)
                .catch(&ctx)
                .map_err(|error| error.to_string())?;
            loop {
                match promise.result::<rquickjs::Value>() {
                    Some(Ok(_)) => return Ok(()),
                    Some(Err(error)) => {
                        return Err(Err::<(), _>(error)
                            .catch(&ctx)
                            .err()
                            .map_or_else(|| "script failed".into(), |error| error.to_string()));
                    }
                    None if ctx.execute_pending_job() => {}
                    None => {
                        return Err("the script is waiting on a promise nothing can settle".into());
                    }
                }
            }
        })
    }

    /// One `codemode` tool over `tools`: nested calls run those tools on the
    /// caller's executor while the script waits on its own thread.
    pub fn into_tool(self, tools: Vec<DynamicTool>) -> DynamicTool {
        use futures::channel::{mpsc, oneshot};
        use futures::{SinkExt, StreamExt};

        type Request = (String, Value, oneshot::Sender<Result<Value, String>>);

        let definition = self.definition();
        let sandbox = std::sync::Arc::new(self);
        let tools = std::sync::Arc::new(tools);
        DynamicTool::new(
            definition.name,
            definition.description,
            definition.parameters,
            move |arguments: Value| {
                let sandbox = sandbox.clone();
                let tools = tools.clone();
                Box::pin(async move {
                    let code = arguments
                        .get("code")
                        .and_then(Value::as_str)
                        .ok_or_else(|| ToolExecutionError::invalid_args("`code` must be a string"))?
                        .to_owned();
                    let (requests, mut incoming) = mpsc::channel::<Request>(0);
                    let (done, finished) = oneshot::channel();
                    std::thread::spawn(move || {
                        let execution = sandbox.execute(&code, move |name, arguments| {
                            let (reply, answer) = oneshot::channel();
                            let mut requests = requests.clone();
                            futures::executor::block_on(requests.send((
                                name.to_owned(),
                                arguments,
                                reply,
                            )))
                            .map_err(|_| "the host stopped".to_owned())?;
                            futures::executor::block_on(answer)
                                .unwrap_or_else(|_| Err("the host stopped".into()))
                        });
                        let _ = done.send(execution);
                    });
                    while let Some((name, arguments, reply)) = incoming.next().await {
                        let result = match tools.iter().find(|tool| tool.name() == name) {
                            Some(tool) => tool
                                .execute(arguments)
                                .await
                                .map(output_value)
                                .map_err(|error| error.to_string()),
                            None => Err(format!("there is no tool named `{name}`")),
                        };
                        let _ = reply.send(result);
                    }
                    let execution = finished
                        .await
                        .map_err(|_| ToolExecutionError::other("the script thread stopped"))?;
                    Ok(ToolOutput::text(execution.render()))
                })
            },
        )
    }
}

/// A tool's output as a script sees it: its JSON, or its text.
pub fn output_value(output: ToolOutput) -> Value {
    match output.as_json() {
        Some(value) => value.clone(),
        None => Value::String(output.render()),
    }
}

struct State {
    output: Vec<String>,
    calls: Vec<Call>,
    call: Box<dyn FnMut(&str, Value) -> Result<Value, String>>,
}

/// `name` with every character that cannot appear in a JavaScript
/// identifier replaced by `_`.
pub fn identifier(name: &str) -> String {
    let mut id: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '$' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if id.chars().next().is_none_or(|c| c.is_ascii_digit()) {
        id.insert(0, '_');
    }
    id
}

fn schema_type(schema: &Value, indent: usize) -> String {
    if let Some(options) = schema.get("enum").and_then(Value::as_array) {
        return options
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join(" | ");
    }
    match schema.get("type").and_then(Value::as_str) {
        Some("string") => "string".into(),
        Some("number" | "integer") => "number".into(),
        Some("boolean") => "boolean".into(),
        Some("null") => "null".into(),
        Some("array") => {
            let items = schema
                .get("items")
                .map_or("unknown".into(), |items| schema_type(items, indent));
            format!("Array<{items}>")
        }
        Some("object") | None if schema.get("properties").is_some() => {
            let required: Vec<&str> = schema
                .get("required")
                .and_then(Value::as_array)
                .map(|names| names.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            let mut text = String::from("{\n");
            if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
                for (name, property) in properties {
                    let pad = " ".repeat(indent + 2);
                    if let Some(description) = property.get("description").and_then(Value::as_str) {
                        text.push_str(&format!(
                            "{pad}/** {} */\n",
                            description.replace("*/", "* /")
                        ));
                    }
                    let optional = if required.contains(&name.as_str()) {
                        ""
                    } else {
                        "?"
                    };
                    text.push_str(&format!(
                        "{pad}{}{optional}: {};\n",
                        identifier(name),
                        schema_type(property, indent + 2)
                    ));
                }
            }
            text.push_str(&format!("{}}}", " ".repeat(indent)));
            text
        }
        Some("object") => "Record<string, unknown>".into(),
        _ => "unknown".into(),
    }
}

const INTRO: &str = "Run JavaScript that calls other tools: chain them, loop, use Promise.all, and \
filter large results down to what you need, in one step.
- The code is the body of an async function in a fresh sandbox: top-level `await` and `return` work.
- Every tool is a method of the global `tools` object that takes one object argument, for example `await tools.read({path: \"a.txt\"})`. A failing call rejects with an Error carrying the tool's error text.
- Only the script's output reaches you: `text(value)` and `console.log(...)` append a text item, and a returned value is appended too. Non-strings are JSON-stringified.
- `exit()` ends the script successfully. `ALL_TOOLS` lists `{name, description}` for every tool.
- There is no file system, network, timers or modules; tool calls are the only side effects, and they are real.";

const PRELUDE: &str = r#"
const __EXIT = Symbol("exit");
function __format(value) {
  if (typeof value === "string") return value;
  try { const json = JSON.stringify(value); return json === undefined ? String(value) : json; }
  catch (_) { return String(value); }
}
globalThis.text = (value) => __emit(__format(value));
globalThis.exit = () => { throw __EXIT; };
const __log = (...values) => __emit(values.map(__format).join(" "));
globalThis.console = { log: __log, info: __log, warn: __log, error: __log, debug: __log };
function __install(catalog) {
  const tools = {};
  for (const entry of catalog) {
    const run = async (args) => {
      const reply = JSON.parse(__call(entry.name, JSON.stringify(args === undefined ? {} : args)));
      if ("error" in reply) throw new Error(reply.error);
      return reply.ok;
    };
    tools[entry.name] = run;
    tools[entry.id] = run;
  }
  globalThis.tools = Object.freeze(tools);
  globalThis.ALL_TOOLS = Object.freeze(catalog.map((entry) => ({ name: entry.id, description: entry.description })));
}
"#;

#[cfg(test)]
mod tests;
