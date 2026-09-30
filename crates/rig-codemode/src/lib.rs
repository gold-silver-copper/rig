//! Code mode: run model-written JavaScript in a QuickJS sandbox whose only
//! capability is calling Rig tools.
//!
//! A script is the body of an async function, so top-level `await` and
//! `return` work. It calls tools as `tools.<name>(args)`, writes output with
//! `text(value)` or `console.log(...)`, and has no file system, network,
//! timers or modules. Nested calls never reach the model; only the script's
//! output and return value do. [`CodeMode::tool`] exposes a sandbox as one
//! tool, so a model can batch, chain and filter tool calls in a single turn.
//!
//! ```no_run
//! # async fn run() {
//! use rig_codemode::CodeMode;
//! use rig_core::tool::{DynamicTool, ToolOutput};
//!
//! let add = DynamicTool::new("add", "Add a and b", serde_json::json!({}), |args| {
//!     Box::pin(async move {
//!         let sum = args["a"].as_i64().unwrap_or(0) + args["b"].as_i64().unwrap_or(0);
//!         Ok(ToolOutput::json(serde_json::json!(sum)))
//!     })
//! });
//! let run = CodeMode::new([add]).run("return await tools.add({a: 1, b: 2});").await;
//! assert_eq!(run.result.ok(), Some(Some(serde_json::json!(3))));
//! # }
//! ```

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

#[cfg(target_family = "wasm")]
compile_error!("rig-codemode runs QuickJS on a native thread and does not support wasm targets");

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rig_core::completion::ToolDefinition;
use rig_core::tool::{DynamicTool, ToolExecutionError, ToolOutput};
use rquickjs::allocator::{Allocator, RustAllocator};
use rquickjs::{Context, Ctx, Function, Promise, Runtime, Value};
use serde_json::json;

/// The default deadline for a whole script.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// The default cap on the script heap.
pub const DEFAULT_MEMORY_LIMIT: usize = 256 * 1024 * 1024;

/// A sandbox that runs scripts against a fixed set of tools.
#[derive(Clone, Debug)]
pub struct CodeMode {
    tools: Vec<DynamicTool>,
    timeout: Option<Duration>,
    memory_limit: usize,
}

/// What one script did.
#[derive(Clone, Debug, PartialEq)]
pub struct CodeModeRun {
    /// Text the script wrote with `text()` or `console`, in order.
    pub output: Vec<String>,
    /// The script's return value as JSON (`None` for `undefined`), or why
    /// it failed.
    pub result: Result<Option<serde_json::Value>, CodeModeError>,
    /// Every tool call the script made, in order.
    pub calls: Vec<NestedCall>,
}

/// One tool call made from a script.
#[derive(Clone, Debug, PartialEq)]
pub struct NestedCall {
    /// The tool's registered name.
    pub name: String,
    /// The arguments the script passed.
    pub arguments: serde_json::Value,
    /// The tool's error text, if it failed.
    pub error: Option<String>,
}

/// Why a script did not complete.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CodeModeError {
    /// The script threw, or rejected, with this error.
    #[error("{0}")]
    Script(String),
    /// The script ran past its deadline and was interrupted.
    #[error("the script did not finish within {0:?}")]
    Timeout(Duration),
    /// The script awaited a promise that nothing can settle.
    #[error("the script awaited a promise that can never settle")]
    Stalled,
    /// The sandbox itself failed.
    #[error("sandbox failure: {0}")]
    Sandbox(String),
}

impl CodeMode {
    /// A sandbox whose scripts may call `tools`, with [`DEFAULT_TIMEOUT`] and
    /// [`DEFAULT_MEMORY_LIMIT`].
    pub fn new(tools: impl IntoIterator<Item = DynamicTool>) -> Self {
        Self {
            tools: tools.into_iter().collect(),
            timeout: Some(DEFAULT_TIMEOUT),
            memory_limit: DEFAULT_MEMORY_LIMIT,
        }
    }

    /// Set the deadline for a whole script, or `None` for none. Time spent
    /// inside a tool call counts, but a running tool is not interrupted.
    #[must_use]
    pub fn with_timeout(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.timeout = timeout.into();
        self
    }

    /// Cap the script heap at `bytes`. Allocations past it throw
    /// `InternalError: out of memory` inside the script.
    #[must_use]
    pub fn with_memory_limit(mut self, bytes: usize) -> Self {
        self.memory_limit = bytes;
        self
    }

    /// The tools scripts can call.
    pub fn tools(&self) -> &[DynamicTool] {
        &self.tools
    }

    /// Run `code` on a dedicated thread and wait for it without blocking the
    /// caller's executor. Tool futures are driven on that thread by a minimal
    /// executor, so a tool that needs a specific runtime must spawn onto it.
    pub async fn run(&self, code: &str) -> CodeModeRun {
        let (sender, receiver) = futures::channel::oneshot::channel();
        let sandbox = self.clone();
        let code = code.to_owned();
        let spawned = std::thread::Builder::new()
            .name("rig-codemode".into())
            .spawn(move || {
                let _ = sender.send(sandbox.run_blocking(&code));
            });
        let failed = |message: String| CodeModeRun {
            output: Vec::new(),
            result: Err(CodeModeError::Sandbox(message)),
            calls: Vec::new(),
        };
        if let Err(error) = spawned {
            return failed(error.to_string());
        }
        receiver
            .await
            .unwrap_or_else(|_| failed("the script thread ended without a result".into()))
    }

    /// Run `code` on the current thread, blocking until it finishes.
    pub fn run_blocking(&self, code: &str) -> CodeModeRun {
        let state = Arc::new(Shared::default());
        let result = self.evaluate(code, &state);
        let output = take(&state.output);
        let calls = take(&state.calls);
        CodeModeRun {
            output,
            result,
            calls,
        }
    }

    fn evaluate(
        &self,
        code: &str,
        state: &Arc<Shared>,
    ) -> Result<Option<serde_json::Value>, CodeModeError> {
        let sandbox = |error: rquickjs::Error| CodeModeError::Sandbox(error.to_string());
        let exhausted = Arc::new(AtomicBool::new(false));
        let runtime = Runtime::new_with_alloc(CappedAllocator {
            inner: RustAllocator,
            used: 0,
            limit: self.memory_limit,
            exhausted: exhausted.clone(),
        })
        .map_err(sandbox)?;
        let timed_out = Arc::new(AtomicBool::new(false));
        if let Some(timeout) = self.timeout {
            let deadline = Instant::now() + timeout;
            let flag = timed_out.clone();
            runtime.set_interrupt_handler(Some(Box::new(move || {
                let late = Instant::now() > deadline;
                flag.fetch_or(late, Ordering::Relaxed);
                late
            })));
        }
        let context = Context::full(&runtime).map_err(sandbox)?;
        let outcome = context.with(|ctx| self.evaluate_in(&ctx, code, state));
        if timed_out.load(Ordering::Relaxed)
            && let Some(timeout) = self.timeout
        {
            return Err(CodeModeError::Timeout(timeout));
        }
        // Out of memory, QuickJS may be unable to allocate the error it
        // throws, so the allocator's record is the reliable signal.
        if outcome.is_err() && exhausted.load(Ordering::Relaxed) {
            return Err(CodeModeError::Script("InternalError: out of memory".into()));
        }
        outcome
    }

    fn evaluate_in(
        &self,
        ctx: &Ctx<'_>,
        code: &str,
        state: &Arc<Shared>,
    ) -> Result<Option<serde_json::Value>, CodeModeError> {
        let thrown = |error: rquickjs::Error| match error {
            rquickjs::Error::Exception => CodeModeError::Script(describe(ctx, ctx.catch())),
            rquickjs::Error::WouldBlock => CodeModeError::Stalled,
            other => CodeModeError::Sandbox(other.to_string()),
        };
        self.install(ctx, state).map_err(thrown)?;
        // A function body, so `return` and top-level `await` work; the
        // newline keeps a trailing line comment from swallowing the brace.
        let promise: Promise = ctx
            .eval(format!("(async () => {{\n{code}\n}})()"))
            .map_err(thrown)?;
        let value: Value = promise.finish().map_err(thrown)?;
        if value.is_undefined() {
            return Ok(None);
        }
        let Some(text) = ctx.json_stringify(value).map_err(thrown)? else {
            return Ok(None);
        };
        let text = text.to_string().map_err(thrown)?;
        serde_json::from_str(&text)
            .map(Some)
            .map_err(|error| CodeModeError::Sandbox(error.to_string()))
    }

    /// Define `tools`, `ALL_TOOLS`, `text` and `console` in the global scope.
    fn install(&self, ctx: &Ctx<'_>, state: &Arc<Shared>) -> rquickjs::Result<()> {
        let globals = ctx.globals();
        let tools = self.tools.clone();
        let calls = state.clone();
        globals.set(
            "__rig_call",
            Function::new(ctx.clone(), move |name: String, arguments: String| {
                call_tool(&tools, &calls, &name, &arguments)
            })?,
        )?;
        let output = state.clone();
        globals.set(
            "__rig_emit",
            Function::new(ctx.clone(), move |text: String| {
                if let Ok(mut output) = output.output.lock() {
                    output.push(text);
                }
            })?,
        )?;
        let catalog: Vec<serde_json::Value> = self
            .tools
            .iter()
            .map(|tool| {
                let definition = tool.definition();
                json!({"name": identifier(&definition.name), "tool": definition.name,
                       "description": definition.description})
            })
            .collect();
        let catalog = serde_json::Value::Array(catalog).to_string();
        ctx.eval::<(), _>(format!("{PRELUDE}\n__rig_install({catalog});"))
    }

    /// A tool named `codemode` that runs its `code` argument in this
    /// sandbox. Its description lists the nested tools and their schemas.
    /// The result is the script's output followed by its return value; a
    /// failed script is a tool error carrying the output so far.
    pub fn tool(self) -> DynamicTool {
        let description = self.description();
        let sandbox = Arc::new(self);
        DynamicTool::new(
            "codemode",
            description,
            json!({
                "type": "object",
                "properties": {"code": {
                    "type": "string",
                    "description": "JavaScript source: the body of an async function. Top-level await and return work."
                }},
                "required": ["code"]
            }),
            move |arguments| {
                let sandbox = sandbox.clone();
                Box::pin(async move {
                    let code = arguments
                        .get("code")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| {
                            ToolExecutionError::other("missing string argument `code`")
                        })?;
                    sandbox.run(code).await.into_tool_result()
                })
            },
        )
    }

    /// The model-facing description of [`Self::tool`].
    pub fn description(&self) -> String {
        let mut text = String::from(DESCRIPTION);
        if self.tools.is_empty() {
            return text;
        }
        text.push_str("\n\nNested tools:");
        for tool in &self.tools {
            let ToolDefinition {
                name,
                description,
                parameters,
            } = tool.definition();
            text.push_str(&format!(
                "\n\n- tools.{}(args): {description}\n  args schema: {parameters}",
                identifier(&name)
            ));
        }
        text
    }
}

impl CodeModeRun {
    /// The output and return value as text for a model: output lines, then
    /// the return value (strings verbatim, other values as JSON).
    pub fn render(&self) -> String {
        let mut lines = self.output.clone();
        match &self.result {
            Ok(Some(serde_json::Value::String(text))) => lines.push(text.clone()),
            Ok(Some(value)) => lines.push(value.to_string()),
            Ok(None) => {}
            Err(error) => lines.push(format!("Error: {error}")),
        }
        lines.join("\n")
    }

    /// [`Self::render`] as a tool result: an error keeps the output so far.
    pub fn into_tool_result(self) -> Result<ToolOutput, ToolExecutionError> {
        let text = self.render();
        match self.result {
            Ok(_) => Ok(ToolOutput::text(text)),
            Err(error) => Err(ToolExecutionError::other(error.to_string())
                .with_model_output(ToolOutput::text(text))),
        }
    }
}

/// The Rust allocator with a byte budget. A refused allocation sets
/// `exhausted`, which is how an out-of-memory failure is told apart from a
/// script that throws `null`.
struct CappedAllocator {
    inner: RustAllocator,
    used: usize,
    limit: usize,
    exhausted: Arc<AtomicBool>,
}

impl CappedAllocator {
    fn admit(&mut self, freed: usize, wanted: usize) -> bool {
        let fits = self.used.saturating_sub(freed).saturating_add(wanted) <= self.limit;
        if !fits {
            self.exhausted.store(true, Ordering::Relaxed);
        }
        fits
    }
}

// SAFETY: every pointer comes from, and goes back to, `RustAllocator`, which
// upholds the trait's contract; this wrapper only refuses requests (by
// returning null, which the contract allows) and counts usable sizes.
unsafe impl Allocator for CappedAllocator {
    fn alloc(&mut self, size: usize) -> *mut u8 {
        if !self.admit(0, size) {
            return std::ptr::null_mut();
        }
        let ptr = self.inner.alloc(size);
        if !ptr.is_null() {
            // SAFETY: `ptr` was just allocated by `RustAllocator`.
            self.used += unsafe { RustAllocator::usable_size(ptr) };
        }
        ptr
    }

    fn calloc(&mut self, count: usize, size: usize) -> *mut u8 {
        if !self.admit(0, count.saturating_mul(size)) {
            return std::ptr::null_mut();
        }
        let ptr = self.inner.calloc(count, size);
        if !ptr.is_null() {
            // SAFETY: `ptr` was just allocated by `RustAllocator`.
            self.used += unsafe { RustAllocator::usable_size(ptr) };
        }
        ptr
    }

    unsafe fn dealloc(&mut self, ptr: *mut u8) {
        if !ptr.is_null() {
            // SAFETY: the caller guarantees `ptr` came from this allocator.
            self.used = self
                .used
                .saturating_sub(unsafe { RustAllocator::usable_size(ptr) });
        }
        // SAFETY: forwarded under the caller's guarantee.
        unsafe { self.inner.dealloc(ptr) }
    }

    unsafe fn realloc(&mut self, ptr: *mut u8, new_size: usize) -> *mut u8 {
        let old = if ptr.is_null() {
            0
        } else {
            // SAFETY: the caller guarantees `ptr` came from this allocator.
            unsafe { RustAllocator::usable_size(ptr) }
        };
        if !self.admit(old, new_size) {
            return std::ptr::null_mut();
        }
        // SAFETY: forwarded under the caller's guarantee.
        let moved = unsafe { self.inner.realloc(ptr, new_size) };
        if !moved.is_null() {
            // SAFETY: `moved` was just allocated by `RustAllocator`.
            let size = unsafe { RustAllocator::usable_size(moved) };
            self.used = self.used.saturating_sub(old).saturating_add(size);
        }
        moved
    }

    unsafe fn usable_size(ptr: *mut u8) -> usize {
        // SAFETY: forwarded under the caller's guarantee.
        unsafe { RustAllocator::usable_size(ptr) }
    }
}

/// State shared between the script's host functions and the caller.
#[derive(Default)]
struct Shared {
    output: Mutex<Vec<String>>,
    calls: Mutex<Vec<NestedCall>>,
}

fn take<T>(items: &Mutex<Vec<T>>) -> Vec<T> {
    items
        .lock()
        .map(|mut items| std::mem::take(&mut *items))
        .unwrap_or_default()
}

/// Run one nested call and answer `{"ok": value}` or `{"error": message}`.
fn call_tool(tools: &[DynamicTool], state: &Shared, name: &str, arguments: &str) -> String {
    let arguments: serde_json::Value = serde_json::from_str(arguments).unwrap_or_default();
    let outcome = match tools.iter().find(|tool| tool.name() == name) {
        None => Err(format!("no tool named `{name}`")),
        Some(tool) => futures::executor::block_on(tool.execute(arguments.clone()))
            .map(|output| match (output.as_json(), output.as_text()) {
                (Some(value), _) => value.clone(),
                (None, Some(text)) => serde_json::Value::String(text.to_owned()),
                (None, None) => serde_json::Value::String(output.render()),
            })
            .map_err(|error| error.message().to_owned()),
    };
    if let Ok(mut calls) = state.calls.lock() {
        calls.push(NestedCall {
            name: name.to_owned(),
            arguments,
            error: outcome.as_ref().err().cloned(),
        });
    }
    match outcome {
        Ok(value) => json!({ "ok": value }),
        Err(error) => json!({ "error": error }),
    }
    .to_string()
}

/// A tool name as a JavaScript identifier: invalid characters become `_`.
pub fn identifier(name: &str) -> String {
    let mut ident: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '$' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if ident.is_empty() || ident.starts_with(|c: char| c.is_ascii_digit()) {
        ident.insert(0, '_');
    }
    ident
}

/// `Name: message` for a thrown error, or the thrown value itself.
fn describe<'js>(ctx: &Ctx<'js>, thrown: Value<'js>) -> String {
    if let Some(object) = thrown.as_object() {
        let name: Option<String> = object.get("name").ok();
        let message: Option<String> = object.get("message").ok();
        if let Some(message) = message {
            return match name {
                Some(name) => format!("{name}: {message}"),
                None => message,
            };
        }
    }
    ctx.json_stringify(thrown)
        .ok()
        .flatten()
        .and_then(|text| text.to_string().ok())
        .unwrap_or_else(|| "unknown error".into())
}

/// Globals written in JavaScript on top of the two host functions.
const PRELUDE: &str = r#"
function __rig_install(catalog) {
  const format = (value) => typeof value === "string" ? value : (JSON.stringify(value) ?? String(value));
  const tools = {};
  for (const entry of catalog) {
    const call = (args) => {
      const answer = JSON.parse(__rig_call(entry.tool, JSON.stringify(args ?? {})));
      if ("error" in answer) throw new Error(answer.error);
      return answer.ok;
    };
    tools[entry.name] = call;
    tools[entry.tool] = call;
  }
  globalThis.tools = Object.freeze(tools);
  globalThis.ALL_TOOLS = Object.freeze(catalog.map(({ name, description }) => ({ name, description })));
  globalThis.text = (value) => { if (value !== undefined) __rig_emit(format(value)); };
  const log = (...values) => __rig_emit(values.map(format).join(" "));
  globalThis.console = Object.freeze({ log, info: log, warn: log, error: log, debug: log });
}
"#;

const DESCRIPTION: &str = "Run JavaScript that calls other tools: chain calls, loop, and filter large results before they reach you.
- The code is the body of an async function in a fresh QuickJS sandbox: top-level `await` and `return` work.
- Call tools as `await tools.<name>(args)`, where args is an object matching the tool's schema. A call returns a string or a JSON value; a failed call throws an Error with the tool's error text.
- `text(value)` and `console.log(...)` append output; the return value is appended after it. Only this output reaches you, not the nested results.
- `ALL_TOOLS` lists `{ name, description }` of the nested tools.
- No file system, network, timers, modules or Node APIs. Tool calls are real and have side effects.";

#[cfg(test)]
mod tests;
