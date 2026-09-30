# rig-codemode

Code mode for [Rig](https://github.com/0xPlaygrounds/rig): the model writes
JavaScript whose only capability is calling tools. One script can chain tool
calls, loop, run them with `Promise.all`, and filter large results, and only
its output reaches the model's context.

Each execution runs in a fresh [QuickJS](https://bellard.org/quickjs/)
interpreter with no file system, network, timers or modules, a memory limit,
and an optional timeout.

- `CodeMode::new(definitions)` builds a sandbox over tool definitions.
  `definition()` is the `codemode` tool to offer the model: its description
  documents the script environment and declares every tool in TypeScript
  syntax.
- `execute(code, call)` runs a script on the current thread and sends every
  `tools.<name>(args)` call to `call`, so the host decides how tools run.
- `into_tool(dynamic_tools)` returns one `DynamicTool` that runs scripts
  over those tools itself, awaiting nested calls on the caller's executor.

Native targets only: QuickJS is a C library.

```rust
use rig_codemode::CodeMode;
use rig_core::completion::ToolDefinition;
use serde_json::json;

let read = ToolDefinition {
    name: "read".into(),
    description: "Read a file".into(),
    parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
};
let execution = CodeMode::new([read]).execute(
    "const text = await tools.read({path: 'notes.txt'}); return text.length;",
    |_, _| Ok(json!("hello")),
);
assert_eq!(execution.output, ["5"]);
```
