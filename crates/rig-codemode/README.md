# rig-codemode

Code mode for [Rig](https://github.com/0xPlaygrounds/rig): run model-written
JavaScript in a [QuickJS](https://bellard.org/quickjs/) sandbox whose only
capability is calling Rig tools.

Instead of one tool call per model turn, the model writes a short script that
calls several tools, loops, and filters their results. The nested calls never
enter the model's context; only what the script prints and returns does.

```rust,no_run
use rig_codemode::CodeMode;
use rig_core::tool::{DynamicTool, ToolOutput};

# async fn run(read_file: DynamicTool) {
let sandbox = CodeMode::new([read_file]);
let run = sandbox
    .run(r#"const text = await tools.read_file({path: "Cargo.toml"});
            return text.split("\n").length;"#)
    .await;
println!("{}", run.render());

// Or give the model one `codemode` tool that runs scripts against the others.
let tool: DynamicTool = sandbox.tool();
# }
```

Inside a script:

- The code is the body of an async function: top-level `await` and `return`
  work.
- `tools.<name>(args)` calls a tool with a JSON object. It returns a string or a
  JSON value, and throws an `Error` carrying the tool's error text on failure.
  Names that are not identifiers are available with `_` for invalid characters
  (`my-tool` is `tools.my_tool`) and under their own name (`tools["my-tool"]`).
- `text(value)` and `console.log(...)` append output. `ALL_TOOLS` lists
  `{ name, description }` for every tool.
- There is no file system, network, timers or modules: nothing but the tools
  reaches outside the VM.

Scripts run on a dedicated thread with a deadline (60 s by default,
`with_timeout`) and a heap cap (256 MiB by default, `with_memory_limit`).
Tool futures are driven on that thread by a minimal executor, so a tool that
needs a specific async runtime must spawn onto it. A running tool call is not
interrupted by the deadline.

The crate is native-only: QuickJS is compiled from C by `rquickjs`.
