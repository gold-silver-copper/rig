# Minimal Bevy coding agent

A headless Bevy 0.20.0-rc.2 app, Rig's raw completion API, and a ratatui terminal.
No `rig-agent`, `rig-ecs`, window, Dioxus CLI, or external hot-reload service.

```sh
# Rust >=1.96, native rustc on PATH, and OPENAI_API_KEY in the environment.
cd examples/bevy_agent
cargo +1.98.1 run --locked -- --root ../..  # or any installed toolchain >=1.96
```

Enter a prompt. The conversation shows tool calls, results, and the final answer.
Tools: `read_file`, `write_file`, `shell`, `native`, `patch_native`, plus external
extensions. PgUp/PgDn scroll; Ctrl-C exits. `--model` (or `BEVY_AGENT_MODEL`)
selects an OpenAI chat-completions model, default `gpt-4o-mini`.

## Native plugins

Edit `native.rs` while the app runs, or ask the model to edit it. The host
snapshots the source, compiles a unique cdylib with `rustc`, validates the ABI,
and installs its entry point into **Subsecond's real jump table**. Bevy's
`HotPatchChanges` and `HotPatched` notify its hotpatch-enabled ECS. The
`NativePlugin` is a Bevy Plugin with an ECS poll system; its reloadable code
uses a deliberately tiny UTF-8 buffer ABI, not Rust `World` pointers across
dynamic libraries. The header and `native` tool immediately use the new code.
`patch_native` waits for the build; automatic detection requires no tool call.
Compile failures and wrong ABI versions retain the previous plugin.

No `dx` is invoked, looked up, or connected to. Bevy's stock HotPatchPlugin
is replaced because it connects to dx. No whole-program symbol rewriting,
state migration, signature changes, or arbitrary Rust Plugin replacement.
Keep the three export signatures and ABI version fixed. Debug builds only;
patch libraries stay loaded so stale code pointers cannot be unloaded.

## External plugins over BRP

The header displays a freshly selected loopback BRP port. `--port` may specify
an already free port; no default port is used. `--endpoint-file FILE` publishes
the URL for external processes. Built-in Bevy methods and `rpc.discover` remain
available. Custom methods:

- `agent/status`: PID, busy state, native generation, extensions, conversation.
- `agent/prompt`: `{"text":"..."}` submits a prompt if idle.
- `agent/extend`: register/replace a model tool at runtime.
- `agent/unextend`: `{"name":"external_echo"}` removes a tool.

Example registration (the callback server must already be running on a free port):

```json
{"jsonrpc":"2.0","id":1,"method":"agent/extend","params":{
  "name":"external_echo","description":"Echo text from an external process",
  "parameters":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false},
  "url":"http://127.0.0.1:YOUR_FREE_PORT/"
}}
```

When the model calls that tool, the host POSTs `{"name":"external_echo",
"arguments":{"text":"hello"}}` to the callback. Its JSON response is returned
as a tool result. Tool definitions refresh at every model round. Callbacks
must be explicit loopback HTTP URLs. Plugins can be written in any language.

## Verification

```sh
cargo +1.98.1 test --locked
cargo +1.98.1 clippy --locked --all-targets -- -D warnings
python3 verify_live.py  # requires OPENAI_API_KEY; three cheap model prompts
```

The live test starts the actual TUI under a PTY, removes `dx` from its private
PATH, checks a real model's read-file call and answer, asks it to rewrite and
hot-patch its native plugin, and registers/calls a Python extension over BRP.
It asserts the same agent PID, and saves real ratatui frame snapshots and a
credential-free evidence report in ignored `verification-artifacts/`.
`--snapshot FILE` is an opt-in export of the actual rendered frame.

This is a trusted local coding agent, **not a sandbox**. File/shell/native tools
run with the user's privileges and BRP has no authentication. Never expose the
port or register untrusted native code or callback peers. Shell output is
capped, calls time out, but shell descendants may outlive their parent. There
is one conversation and no streaming, approval UI, persistence, context
compaction, or cancellation beyond exiting. BRP brings sizeable transitive
Bevy dev-tools dependencies even without rendering enabled. The ephemeral
port probe has a small bind race; startup fails rather than sharing a port.
