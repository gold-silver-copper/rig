# rig-pi: a minimal self-reloading coding agent on Bevy and Rig

About 1,850 lines of Rust in one crate, plus a 76-line Python example plugin.

```sh
cd agent && cargo build && ./target/debug/rig-pi [--model vendor:model] [--port N] [--state DIR]
python3 examples/brp_plugin.py PORT     # optional: an external plugin adding a `reverse` tool
```

In the TUI, Enter sends a prompt, or queues it while a turn runs. Esc cancels the turn. PageUp and
PageDown scroll. The commands are `/model` (list), `/model <n|vendor:model>`, `/reload`, `/clear`
and `/quit`. The presets are `openai:gpt-6.1-sol`, `anthropic:claude-opus-5-5`,
`gcp.gemini:gemini-3.8-flash` and `deepseek:deepseek-flash`. Any reference that rig's
`ProviderRef` registry resolves also works. Credentials come from the usual environment
variables.

## Shape

- **Two processes, one binary.** `rig-pi` is a supervisor. It runs `rig-pi --child`, which is the
  Bevy app. Both run from copies in `<state>/bin`, so `cargo build` can overwrite `target/` without
  touching a binary the supervisor might need to fall back to (`supervisor.rs`).
- **The Bevy app** is `MinimalPlugins` on a 16 ms `ScheduleRunnerPlugin` loop plus seven plugins:
  `SessionPlugin`, `ToolsPlugin`, `AgentPlugin`, `ReloadPlugin`, `BrpPlugin`, `TuiPlugin` and
  `NativePlugins`. Native plugins are ordinary Bevy plugins in `src/plugins`. They add tools with
  `app.add_tool(name, description, schema, fn)`. `greet` is the example.
- **The turn loop** (`agent.rs`) is one system driving a persisted `Phase`: `Idle`, then `Model`,
  then `Tools { calls, next, results }`. Model calls are `DynModel<Completion>::call` futures on
  Bevy's `IoTaskPool`. rig-reqwest brings its own fallback runtime, so no tokio runtime is needed.
  Tool calls run one at a time on `AsyncComputeTaskPool`. Running them one at a time means every
  point where the turn can stop and resume is well defined.
- **The session** (`session.rs`) holds the model (`ProviderRef`), the rig `Message`s, the
  transcript, the prompt queue, the phase, and the tools BRP plugins registered. It is saved
  atomically to `<state>/session.json` whenever it changes (`resource_changed`). This file is what
  survives a restart.
- **The TUI** (`tui.rs`) is ratatui on crossterm, owned as a non-send resource. It reads input in
  one system and redraws in another, only when something changed. It enables bracketed paste.

## Reload

1. The `reload` tool or `/reload` runs `cargo build` in the agent's own crate (a path baked in with
   `CARGO_MANIFEST_DIR`) on a task-pool thread. The agent keeps running while it builds.
2. If the build fails, the error diagnostics become the tool result. Warnings and cargo progress
   lines are dropped, so the errors are not crowded out. The old binary keeps running.
3. If the build succeeds, the binary is copied to `<state>/bin/rig-pi-<ms>`. Its path goes to
   `<state>/next-binary`, the session is saved, and the agent exits with code 75. At that moment
   the phase is `Tools` with `next` pointing at the `reload` call.
4. The supervisor starts the new binary with `RIG_PI_NOTE` set to "Reload succeeded…". On startup
   the agent sees the unfinished `reload` call and uses the note as its result. The turn then
   continues: the model can call the tool it just changed in the same turn. Queued prompts,
   transcript, model choice and BRP port all come from the session or the supervisor, so they
   carry over.
5. A child counts as up once it writes `<state>/ready`, 10 frames in. That is after every plugin's
   `build` and the startup systems. If the child exits first, or is not up within 60 s, the
   supervisor restarts the last good binary. The note then says what happened: the exit status
   and the tail of the child's stderr, which goes to `<state>/agent.log`. The model gets that
   note as the `reload` result.
6. Once a new binary is up, the supervisor `exec`s into it with `--adopt <child pid>`. It keeps
   its PID, so the child stays its child, and it reaps the child with `waitpid`. This makes a
   reload update the supervisor's own code too. Old binaries are then pruned. Each one is about
   150 MB in debug.

A crash after startup restarts the same binary with a note, up to three times in a row.

## BRP plugins

`BrpPlugin` adds `bevy_remote`'s `RemotePlugin` and `RemoteHttpPlugin` on a fixed port. The
supervisor picks the port once (`--port`, else the one saved in `<state>/port`, else a free one)
and passes the same port to every child. It adds four methods:

- `rig_pi.serve+watch {plugin, session, tools}` is a watching method (SSE). Its first event
  registers the tools. After that it streams `{call_id, name, arguments}` for each call the model
  makes to them. `session` is a per-connection id. When a plugin reconnects with a new id, its
  tools are replaced and the old id is retired, so a stale stream never gets calls. That makes
  re-registering after a reload error-free, and the transcript shows "reconnected".
- `rig_pi.tool_result {call_id, output | error}`
- `rig_pi.prompt {text}` queues a prompt. `rig_pi.status` returns the pid, model, phase, queue,
  tools and recent transcript. The demos used it to wait for idle.

Registered tools are persisted. After a reload the model can call a plugin tool before the plugin
has reconnected. The call waits for the plugin, and fails if the plugin stays away for 30 s.
`examples/brp_plugin.py` is about 50 lines of standard-library Python and reconnects in a loop.

## Decisions, and what was tried and abandoned

- **Bevy 0.20.0-rc.2** is the newest release on crates.io. It needs rustc 1.96, and `wesl`, pulled
  in through `bevy_remote` → `bevy_dev_tools` → `bevy_shader`, needs 1.97.1. rig pins 1.95. So the
  agent is its own Cargo workspace (`agent/`, excluded from rig's lockfile and CI) with its own
  `rust-toolchain.toml` (1.98.1, already installed). It still uses rig's crates by path.
- **`bevy` with default features off, plus `bevy_remote` directly** with only `http`. The `bevy`
  crate's `bevy_remote` feature enables `bevy_remote`'s default features on native, which pull in
  `bevy_render`.
- **No hot patching.** Reload is always rebuild, exit, restart.
- **A supervisor process is required** for crash fallback. Having the agent simply `exec` the new
  binary was rejected: nothing would survive to fall back if the new binary died.
- **A fixed supervisor was tried and abandoned.** At first the supervisor never changed after
  launch. So edits to `supervisor.rs` needed a manual restart, and a bug in its pruning survived
  every reload. The `exec`/`--adopt` handover fixed both. It was checked by changing the reload
  note and reloading twice: the second note used the new text.
- **Non-streaming completions.** `DynModel::call` returns whole responses, and a spinner shows
  progress. Streaming would add complexity for little benefit here.
- **Missing GPT-6.1 Sol reasoning items were investigated.** The session held no OpenAI reasoning
  items. The raw responses showed the API sends none when the model decides not to reason. Hard
  prompts do return encrypted reasoning, and rig keeps it. Not a rig bug.

## Rig changes (separate commits)

- `gemini::completion::GEMINI_3_8_FLASH = "gemini-3.8-flash"` and
  `deepseek::DEEPSEEK_V4_1_FLASH = "deepseek-flash"`. The IDs come from the Gemini and DeepSeek
  model-list APIs. DeepSeek lists V4.1 Flash as `deepseek-flash`; rig's `deepseek-v4-flash` is no
  longer listed. The agent's presets are built from these constants.
- `ProviderId::resolve` accepts a bare family name when the family has exactly one vendor. Gemini's
  vendor is `gcp.gemini`, so `ProviderRef::parse("gemini:gemini-3.8-flash")` used to fail with
  "no registered provider is named `gemini`". It now resolves, and it still displays canonically.
  This is covered by a registry test.

No other rig bugs turned up. All four providers, and switching between them in both directions in
one conversation, worked through the registry. Each provider's reasoning, thought signatures and
item ids were kept with the correct issuer.

## Known limitations

- Unix only: the handover uses `exec` and `waitpid`.
- If a new binary starts but its *supervisor* code is broken, there is no fallback for the
  supervisor. The handover happens only after the new child is up, which proves most of the
  binary.
- `/reload` waits for a running tool to finish before restarting. A restart during the `Model`
  phase sends the request again.
- A tool call delivered to a BRP plugin that dies before answering waits up to 300 s, or 30 s once
  the plugin is gone. Esc cancels.
- The BRP port has no authentication. Any local process can drive the agent and call `world.*`,
  as with any BRP app.
- `bevy_remote`'s HTTP server fails silently when it cannot bind, so the agent checks the port
  first and reports it. Upstream bug, not fixed here: `process_remote_requests` `return`s instead
  of `continue`s on an unknown method, which delays the rest of that frame's requests.
- The transcript wraps by character, not by display width. The `bash` tool's timeout kills `bash`
  but not its grandchildren.
