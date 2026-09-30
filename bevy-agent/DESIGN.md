# bevy-agent

A minimal pi-style coding agent: Bevy `0.20.0-rc.2` (latest on crates.io,
pre-releases included) for the app, `rig-core` from this repository for the
models, ratatui for the terminal UI. About 1,600 lines of Rust.

```sh
cd bevy-agent && cargo build
./target/debug/bevy-agent [--model 1|2|3|4|<vendor:model>] [--port N] [--continue]
python3 examples/brp_plugin.py "$(cat .bevy-agent/brp-port)"   # an external plugin
```

TUI: Enter sends (queued while a turn runs), `/model [choice]`, `/reload`,
`/tools`, `/quit`, PgUp/PgDn scroll, Ctrl+C clears or quits.

## Shape

| file | role |
|---|---|
| `main.rs` | flags, session load, the Bevy `App` (`MinimalPlugins` + ours) |
| `supervisor.rs` | the process the user starts; runs the agent as a child and restarts it |
| `agent.rs` | the whole agent loop as one Bevy system over the `Session` resource |
| `session.rs` | everything that survives a restart, one JSON file |
| `tools.rs` | tool registry; `app.add_tool(...)` for native plugins |
| `plugins/` | native plugins: `coding` (read/write/edit/bash), `greeter` (demo) |
| `brp.rs` | BRP methods for external plugins |
| `reload.rs` | `cargo build`, stage the binary, promote it once it starts |
| `tui.rs` | ratatui input + draw systems |

**Models.** The model is a serializable `rig_core::providers::registry::ProviderRef`
kept in the session, so switching (`--model` at startup, `/model` in the TUI)
never touches the conversation and the choice survives reloads. Each call
builds a `DynModel<Completion>` from it and sends the provider-neutral
`Vec<Message>` history. rig's per-issuer sealed reasoning makes replaying one
provider's turns to another work. OpenAI's dialect defaults to the Responses
API, which `gpt-6.1-sol` needs for function tools. Presets: `openai:gpt-6.1-sol`,
`anthropic:claude-opus-5-5`, `gcp.gemini:gemini-3.8-flash`, `deepseek:deepseek-flash`.
The Gemini and DeepSeek IDs came from the providers' model-list APIs.

**Loop.** One `drive` system per frame: poll the model task (Bevy `IoTaskPool`;
rig-reqwest brings its own tokio runtime), collect finished tools, start the
next step. A reply's tool calls run one at a time, in order, with `reload`
last. Every step is saved to `session.json` (history, transcript, queued
prompts, remote tools, the in-flight turn with its per-call results, model and
port).

**Native plugins** are ordinary Bevy `Plugin`s in the `NativePlugins` group;
they register tools on the `Tools` resource. A tool body runs on its own
thread. Changing one means editing the source and reloading.

**Reload** (the `reload` tool, `/reload`): run `cargo build` on the agent's own
source while the old agent keeps running. The source path is compiled in and
the rig crates are path dependencies, so the agent can edit itself, its
manifests, and rig. A failed build returns only the `error` diagnostics as the
tool result. A successful one copies the binary to `bin/candidate`, saves the
session with the `reload` call still open, and exits with code 75. The
supervisor starts the candidate. After 10 frames the candidate promotes
itself (`good` → `prev`, `candidate` → `good`). On startup the agent answers
the open `reload` call ("succeeded") and the turn continues where it left off.

**Fallback.** If the candidate exits before promoting itself, the supervisor
deletes it, writes a note (exit status and the last lines of stderr), and
starts `bin/good`. That binary answers the open `reload` call with the note. A
promoted binary that crashes later is rolled back to `bin/prev` the same way.
Calls that were in flight during a crash are answered "interrupted", so the
history stays valid for every provider.

**External plugins over BRP.** `RemotePlugin` + `RemoteHttpPlugin` on a free
port, which is picked once and kept in the session. Five methods:
`agent.register_tools` (idempotent; it replaces the plugin's tools),
`agent.tool_calls+watch` (a BRP watching method: an SSE stream of calls for
that plugin), `agent.tool_result`, `agent.prompt`, `agent.status`. Bevy's
`world.*` methods are there too. On reload the stream ends. The plugin
reconnects to the same port and registers again. Registered tools persist in
the session, so the model can call them right after a restart; calls wait
until the plugin is back and are resent after it registers again.

## Verified with real models (transcripts in `evidence/`)

1. Prompt → tool call → answer in the TUI; each of GPT-6.1 Sol, Claude Opus 5.5,
   Gemini 3.8 Flash and DeepSeek V4.1 Flash completes a tool-call turn, switched
   mid-conversation and back, recalling earlier turns (`01`).
2. The model edits `plugins/greeter.rs`, calls `reload`, and uses the new
   greeting in the same turn (`02`).
3. A non-compiling edit returns rustc's errors while the old binary keeps
   serving (`03`; the first attempt showed my error filter cutting off the
   source lines, now fixed and unit-tested).
4. A binary that panics on startup falls back to the last good one, and the
   model is told why (`04`).
5. The external Python plugin's tool works before and after a model-triggered
   reload in one turn. The agent's PID changes; the port and the plugin
   process stay the same (`05`).
6. A prompt queued during a turn survives the restart and then runs (`06`).
7. A final run on the final build covers every path in one turn: a compile
   error, a startup crash with fallback, a self-edit reload, and the BRP tool
   before and after (`07`). In it the model found a bug: it had sent two
   `edit`s of one file in parallel, and they raced. Tool calls now run in
   sequence, and a re-test shows both edits applied (`08`).

## Rig changes

- `gemini::completion::GEMINI_3_8_FLASH` (`gemini-3.8-flash`) and
  `deepseek::DEEPSEEK_V4_1_FLASH` (`deepseek-flash`, which is how DeepSeek's
  model list names V4.1 Flash; `deepseek-v4-flash` and `deepseek-chat` are aliases).
- No rig bugs showed up. Replay across the four providers worked, and a
  warn-level `tracing` log (`$BEVY_AGENT_HOME/agent.log`) stayed empty.

## Tried and abandoned

- **`bevy/bevy_remote` feature**: on native targets it enables bevy_remote's
  default features, which pull in rendering and assets. I depend on
  `bevy_remote` directly with only `http`.
- **`exec()` in place** (same PID, could even keep the socket): nothing would be
  left to fall back if the new binary crashes. I use a small supervisor parent instead.
- **Polling for plugin calls**: replaced with a BRP watching method. Its
  stream ending is also how a plugin learns about a reload.
- **Resolving `gemini:` in rig's registry**: rig's registry deliberately never
  falls back, and a suggestion field would break `SelectionError`. The agent
  accepts a unique dotted vendor suffix itself (`gemini` → `gcp.gemini`).
- **Running a reply's tool calls in parallel**: two edits to one file raced
  and corrupted it. They now run in sequence, as pi does.
- **ratatui's wrapped line count** needs an unstable feature. The transcript is
  hard-wrapped by hand.

## Known limitations

- Unix only (the supervisor uses exit signals); tested on macOS.
- No streaming: each model reply appears when it is complete. No Esc to cancel a turn.
- The supervisor is the binary you launched, so changes to `supervisor.rs`
  take effect only on the next manual start. Rollback goes back one binary only.
- "Started successfully" means it survived 10 frames (~200 ms). Later crashes
  are handled by the `prev` rollback.
- A tool interrupted by a crash is reported, not re-run. A plugin call fails
  after 120 s without an answer. `bash` timeouts kill `sh`, not its process group.
- Self-edits change the real checkout; git is the undo.
- Seen in Bevy, not fixed: `bevy_remote`'s `process_remote_requests` `return`s
  instead of `continue`s on an unknown method, delaying the rest of that
  frame's requests.
