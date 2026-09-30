# bevy-agent

A minimal coding agent in the spirit of pi. Rig (`rig-core` only) talks to
the model. Bevy 0.20.0-rc.2 runs everything else as plugins. ratatui draws
the terminal UI. The agent can rebuild itself from its own source and
restart into the new binary without losing the conversation.

```sh
cd bevy-agent
cargo run -- --model anthropic:claude-opus-5-5   # or openai:gpt-6.1-sol, gemini:gemini-3.8-flash, deepseek:deepseek-flash
python3 examples/brp_plugin.py <brp port from the status line>   # optional external plugin
```

In the TUI, Enter sends a prompt, or queues it while the agent is busy, and
Esc cancels the turn. The arrow keys and PgUp/PgDn scroll.
`/model vendor:model` switches models, `/model` lists
the four presets, `/reload` rebuilds and restarts, and `/quit` exits.
`--continue` resumes the last session and `--state-dir` moves it from
`./.bevy-agent`.

## How it works

**Two processes.** The binary you start is a small supervisor
(`supervisor.rs`). It copies itself into the state directory and runs that
copy as the agent, a child process that shares the terminal. The copy
matters: a later `cargo build` cannot overwrite the binary the supervisor
falls back to. The agent's stderr goes to `agent.stderr.log`, so panics
don't garble the TUI.

**One Bevy app.** The agent is a headless Bevy app built from
`MinimalPlugins` running at 30 Hz, plus five plugins:

- `AgentPlugin` (`agent.rs`) runs the turn loop as a state machine in the
  `Session` resource: `Idle`, then `Request`, then `Tools { calls, results }`,
  then back to `Request`, and finally `Idle`. Model calls run on a tokio
  runtime owned by the `Llm` resource. Streamed text arrives over a channel
  and is polled each frame. Tool calls run one at a time, in order.
- `TuiPlugin` (`tui.rs`) reads keys without blocking in `PreUpdate` and
  draws in `PostUpdate`.
- `ReloadPlugin` (`reload.rs`) provides the `reload` tool and handles `/reload`.
- `BrpPlugin` (`brp.rs`) is Bevy's `RemotePlugin` plus `RemoteHttpPlugin`
  on the session's port, with four agent methods.
- `NativePlugins` (`plugins/`): `read`, `write`, `edit` and `bash`, one
  ordinary Bevy plugin each.

**Tools are one-shot systems.** A plugin calls
`app.add_tool(name, description, schema, system)`. The system takes
`In<serde_json::Value>` and returns a `ToolReply`: either output now, or a
channel the output arrives on later, which `ToolReply::spawn` fills from a
worker thread. Because tools are systems, they can use any resource or
query. Every tool output passes through `Secrets::redact` before it is
logged or sent to the model.

**Everything that must survive a restart is `Session`.** That is the model
choice, the BRP port, the Rig message history, the transcript, the queued
prompts and commands, the turn state, external tool definitions, and notes
for the model. It is written to `session.json` at most once a second while
it changes, and again in the frame the app exits.

**Reload.** Both the `reload` tool and `/reload` run `cargo build
--message-format=json-render-diagnostics` on the agent's own manifest
(`CARGO_MANIFEST_DIR`) on a worker thread. Rig's crates are path
dependencies, so edits to them and to `Cargo.toml` are rebuilt too.

- If the build fails, the rendered `error` diagnostics become the tool
  result, or a transcript error for `/reload`. The agent keeps running.
- If it succeeds, the agent copies the new executable into the state
  directory, writes its path to `next-binary`, saves the session and exits
  with code 75. The `reload` call is left unanswered in the saved turn.
- The supervisor snapshots the session and starts the new binary. The new
  process finds the open `reload` call and answers it with "Reloaded: …".
  The turn then continues, so the model can use its change in the same turn.
- A queued `/reload` waits until the agent is idle. No turn starts while it
  is building.

**Fallback.** A binary counts as started once it has run for a second *and*
its BRP port answers. If it exits before that, the supervisor does four
things: it restores the session snapshot, writes a note with the exit status
and the tail of stderr, deletes the broken binary, and restarts the last
binary that started. That process answers the open `reload` call with the
note, so the model learns what happened. If the reload came from `/reload`,
the note is shown in the TUI and also prepended to the next prompt. A binary
that crashes after starting is restarted with a similar note. The supervisor
gives up after three crashes that each come within 30 s of a start.

**BRP plugins.** An external process is a plain BRP client and does not
need to run a server:

- `agent.serve+watch {"tools": [...]}` registers tools, which replaces
  earlier definitions with the same name. It returns an SSE stream of
  `{call_id, tool, arguments}`.
- `agent.tool_result {call_id, output}` answers a call.
- `agent.prompt {text}` sends a prompt or command exactly as typing it would.
- `agent.transcript {since}` reads the transcript.

Bevy's built-in `world.*` methods are available too. A reload closes the
stream. The plugin opens it again on the same port, which is chosen once and
kept in the session, and so re-registers. Definitions persist in the
session, so the tools stay in the model's tool list while the plugin
reconnects. A call waits up to 30 s for its plugin to connect.
`examples/brp_plugin.py` is a complete plugin in about 60 lines of standard
library Python.

**Providers.** A model is a `rig_core::providers::registry::ProviderRef`
string, for example `openai:gpt-6.1-sol`. `ProviderRef::completion_model()`
builds it with credentials from the environment. The whole Rig surface the
agent uses is `ProviderRef`, `DynModel<Completion>::stream`, `Streamed::finish`
and the message types. Switching models replaces the `DynModel` and keeps the
`Vec<Message>`. Rig's wires drop reasoning that another provider sealed. The
OpenAI preset already routes to the Responses API, which GPT-6.1 Sol needs
for function tools.

## Decisions

- **Separate Cargo workspace and toolchain.** Bevy 0.20.0-rc.2 needs
  rustc 1.96, but Rig pins 1.95. `bevy-agent/` has its own `[workspace]`,
  `Cargo.lock` and `rust-toolchain.toml` (1.98.1). Rig's crates are path
  dependencies, so its lockfile and CI are untouched.
- **`bevy_remote` as a direct dependency** with only `http`. Enabling the
  `bevy` facade's `bevy_remote` feature turns on `bevy_render` (wgpu) on
  native targets.
- **Model IDs came from each provider's model-list API.** Gemini's
  `/v1beta/models` lists `gemini-3.8-flash` (display name "Gemini 3.8
  Flash"). DeepSeek's `/models` lists `deepseek-flash`, named
  "DeepSeek-V4.1-Flash". DeepSeek still serves the old `deepseek-v4-flash`
  as an alias.
- **Tool calls run one at a time.** This is simpler, and it makes
  `[edit, reload]` in one message do the right thing.
- **Supervisor and child instead of exec-in-place.** The supervisor is the
  only process that survives a crashing new binary.

## Tried and abandoned

- **Restarting by `exec`ing the new binary in place.** Nothing is left to
  fall back to when that binary crashes. A pre-flight "--check" run of the
  new binary can't catch crashes that happen during real startup either.
- **A callback plugin protocol**, where the plugin gives a URL and the agent
  calls it. Every plugin would need its own HTTP server and free port.
  Polling with an instant method was also dropped. The `+watch` stream
  pushes calls, and its closing is the plugin's reconnect signal.
- **Saving the session after `App::run()` returns.** `App::run` moves the
  app out and leaves an empty `App`, so that save never ran. Live testing
  caught the resulting lost notice. The session is now saved in `Last` on
  `AppExit`.
- **Handing BRP-injected text straight to the queue.** `/model …` sent by a
  plugin reached the model as a prompt. BRP now goes through the same
  `submit` path as the keyboard.
- **ratatui's `Paragraph::wrap`.** The wrapped height needed for pinning the
  view to the bottom is only available behind an unstable feature, so the
  TUI wraps text itself.

## Rig changes

One commit, separate from the agent:

- `gemini::completion::GEMINI_3_8_FLASH` (`gemini-3.8-flash`) and
  `deepseek::DEEPSEEK_V4_1_FLASH` (`deepseek-flash`). rig-core had
  constants for neither model. The IDs are the ones the model-list APIs
  report.
- `ProviderId::resolve` reads a dotted vendor from its last segment when
  that is unambiguous, so `gemini:gemini-3.8-flash` works. Before this,
  typing `/model gemini:…` failed with "no registered provider is named
  `gemini`", because Gemini's registry vendor is its telemetry name
  `gcp.gemini`. References still write back qualified
  (`gcp.gemini/gemini:…`). A registry test covers this.

Observed in Rig and left alone: Anthropic models Rig does not know require
an explicit `max_tokens`, which is a deliberate choice. Observed in Bevy
0.20.0-rc.2: `process_remote_requests` returns instead of continuing after
an unknown method, which only defers the rest of that frame's requests.

## Verified live

All of the following ran against the real APIs in the TUI (driven through
tmux), in one session on a scratch copy of this tree. `evidence/` holds the
session's transcript, screen captures with the status line (model and
generation), and the Python plugin's log.

1. On DeepSeek V4.1 Flash, a prompt produced `bash` and `read` calls and a
   final answer.
2. The same conversation continued with `/model` switches to GPT-6.1 Sol
   (`read` plus the BRP `shout` tool), Claude Opus 5.5 (`bash`, correctly
   listing every earlier tool call) and Gemini 3.8 Flash (`bash`). Earlier,
   parallel tool calls also worked on all four models, with switches between
   them.
3. Claude Opus 5.5 edited `src/plugins/read.rs` to prefix `[read v2]`,
   called `reload` (generation 0 to 1), and read `a.txt` in the same turn,
   getting `[read v2]\nhello`. A prompt queued during the build ran after
   the restart. It called the BRP `shout` tool, which the Python plugin
   answered after re-registering.
4. A deliberate type error made `reload` return the rustc `E0308`
   diagnostic. The generation stayed at 1.
5. A `panic!` in `ReadPlugin::build` built and then crashed on startup
   (exit 101). The supervisor fell back (generation 3). `reload` returned
   the fallback note with the panic message, and the fallback binary still
   had the `[read v2]` behaviour.
6. After `/model deepseek:deepseek-flash` and `/reload`, the status line
   showed DeepSeek at the next generation. `read` and the BRP tool still
   worked.

## Known limitations

- The supervisor's own code changes only when the user restarts it, because
  a reload replaces only the agent process.
- A model request that is in flight during a restart is sent again. A new
  binary that hangs, instead of crashing or failing to open its BRP port,
  is not detected.
- The external plugin protocol hands a call to one connected plugin. If that
  plugin disconnects before answering, the call waits until Esc.
- The TUI assumes single-width characters, has no multi-line editing, and
  shows only the first lines of long tool output.
- There is no context compaction, no images, and no sandbox: `bash` runs
  with your permissions. Redaction only covers credential-like environment
  variables. It exists because, during testing, a model ran `env` and sent
  two API keys to another provider. That was before redaction was added.
- Each generation's binary (about 140 MB in debug) is copied into the state
  directory. Superseded copies are deleted, and the rest are deleted on exit.
