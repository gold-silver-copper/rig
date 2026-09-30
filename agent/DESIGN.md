# bevy-agent: design notes

A minimal pi-style coding agent: Bevy 0.20.0-rc.2 (the newest Bevy on
crates.io) runs the agent loop, Rig (`rig-core` from this repository, by path)
talks to the models, ratatui draws the terminal UI, and the Bevy Remote
Protocol (BRP) is the plugin system for other processes. The agent can edit its
own source, including its manifest and the Rig crates, and reload into the
result.

```sh
cd agent && cargo build
./target/debug/bevy-agent --model opus        # or sol, gemini, deepseek, or any vendor:model
./target/debug/bevy-agent --continue          # resume the saved session
python3 examples/brp_plugin.py PORT           # an external tool; the TUI header shows PORT
```

In the TUI: Enter sends (prompts typed while busy are queued), `/model [name]`,
`/reload`, `/quit`, PgUp/PgDn, ↑/↓ history. State lives in `.bevy-agent/`
(`--state-dir`).

## Shape

| file | role |
| --- | --- |
| `main.rs` | options; runs the supervisor, or with `--child` the Bevy app |
| `supervisor.rs` | starts children, restarts into rebuilt binaries, falls back |
| `session.rs` | everything that survives a restart (`session.json`) |
| `agent.rs` | the loop: conversation, queue, model, tool calls as entities |
| `reload.rs` | the `reload` tool and `/reload` |
| `brp.rs` | BRP server and the external-plugin methods |
| `tui.rs` | ratatui UI |
| `plugins/` | native plugins: `coding` (read, write, edit, bash) and a sample `greeting` |

**The loop.** A `Turn` resource is `Idle`, `Thinking` (a Rig call running on a
Tokio runtime held as a resource) or `Tools`. A response's tool calls become
entities with a `PendingCall` component; the tool's handler, a Bevy system
registered with `app.add_tool(name, description, schema, system)`, answers by
inserting `CallOutput` (directly, or later through `ToolTasks`, which runs a
future and inserts its result). When every call of the batch has an output the
results go back to the model. Native plugins are ordinary Bevy plugins that do
nothing more than that; changing one means editing it and reloading.

**Models.** A selection is an alias or any Rig registry reference
(`vendor[/format]:model`); `resolve_model` turns it into a `DynModel<Completion>`
through `ProviderRef::completion_model`, with credentials from the
environment. The conversation is a plain `Vec<rig_core::message::Message>`, so
switching models only swaps the `DynModel`; Rig decides which reasoning each
provider can replay. Aliases: `sol` = `openai:gpt-6.1-sol` (the OpenAI preset
uses the Responses API), `opus` = `anthropic:claude-opus-5-5`, `gemini` =
`gcp.gemini:gemini-3.8-flash`, `deepseek` = `deepseek:deepseek-flash`. The
Gemini and DeepSeek IDs come from their model-list APIs: Gemini lists
`models/gemini-3.8-flash` ("Gemini 3.8 Flash"); DeepSeek lists
`DeepSeek-V4.1-Flash` under the id `deepseek-flash`.

**Reload.** The binary runs twice over: the process the user starts is a small
supervisor that picks the BRP port (a free one, or `--brp-port`), copies itself
into `state/bin/`, and starts that copy with `--child`. A reload (the `reload`
tool, or `/reload`):

1. waits for the other calls of the batch, so edits made alongside it are on
   disk, then runs `cargo build --message-format=json-diagnostic-short` in the
   agent's own crate (its path is compiled in);
2. on failure answers the call with the compiler errors, and nothing restarts;
3. on success copies the new executable into `state/bin/`, saves the session
   (conversation, transcript, queue, model, unfinished tool batch, which call
   was the reload, external tool registrations), writes `reload.json` and exits
   with code 75;
4. the supervisor starts the new binary with `--resume --reload-ok`. It answers
   the pending `reload` call with "Reload succeeded" and the turn continues.

A child is *ready* once its BRP server accepts connections (it probes itself)
and it has written `state/ready`. If a new binary exits before that, cannot be
spawned, or is not ready within 30 s (then it is killed by PID), the supervisor
restarts the last binary that became ready with `--reload-failed <reason>`,
where the reason includes the tail of the child's stderr. The `reload` call
then reports the failure, e.g. the panic message. A `/reload` answers no call,
so its outcome is prefixed to the next prompt as an `[agent runtime: …]` note.

**BRP plugins.** Children serve BRP on the session's port with Bevy's built-in
`world.*` methods plus `agent.register_tool`, `agent.unregister_tool`,
`agent.tool_calls+watch` (an SSE stream of calls for one plugin),
`agent.tool_result`, `agent.prompt` and `agent.status`. A plugin needs only an
HTTP client (`examples/brp_plugin.py` uses Python's standard library).
Registrations are saved with the session, so a plugin's tools stay offered
across a reload; its watch stream ends when the old process exits, it
reconnects to the same port and registers again (registering an existing name
replaces it, so there is no error), and calls it had not answered are sent
again.

## Decisions, and what was tried and dropped

- **Supervisor instead of `exec`.** Replacing the process in place keeps the
  PID and the terminal, but nothing would be left to fall back when the new
  binary crashes. A parent process that owns only the restart policy is the
  smallest thing that can.
- **No socket hand-over.** Passing the listening socket from the supervisor to
  each child would avoid refused connections during the ~1 s restart, but
  plugins must reconnect anyway because the watch stream ends, and std binds
  with `SO_REUSEADDR`, so rebinding the same port is reliable.
- **`bevy_remote` as a direct dependency**, not through `bevy/bevy_remote`,
  which turns on its default features and with them the renderer. Bevy itself
  is `default-features = false` with `MinimalPlugins` and a 16 ms
  `ScheduleRunnerPlugin` loop.
- **Toolchain.** Bevy 0.20.0-rc.2 needs rustc 1.96; Rig's workspace pins 1.95,
  so `agent/rust-toolchain.toml` pins the installed 1.98.1. The agent is its own
  Cargo workspace (own lockfile and `target/`) that uses Rig by path, so Rig's
  lockfile is untouched and an incremental rebuild takes about 2 s.
- **Unary model calls.** Streaming would make the TUI livelier but adds a
  second event path; a spinner is enough for a minimal agent.
- **Anthropic thinking-block binding.** Registering a BRP tool mid-conversation
  made every Opus 5.5 request fail with a 400: the model binds thinking blocks
  to the tools list. The first fix lived in the agent (a beta flag and an extra
  request field for every Anthropic model). Probing the API showed that only
  Opus 5.5, Fable 5.1 and Fable 5 accept the field; the others require
  `thinking.type`. That per-model knowledge belongs in Rig, so the fix moved
  there and the agent code was removed.
- **Gemini's registry vendor is `gcp.gemini`** (the GenAI semantic-convention
  name), so `gemini:…` does not parse. The agent's alias covers it; changing
  how Rig's registry resolves names was out of scope.
- `plugin_group!` for the native plugins needs `Default` plugins; a plain
  plugin that adds the others is simpler.

## Rig changes

- `feat(providers)`: `gemini::completion::GEMINI_3_8_FLASH` and
  `deepseek::DEEPSEEK_V4_1_FLASH`, with the IDs the model-list APIs return.
- `fix(anthropic)`: for models that bind thinking blocks (Opus 5.5, Fable 5.1,
  Fable 5), a request that replays thinking asks the API to drop stale blocks
  (`thinking.block_binding.prefix_mismatch_behavior = "drop_block"` plus the
  `thinking-binding-controls-2026-08-01` beta flag) instead of failing. Without
  it, any Rig user whose tools change mid-conversation (rig-agent's dynamic
  tools included) gets a 400 on these models. Callers' own settings are kept,
  and gateways opt out through a dialect quirk. Covered by wire tests.

## Known limitations

- The supervisor is not reloaded: changes to `supervisor.rs` take effect the
  next time the user starts the agent.
- Only startup failures fall back. A binary that crashes later ends the
  session; the session is saved at every turn end, so `--continue` resumes it.
- Calls to external tools that are in flight when the process restarts are
  answered with an "interrupted" error. The reload waits for the rest of its
  own batch, so this only affects calls from other sources.
- An older fallback binary reads a newer session through serde defaults; a
  change to the meaning of an existing field would not be understood.
- No streaming, no way to cancel a running turn, and one agent per state
  directory. BRP has no authentication; it binds to 127.0.0.1 only.
- The screen flickers once per restart, as the old process leaves the
  alternate screen and the new one enters it.
