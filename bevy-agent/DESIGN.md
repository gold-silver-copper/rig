# bevy-agent

A minimal coding agent in the spirit of [pi](https://github.com/earendil-works/pi)
(read at commit `955cc6665ee3986c6a033db52200779310d10dfd`). Rig does the model
layer and as much of the infrastructure as it can: memory, MCP, telemetry,
cassettes and code mode. Bevy 0.20.0-rc.2 runs everything else, ratatui draws
the TUI, and a small supervisor restarts the agent into a freshly built binary
when it reloads itself. rig-agent and rig-ecs are not used.

```sh
cd bevy-agent
cargo run -- --model anthropic:claude-opus-5-5          # or openai:gpt-6.1-sol, gemini:gemini-3.8-flash, deepseek:deepseek-flash
cargo run -- --continue                                  # resume the last session
cargo run -- --mcp "facts=target/debug/examples/mcp_server" --otlp http://127.0.0.1:4318
cargo run -- --record demo    /  --replay demo --headless
cargo run -- eval evals/providers.json [--record]
```

`--disable brp,remote,durable,codemode,mcp,telemetry` leaves plugins out. In
the TUI, Enter sends a prompt (it waits in a queue while the agent is busy),
Alt+Enter adds a line and Esc cancels. Ctrl+C clears the editor or quits,
Ctrl+D quits, Ctrl+P switches to the next model and Ctrl+O expands tool
output. `/help` lists the commands.

## The Bevy-Rig glue (`src/glue.rs`)

This is the only module that uses both Bevy and Rig. It is meant as the seed
of the rig-ecs rewrite, and its whole API is:

- **An agent is an entity.** `Agent` is a marker that requires three
  components:
  - `Conversation(Vec<Message>)`: the Rig messages.
  - `Turn`: `Idle`, `Request`, or `Tools { calls, results }`. It is
    serializable, so a turn can be saved and resumed; a saved `Request` is
    simply sent again.
  - `Instructions(String)`: the system prompt.

  Each agent also has a `Model` component (a `vendor:model` reference and
  its `DynModel`), built by the `ModelFactory` resource.
- **Plugins drive agents with components.**
  - To start a turn, set `Turn::Request`.
  - `send_requests` streams the request on the tokio runtime in
    `RigRuntime`. `receive_replies` appends the reply and writes
    `AgentEvent` messages: `Text`, `Replied`, `CallStarted`, `CallFinished`
    and `Ended`.
- **A tool call is an entity too.** It has `ToolCall(message::ToolCall)` and
  a `CallOf(agent)` relationship. Its target, `ToolCalls`, has
  `linked_spawn`, so calls are despawned with their agent.
  - Calls run one at a time, in the model's order.
  - A call ends when anything inserts `ToolOutput(String)`.
  - The glue runs calls to Rig `DynamicTool`s itself: `app.add_rig_tool`,
    run on tokio. A plugin can instead register a definition with
    `app.add_system_tool` and answer the calls from its own systems; reload,
    external BRP tools and code mode do this.
  - `Nested` marks calls made by another tool rather than the model. These
    never enter the conversation.
- **Other pieces.** `Tools` is the registry: definitions in a stable order,
  plus an optional `DynamicTool`. `cancel(world, agent)` stops a turn and
  keeps the history canonical. `TurnSpan` and `CallSpan` hold the tracing
  spans: `invoke_agent`, and `execute_tool` nested under it. Rig's own
  completion span becomes a child of the turn's span.

Nothing is global per agent. The TUI session, remote sessions and eval runs
are all agent entities in the same world, and each runs its own turn
concurrently. The glue has no knowledge of any of those plugins.

The glue adds only one abstraction, the `ModelFactory` closure, and only
because two parts need it: the default factory reads credentials from the
environment, and the cassette plugin sends through a recording proxy. It is
also where provider quirks the agent relies on live: `store: false` for
OpenAI Responses, and dropping stale thinking blocks on Anthropic (see the
Rig changes).

## The core (`CorePlugins`)

The core is the glue, sessions, the native tools, reload and the TUI:

- `session.rs` keeps what an agent is besides its Rig state:
  - `SessionId` and `Origin` (Tui, Remote or Eval).
  - `Transcript`, the view people see, built from `AgentEvent`s.
  - `Prompts`, the queue plus notes for the model.
  - `Inputs`, every line typed.
  - `Usage`.
  - `submit` handles slash commands and prompts. `SlashCommand` messages
    carry plugin commands.
  - `SessionStore` saves a session with Rig's `FileConversationMemory` for
    the conversation and a small JSON file for the rest. Reload uses it, and
    so does the durable plugin.
- `tools/` holds pi's four tools, each an ordinary Bevy plugin that registers
  a Rig `DynamicTool`. The descriptions, schemas, truncation limits and edit
  semantics follow pi:
  - `read`: offset and limit, capped at 2000 lines or 50 KB.
  - `write`
  - `edit`: several `edits[].oldText/newText` pairs, each matched uniquely
    against the original file, with no overlaps. It keeps CRLF, and falls
    back to a loose whole-line match for trailing whitespace, curly quotes
    and dashes.
  - `bash`: no default timeout, and the last 2000 lines or 50 KB of output,
    with the full output saved to a temp file when it is cut.

  An observer on `Insert<ToolOutput>` redacts credential-like environment
  values from every tool output.
- `reload.rs` and `supervisor.rs` handle reload:
  - The `reload` tool and `/reload` run `cargo build` on the agent's own
    manifest. The agent's crates, its manifests and the Rig crates it uses
    by path are all built this way.
  - If the build fails, the rendered compiler errors become the tool result
    and nothing restarts.
  - If it succeeds, every session and the process state (the BRP port,
    external tool definitions, the list of agents) are saved, and the
    process exits with code 75.
  - The supervisor starts a private copy of the new binary with the same
    command line. The new process restores all agents and answers the open
    `reload` call, so the turn continues. Queued prompts survive too.
  - A binary counts as started once it has run for a second and its BRP
    port answers. If one exits before that, the supervisor restores the
    session snapshot, writes a note with its exit status and the end of its
    stderr, and restarts the last binary that started. The model reads that
    note as the result of its `reload` call.
- `tui.rs` shows the `Origin::Tui` agent. Its footer is modelled on pi's:
  cwd, model, activity, token usage, queue length, BRP port and generation.

## Plugins (`src/plugins/`) and how pi's features map onto Rig

| pi package | Plugin | Built on |
| --- | --- | --- |
| server, protocol, client | `brp` + `remote` | Bevy's BRP. `agent.serve+watch` / `agent.tool_result` let external processes add tools. `session.create/list/prompt/transcript/cancel/close` drive agent entities remotely. No second protocol. |
| mcp | `mcp` | `rig-rmcp` (`tools_from_server`, `McpTool` → `DynamicTool`) over rmcp's child-process transport. Tools are named `<server>__<tool>`. Servers connect before the first frame. |
| durable, session-backends | `durable` | Rig's `ConversationMemory`, through the new file backend in `rig-memory`. Details below this table. |
| telemetry | `telemetry` | `tracing` spans from the glue plus Rig's GenAI spans (`rig_core::telemetry`), exported over OTLP/HTTP JSON with `tracing-opentelemetry`. |
| evals | `eval` | Suites of JSON cases run as `Origin::Eval` agent entities, recorded and replayed with `rig-cassette`'s HTTP engine. |
| codemode | `codemode` | The new `rig-codemode` crate. |
| (debugging) | `cassette` | `rig-cassette`'s HTTP engine. Details below this table. |

**Durable sessions.**
- Messages are appended to the file store as they arrive.
- `--continue` and `--resume` bring a session back, and `/sessions` and
  `/resume` do the same from the TUI.
- A restored history is checked with `rig_core::transcript`. Calls a crash
  left unanswered get a result saying so.
- `/compact`, and a token budget, use rig-memory's `TokenWindowMemory` with
  `HeuristicTokenCounter` and a `TemplateCompactor` summary.

**Code mode.** The `codemode` tool runs JavaScript whose only capability is
`tools.<name>(args)`. Each call the script makes is a `Nested` tool-call
entity of the same agent, answered by the same systems as the model's own
calls.

**Cassette record and replay.**
- `--record NAME` sends every model request through the engine's recording
  proxy and writes one scrubbed cassette per provider, together with the
  lines that were typed.
- `--replay NAME` answers the same requests with no network and no keys,
  types the recorded lines again, and compares tool calls and the final
  answer. `--headless` exits 0 only on a match.

**Deterministic requests.** Requests stay the same from one machine to the
next:
- In cassette mode the system prompt has no timestamps and uses paths
  relative to the working directory.
- Tools are listed in a stable order.
- Evals run in a fixed copy of their workspace under `target/eval`.

From pi the agent takes:
- the tool set, tool descriptions and schemas;
- the system-prompt layout (`<tools>`, `<rules>`, `<cwd>` sections) and
  guidelines;
- the edit semantics;
- the TUI keys and slash-command names;
- queued follow-up prompts;
- the code-mode script environment (`tools`, `text`, `console`, `exit`,
  `ALL_TOOLS`, return value appended).

Switching providers mid-conversation works the way pi's `transformMessages`
does. Rig already seals reasoning to the service that issued it and replays
tool-call IDs across providers.

## Rig changes, one commit each, with tests and docs

- **Model IDs.** `GEMINI_3_8_FLASH` and `DEEPSEEK_V4_1_FLASH` were added,
  with IDs taken from the providers' model-list APIs:
  - Gemini's model list includes `gemini-3.8-flash`.
  - DeepSeek's model list reports `deepseek-flash`, named DeepSeek-V4.1-Flash.

  `ProviderId::resolve` also reads `gemini` as `gcp.gemini`, which makes
  `/model gemini:…` work.
- **`ProviderConfig::with_base_url` and a public
  `ProviderConfig::completion_model`.** A registry selection can now point at
  a recording proxy or a gateway.
- **`transcript::answer_unanswered`.** It repairs a history cut short
  mid-turn; durable sessions use it.
- **rig-memory `FileConversationMemory`** (native-only, `file` feature). It
  keeps one JSON Lines file per conversation, syncs appends, skips and cuts
  a torn final line, replaces a history atomically for compaction, and
  lists conversations newest first. Rig previously shipped only an
  in-memory backend.
- **`rig-codemode`** (new crate; facade feature `codemode`, at
  `rig::tool::codemode`). It runs code in a fresh QuickJS interpreter with no
  I/O, a memory limit and an optional timeout.
  - `CodeMode::execute(code, callback)` lets a host route nested calls; this
    is what the agent does, turning them into entities.
  - `into_tool(dynamic_tools)` is the standalone form.
  - `definition()` renders TypeScript declarations for the nested tools.
- **rig-cassette `try_start_at` / `try_finish`.** The engine reports bad
  fixtures, refused recordings and unplayed interactions by panicking,
  which fits tests. An app recording or replaying a live session needs them
  as errors.
- **`AnthropicConfig::with_thinking_prefix_mismatch`.** This fixes a bug
  found in the live reload demo. Claude Opus 5.5 binds each thinking
  block's signature to the tool list. The first request after a reload
  offered a different tool list, because an MCP server had not connected
  yet, and every later request was then rejected with a 400. The option
  sets `thinking.block_binding.prefix_mismatch_behavior: "drop_block"` and
  its beta flag. I checked this against the live API before adding it.
  `tests/thinking_binding.rs` replays the failing sequence from a recorded
  cassette. It fails without the option, because the request no longer
  matches.

## Effect-level recording for the rig-ecs rewrite

The agent records at the HTTP layer, which reproduces a session exactly but
only for the same tool results. To record and replay an agent world at the
effect level, the rig-ecs rewrite would need the following:
- Model requests and replies, and tool calls and outputs, emitted as
  effects keyed by agent entity and turn.
- Replay that feeds recorded tool outputs instead of running the tools, so
  a session reproduces without its workspace.
- Stable identities for entities and calls across processes. Session ids
  and call ids would do; Bevy entity ids would not.
- An ordering record for concurrent agents.
- Hooks where the glue's `AgentEvent`s already sit, so recording is just
  another observer.

The `effect_log` recorder in rig-cassette can hold such a log. It is not used
here because its adapters are the banned `agent` and `ecs` features.

## Tried and abandoned

- **Restarting by exec'ing the new binary in place.** If that binary crashes
  on startup, no process is left to fall back to the old one.
- **A single global `Session` resource, as in my first version.** Replaced
  by agent entities.
- **Saving the session after `App::run` returns.** `App::run` moves the app
  out, so the save never ran. Sessions now save in `Last` on `AppExit`.
- **Connecting MCP servers asynchronously after startup.** Requests made
  before a server connected had a different tool list, which triggered the
  Anthropic failure above.
- **Passing the command line only to the first process.** After a reload,
  MCP and telemetry were gone. Every generation now gets it.
- **A callback plugin protocol (an HTTP server per plugin) and a polling
  protocol for external tools.** Replaced by a BRP `+watch` stream, whose
  closing tells the plugin to reconnect.
- **ratatui's paragraph wrapping.** Measuring wrapped height needs an
  unstable feature, so the TUI wraps text itself.

## Known limitations

- **The supervisor updates only on a manual restart.** Changes to its own
  code need you to restart the agent.
- **Two restart gaps.** A request that is in flight during a restart is
  sent again. A new binary that hangs, rather than crashing or failing to
  open its port, is not detected.
- **Tools act on the process's working directory.** Concurrent sessions
  share it, and evals run their cases one at a time in their own workspace.
- **Cassette replay only covers the TUI session.** It replays what was typed
  into that session and needs the same workspace contents. A recording
  does not continue across a reload.
- **Pi features left out:** code mode's `store`/`load`, images,
  branching/forking sessions and an LLM-written compaction summary.
- **Anthropic models Rig does not know need an explicit `max_tokens`.**
  This is Rig's deliberate choice.
- **Output redaction and sandboxing are limited.** Redaction only covers
  credential-like environment variables. During testing, before redaction
  existed, a model ran `env` and two API keys reached another provider.
  `bash` runs with the user's permissions.

## Verified live (`evidence/`)

These ran against real APIs in the TUI, in one session on a scratch copy of
this tree:
- A DeepSeek turn made tool calls and gave a final answer.
- The same conversation continued on GPT-6.1 Sol, Claude Opus 5.5 and
  Gemini 3.8 Flash, each with a tool call and each recalling earlier results.
- Gemini called the MCP server's tool, and DeepSeek completed a task in
  code mode through nested `read` calls.
- The external BRP `shout` tool worked before a reload, after it and after
  a fallback.
- Opus edited `src/tools/read.rs`, reloaded, and used `[read v2]` in the
  same turn. A prompt queued during the build ran afterwards.
- A compile error came back to the model while the old agent kept running
  (the generation number did not change).
- A panic on startup fell back to the last good binary, and the model
  received the panic message.
- A `/model` choice survived `/reload`.
- After quitting, `--continue` resumed the session, and the model recalled
  its earlier results.
- OTLP spans (`invoke_agent` → `chat_streaming` / `execute_tool`) reached the
  external collector before and after a reload.
- A live Gemini session recorded with `--record` replayed offline with the
  same tool calls and final answer.
- The four-provider eval suite passed from its cassettes with every API key
  unset. CI runs it as `tests/replay.rs`, next to the Anthropic regression
  replay.
