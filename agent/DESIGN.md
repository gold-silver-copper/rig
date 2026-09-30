# rig-pi: a minimal coding agent on Bevy and Rig

rig-pi is a pi-style coding agent. Bevy runs it, Rig does the model and
agent infrastructure, ratatui draws the terminal, and the Bevy Remote Protocol
(BRP) lets other processes plug in. It can edit its own source, including the
rig crates it depends on by path, and reload into the new build. It is about
4,500 lines of Rust in one crate, plus three small Python example clients.

```sh
cd agent && cargo build && ./target/debug/rig-pi [--model 1-4|vendor:model] [--new] [--port N]
    [--record NAME | --replay NAME] [--mcp FILE] [--otlp URL] [--disable a,b]
./target/debug/rig-pi eval [NAME...]          # replay evals offline
```

The model presets are `openai:gpt-6.1-sol` (Responses API), `anthropic:claude-opus-5-5`,
`gcp.gemini:gemini-3.8-flash` (`gemini:` also works) and `deepseek:deepseek-flash`. The
TUI keys follow pi:

| Key | Action |
|---|---|
| Enter | send; while a turn runs, it steers that turn |
| Alt+Enter | queue a follow-up |
| Esc | interrupt |
| Ctrl+C | clear the input |
| Ctrl+D | quit on an empty input |
| Ctrl+P | next model |
| Ctrl+O | expand tool output |

Commands: `/model`, `/reload`, `/new`, `/sessions`, `/focus <n>`, `/quit`. Plugins
add `/resume` and `/eval`.

## Shape

- **Core:** the agent loop (`glue.rs`), the tools (`tools.rs`, `prompt.rs`), the TUI
  (`tui.rs`) and reload (`reload.rs`, `supervisor.rs`). Session persistence
  (`session.rs`) and the BRP server (`brp.rs`) support them.
- **Feature plugins:** ordinary Bevy plugins in `src/plugins`. `--disable` turns each one
  off:
  - `greet` (the example native plugin)
  - `brp-tools`
  - `remote`
  - `mcp`
  - `durable`
  - `telemetry`
  - `cassette`
  - `evals`
  - `codemode`

  With all of them off, the agent still runs with `read`, `bash`, `edit`, `write` and
  `reload`.
- **Processes:** `rig-pi` is a supervisor that runs `rig-pi --child`. Both run from copies
  in `<state>/bin`, so `cargo build` can overwrite `target/` safely.

## The Bevy ↔ Rig glue (`src/glue.rs`, the seed for the rig-ecs rewrite)

One module, plain Bevy building blocks over plain Rig values, and nothing in it knows
about the TUI, reload, BRP or any plugin.

**Agents are entities.** An agent has these components:

- `Agent { preamble }`
- `Model(ProviderRef)`: change it at any time; the next request uses it.
- `Conversation(Vec<Message>)`
- `Inbox { follow_ups, steering }`
- `Turn`: `Idle`, `Ready`, `Thinking(Task)` or `Acting`.

These are optional:

- `Session(ConversationId)`: persist the conversation.
- `Endpoints`: send a vendor's requests elsewhere, with another key.
- `TurnSpan`

The TUI session, remote sessions and eval runs are all such entities in one world. No
agent state lives in a resource.

**Tool calls are entities.** A reply's tool calls become one `Call { call, index }` entity
each. They are tied to the agent by the `CallOf`/`Calls` relationship, whose
`linked_spawn` means despawning the agent despawns its calls. A call is marked `Ready`
when it is next; calls run one at a time, in order. It then gets `Running(Task)` and
finally `Done(Result<ToolOutput, String>)`.

**Tools** are one resource, `Tools`, holding two kinds of tool:

- `Tool::Task(DynamicTool)`: a Rig tool the glue runs on the async compute pool.
- `Tool::World(ToolDefinition)`: some other system runs it. That system finds `Ready`
  calls with its tool's name and inserts `Done`. `reload` and BRP tools work this way,
  because they need the world.

This split is the one real design decision in the module. Some tools are async functions;
others are the ECS itself.

**Systems.** One chain runs in `AgentSet::Run`:

1. `interrupt`
2. `start_turns`: the inbox becomes a user message.
3. `send_requests`: `DynModel::call` on the IO pool, instrumented with the turn span.
4. `receive_replies`: spawns `Call`s.
5. `schedule_calls`: marks the next call `Ready`.
6. `run_task_calls`
7. `poll_task_calls`
8. `report_calls`
9. `finish_calls`: the tool results, plus any steering, become the next user message.

The sets `AgentSet::Input` and `AgentSet::Route` run before the chain, so a model change
and its routing land before the next request is built.

**Output** is `AgentEvent` messages: `Input`, `Text`, `CallStarted`, `CallFinished`,
`TurnEnded` and `Error`. The transcript view, remote watch streams, eval checks and
recording all read them. **Input** is the `Inbox` component and the `Interrupt` message.

**Persistence.** With a `SessionMemory(Arc<dyn ConversationMemory>)` resource, each new
message is appended to Rig's memory. Each request is built from what the memory loads,
so a compacting memory shapes requests without the glue knowing. Model routing uses
`ProviderRef`: the registry resolves it, and `ProviderConfig::with_base_url` redirects
it for `Endpoints`.

**For rig-ecs:**

- Keep agent state in components and tool calls as related entities.
- Keep the `Task`/`World` tool split.
- Report progress as messages and take input through components.
- Let `ConversationMemory` be the persistence and compaction seam.

What rig-ecs would add is effect-level recording. Each request, reply and tool outcome
would become a typed effect with a stable identity (agent id, turn number, call index),
written to a log before it is applied. Replay would feed recorded outcomes back in place
of the IO pool tasks and tool runs. The `Running`/`Done` markers are already where such a
log would hook in. HTTP cassettes (below) replay the provider but still execute tools.
Effect logs would also replay tool outcomes, and so make replays independent of the file
system.

## Reload

1. `reload` (the tool) or `/reload` runs `cargo build --bin rig-pi` in the agent's crate
   while the agent keeps running.
2. If the build fails, the error diagnostics (not warnings) become the tool result.
3. If it succeeds, the binary is copied to `<state>/bin`. Once no task tool is running
   anywhere, every agent is saved, the path is written to `next-binary`, and the agent
   exits with code 75.
4. The supervisor starts the new binary with `RIG_PI_RESTARTED` and a note. Every saved
   agent comes back. An unfinished tool batch whose next call is `reload` gets the note
   as that call's result, and the turn continues, so the model can use the change in the
   same turn. Queued prompts, the transcript, the model choice and the BRP port carry
   over.
5. If the new binary exits before it writes its ready marker (10 frames in, after every
   plugin's `build`), or is not up within 60 s, the supervisor starts the last binary
   that did come up. The model is told why, with the tail of stderr.
6. Once a new binary is up, the supervisor `exec`s into it (`--adopt <pid>`, same PID,
   the child stays its child). That way a reload also updates the supervisor.

## Sessions and durability

Conversations go to rig-memory's new `FileConversationMemory`: one append-only JSON Lines
file per session. What only this agent needs goes to `<state>/agents/<id>.json`:

- kind and name
- the model
- the transcript view
- queued prompts
- an unfinished tool batch

Registered BRP tools go to `brp-tools.json`, and the BRP port to `port`.

The core restores from these after a reload. The `durable` plugin also restores after a
quit (`--new` opts out) and adds `/resume`. It wraps the store in rig-memory's
`CompactingMemory(TokenWindowMemory, TemplateCompactor)`, so requests keep a 120k-token
window and older turns fold into a summary, while the file keeps everything.

A history restored after a crash or quit is checked with
`rig_core::transcript::validate_canonical`. Calls that were interrupted are answered with
`tool_result_message`. Reload uses the same mechanism.

## What I took from pi (earendil-works/pi @ d4d74eb19be92c559f629a7f9707c5503a840edc)

- **Tools** (`packages/coding-agent/src/core/tools`): `read`/`bash`/`edit`/`write` with
  pi's parameter names and descriptions and its 2000-line/50 KB limits. `edit` takes
  `edits[{oldText,newText}]`, each matched once against the original file and not
  overlapping; line endings and BOM are preserved.
- **System prompt** (`system-prompt.ts`): a preamble, then tagged `<tools>` with
  one-line snippets and `<rules>` gathered from each tool's guidelines, plus sections
  plugins add. Unlike pi, it has no cwd or date. Requests must be byte-stable for replay,
  so the self section uses a relative path.
- **Loop** (`packages/agent`): steering input joins the running turn at its next request;
  follow-ups wait for the turn to end. Events are like pi's
  `message_*`/`tool_execution_*`/`turn_end`. Tool execution is sequential (pi's opt-in
  mode), because a resumable batch needs a defined order.
- **TUI** (`packages/tui`, keybindings, slash commands): the keys above, and `/model`,
  `/new`, `/resume`, `/reload`, `/quit`.
- **Switching models** (`packages/ai`, "cross-provider handoffs"): pi turns another
  provider's thinking into `<thinking>` text. Rig seals reasoning with its issuer and
  replays it only to the provider that made it. I kept Rig's approach; switching in both
  directions across all four providers works.

## How each pi feature maps onto rig

- **Remote sessions** (`server`/`protocol`/`client`) → the `remote` plugin serves BRP
  methods `rig_pi.session.{create,list,prompt,model,get,interrupt,close}` and the stream
  `rig_pi.session.watch+watch`, instead of a second protocol. A session is an agent
  entity. `examples/remote_client.py` drives one.
- **MCP** (`mcp`) → `rig-rmcp`'s `McpTool` → `DynamicTool`, named
  `mcp__<server>__<tool>`, with calls spawned on a shared tokio runtime.
  `examples/mcp_server.rs` is a stdio server to try it with.
- **Durable sessions** (`durable`, `session-backends`) → `ConversationMemory` with the
  new `FileConversationMemory`, `CompactingMemory` for compaction, and `rig_core::transcript`
  to validate and repair restored histories.
- **Telemetry** (`telemetry`) → `rig_core::telemetry`'s GenAI spans. The glue opens an
  `invoke_agent` span per turn and an `execute_tool` span per call; Rig's `chat` spans
  nest inside. They are exported over OTLP/HTTP JSON through Bevy's
  `LogPlugin::custom_layer`. `examples/otlp_collector.py` receives them.
- **Evals** (`evals`) → `rig-cassette`'s HTTP engine. A script is steps of model
  switches, and prompts with the tool calls and answer they must produce. It replays
  against its cassettes as its own agent entity, from `rig-pi eval`, `/eval` or
  `--replay`.
- **Code mode** (`codemode`) → the new `rig-codemode` crate: QuickJS, where the only
  capability is calling `DynamicTool`s. The plugin offers every task tool (MCP included)
  as `tools.<name>()`.

## Cassettes

`--record NAME` routes the TUI session through a `ProviderCassette` recording proxy for
each vendor it uses (rig-cassette's `http` engine, `start_named`). On exit the proxy
writes two things:

- `fixtures/cassettes/<vendor>/NAME.yaml`, scrubbed by the engine;
- `evals/NAME.json`: the prompts, model switches, tool calls and answers.

`--replay NAME` and `rig-pi eval` point each vendor at a local replay server with a dummy
key; a vendor with no recording is pointed at a closed port, never at the network.

Requests are deterministic: no paths, times or random values in the prompt, tools in
name order, and `store: false` on OpenAI Responses. The engine refuses recordings that
leave stored responses behind.

The fixtures are:

- `greet-{openai,anthropic,gemini,deepseek}`: a tool call per provider.
- `switching-session`: a live TUI session through all four providers, with `read`,
  `greet`, `codemode` and `bash`.

`tests/replay.rs` runs them all with the API keys removed. `.github/workflows/agent.yaml`
runs it in CI. It also reproduces the one replay bug fixed so far: a preamble built
before code mode was registered.

## Rig changes (each its own commit, tested and documented)

- `GEMINI_3_8_FLASH` and `DEEPSEEK_V4_1_FLASH` constants. The IDs come from the model-list
  APIs; DeepSeek calls V4.1 Flash `deepseek-flash`.
- `ProviderId::resolve` accepts a bare family name that has one vendor. Before this,
  `gemini:…` failed because the vendor is `gcp.gemini`.
- `ProviderConfig::base_url`/`with_base_url`, to route any registry preset through a
  proxy or replay server.
- rig-memory `FileConversationMemory` (native-only `file` feature): append-only JSON
  Lines, synced on every append, torn lines skipped, IDs percent-encoded, and
  `conversations()` to list them.
- `rig-codemode`, a new crate and the facade's `codemode` feature. `CodeMode::run` and
  `CodeMode::tool` run scripts with a deadline, a counting allocator as the heap cap
  (which also tells out-of-memory apart from `throw null`), and nested-call records.
- rig-cassette `ProviderCassette::start_named`, for provider and scenario names an
  application only knows at run time.

## Tried and abandoned

- **A single global session resource** (the first version) → agents and calls as
  entities, as the brief requires.
- **A supervisor that never changes** → the `exec`/`--adopt` handover. Before it, edits
  to the supervisor needed a manual restart.
- **The agent `exec`ing itself** → a supervisor process, so there is something left to
  fall back to.
- **`bevy`'s `bevy_remote` feature**: it forces `bevy_render` on → `bevy_remote` directly
  with only `http`.
- **Bevy 0.20.0-rc.2 on the workspace toolchain**: it needs rustc 1.96, and `wesl`
  needs 1.97.1 → the agent is its own workspace with `rust-toolchain.toml` 1.98.1.
- **A prompt refresh in `PreUpdate`**: it raced code mode's tool registration and broke
  replay → it now runs in `Update`, in the route set.
- **Missing OpenAI reasoning items in history**: investigated, not a bug. GPT-6.1 Sol
  sends none when it does not reason.

## Known limitations

- Unix only (`exec`, `waitpid`).
- Streaming is not used: replies arrive whole, with a spinner meanwhile.
- A reload waits for running task tools. A model request in flight is sent again after
  the restart. A recording ends when its process does, so after a reload the session
  runs live.
- Cassettes replay the provider but still run the tools. Replayed evals therefore use
  deterministic tools and checked-in fixtures.
- Recording captures follow-ups, not mid-turn steering.
- Code mode can call task tools, including MCP tools, but not world tools (`reload`, BRP
  tools).
- BRP has no authentication; any local process can drive the agent.
- `bevy_remote` fails silently if it cannot bind, so the agent checks the port first.
  Upstream, `process_remote_requests` `return`s on an unknown method.
- Compaction summaries use `TemplateCompactor`, not a model. They are recomputed after a
  restart.
- The transcript wraps by character. The `bash` timeout kills `bash` but not its
  grandchildren.
