# rigpi: design notes

rigpi is a minimal, pi-style coding agent. It uses Rig (`rig-core` only) for
models, Bevy 0.20.0-rc.2 as its runtime, and ratatui for the TUI. It
hot-patches its own native plugins while it runs, without the Dioxus CLI, and
it accepts out-of-process plugins over the Bevy Remote Protocol.

```sh
cd bevy-agent
cargo run -- [--model provider:model] [--brp-port PORT]   # default: anthropic:claude-haiku-4-5
python3 examples/brp_plugin.py PORT                       # an external plugin (adds `py_eval`)
```

Keys: Enter sends, Shift/Alt-Enter or Ctrl-J adds a newline, Esc stops the
turn, PgUp/PgDn/arrows scroll, Ctrl-C quits. Commands: `/reload`, `/clear`,
`/quit`.

## Shape

The agent is one Bevy `App`. It runs `MinimalPlugins` with a 30 Hz
`ScheduleRunnerPlugin` and has no window or renderer.

- **Agent loop** (`src/agent.rs`). An `Agent` resource holds a
  `DynModel<Completion>` built from Rig's `ProviderRef` registry, the message
  history, and a `Turn` state machine (`Idle` → `Thinking` → `Tools` → ...).
  Replies stream on a small tokio runtime and come back over a channel. Each
  tool call becomes a `ToolCall` entity. When every call has a `ToolOutput`,
  the results go back to the model. Anthropic requests set top-level
  `cache_control`, so the growing history is cached.
- **Tools as ECS data** (`src/tools.rs`). A tool is an entity with a
  `ToolSpec`. A call is answered by whoever inserts `ToolOutput`: a native
  function, the `reload` plugin, or an external process. The components are
  reflected, so plain BRP can see and edit them too.
- **Native plugins** (`src/plugins/`). Tools (`read`, `write`, `edit`, `bash`)
  are plain `fn(Value) -> Result<String, String>` values listed in
  `native_tools()`. Plugins that need the ECS (`reload`, the status line) are
  ordinary Bevy plugins. Every Bevy system and observer is hot-patchable
  (`bevy_ecs/hotpatching`). The tool registry is rebuilt from
  `native_tools()` on each `HotPatched` message, so a patch can change,
  add, or remove tools.
- **TUI** (`src/tui.rs`). crossterm input runs in `PreUpdate` and ratatui
  drawing in `PostUpdate`. Both are Bevy systems, so the UI hot-patches too.
- **BRP plugins** (`src/remote.rs`). `RemotePlugin` + `RemoteHttpPlugin` on
  127.0.0.1, with a free port by default. The `rigpi/*` methods add a small
  plugin API on top of the stock methods: `register_tool`, `unregister_tool`,
  `take_calls`, `complete_call`, `prompt`, `cancel`, `transcript`, `state`.
  `examples/brp_plugin.py` (standard library only) registers `py_eval` and
  serves it.

## Hot-patching without `dx` (`hotpatch/`, `src/hot.rs`)

Bevy's hot reloading is subsecond: systems call through a jump table that
`subsecond::apply_patch` swaps. Normally `dx serve` builds the patches.
`rigpi-hotpatch` ports what `dx` 0.7.10 does (`build/link.rs`,
`build/patch.rs`, `rustcwrapper.rs`, `cli/link.rs`), so the agent can do it
for itself:

1. **Fat build at startup.** When the process isn't already a fat build, it
   bumps `src/main.rs` and runs `cargo rustc -- -Csave-temps=true
   -Clink-dead-code -Clinker=<shim>` with `RUSTC_WORKSPACE_WRAPPER=<shim>`.
   The shim is a copy of the agent binary; `intercept()` at the top of
   `main` turns it into a rustc wrapper (records the tip crate's exact rustc
   invocation) or a linker (records the link line and writes an empty
   object). The agent then links the executable itself: every `.rcgu.o` of
   every dependency rlib goes into one force-loaded archive, and `_main` is
   exported. Finally it `exec`s into that fat binary.
2. **Thin build per patch.** A file watcher (debounced) or the `reload` tool
   replays the recorded rustc invocation (incremental, about 0.4–4 s). It
   collects the fresh tip objects, writes a stub object that defines every
   symbol they need from the running process (jumps for functions,
   absolute symbols for data, fresh TLS), and links a dylib. It then maps
   old function addresses to new ones by symbol name.
3. **Apply.** An exclusive system calls `subsecond::apply_patch`. Our
   subsecond handler bumps `HotPatchChanges` and emits `HotPatched`, and
   Bevy re-resolves every system on its next run. Compile errors come back
   as rendered diagnostics, and the running code stays unchanged.

`reload` waits for a build that started after its request, so it never
reports a stale result.

## Decisions

- **Separate workspace.** `bevy-agent/` is its own Cargo workspace (with
  its own toolchain pin: Bevy 0.20 needs Rust ≥ 1.96). This keeps Bevy out
  of Rig's lockfile and CI. It depends on `crates/rig-core` by path.
- **Bevy sub-crates.** The agent uses `bevy` (no default features),
  `bevy_ecs` with `hotpatching`, and `bevy_remote` with only `http`.
  Enabling `bevy/hotpatching` would pull in `dioxus-devtools`, and
  `bevy/bevy_remote` would pull in the render stack.
- **Own plugin instead of `HotPatchPlugin`.** `HotReloadPlugin` does the
  same bookkeeping as Bevy's `HotPatchPlugin` (`HotPatched`,
  `HotPatchChanges`) without connecting to a `dx` devserver.
- **Default model.** `anthropic:claude-haiku-4-5` when `ANTHROPIC_API_KEY`
  is set, else `openai:gpt-5-mini`. Any `ProviderRef` works.

## Tried and abandoned

- **Bevy's `HotPatchPlugin` / `bevy/hotpatching`.** It expects `dx`'s
  websocket and drags in `dioxus-devtools`. Replaced as described above.
- **One generic Bevy system per native tool** (`add_native_tool::<T>()`).
  `Plugin::build` never reruns, so a patch could change a tool but never add
  one. Switched to the function-pointer registry that is re-read on
  `HotPatched`.
- **Watcher without debounce, and `reload` waiting on "the next build".**
  Multi-edit changes produced failed intermediate builds, and `reload` could
  report a build that predated the model's last edit. Now changes are
  debounced, and each request returns the build generation that covers it.
- **Considered, not built:**
  - `-Wl,-all_load` for the fat link. `dx` stopped doing this because rlibs
    carry non-rcgu objects that break it (dioxus #4237), so rigpi uses `dx`'s
    archive approach.
  - `cdylib` plugins reloaded with `libloading`. That isn't Bevy's hot
    reloading, and it has type-identity and ABI problems across dylibs.
- **Hand-numbered `read` output** (`NNNNN␠␠line`). Haiku copied the
  separator into an `edit` `old_text`. Changed to the familiar
  `number<TAB>line`.

## Rig changes

- **`fix(registry)`: unknown provider selections now say what is registered.**
  `ProviderRef::parse("gemini:…")`, `OpenAI:…`, and `claude:…` used to fail
  with only "no registered provider is named …", so a CLI like this one had
  nothing to suggest. `SelectionError::Unknown` now carries `candidates`,
  chosen in this order: a case-insensitive match, else a dotted-segment match
  (`gemini` → `gcp.gemini`), else every registered vendor. The message says
  "did you mean `gcp.gemini`?" or lists the vendors. It is a separate commit
  with a unit test.

I found no other Rig bugs. The agent ran the same scenarios through Rig on
Anthropic, OpenAI (`gpt-5-mini`, `gpt-5.4-mini`), Gemini 2.5/3, DeepSeek, xAI,
Groq, and OpenRouter: parallel tool calls, multi-round turns, follow-ups that
replay tool history, empty tool output, cancellation, and streamed usage with
caching. Mistral only returned rate-limit errors. Tool errors travel as
`Error: …` text, because Rig's `ToolResult` has no error flag. Anthropic's
`is_error` is therefore never set; the change was too invasive for the gain.

## Known limitations

- **Platforms.** Hot-patching supports macOS and Linux on x86_64/aarch64.
  Only macOS arm64 was exercised; the Linux branches are ported from `dx` but
  untested. Hot-patching also needs a debug build (subsecond is inert
  without `debug_assertions`) plus the source and a toolchain at the
  compile-time `CARGO_MANIFEST_DIR`.
- **What a patch can change.** Only the `rigpi` crate is patched; edits to
  `rigpi-hotpatch` or rig-core need a restart. A patch must keep the layout
  of types that live across it (resources, components, `Turn`, …) and the
  signatures of existing systems; otherwise it crashes (subsecond's rule).
  New Bevy plugins and systems need a restart, but new tools do not.
  Statics and thread-locals defined in the crate start fresh in each patch.
- **Dependencies.** Adding a dependency needs a restart. Patches can call
  anything in the current dependencies and std, because the fat binary holds
  every object of them (verified with a patch calling `std::os::unix::fs::chroot`).
- **Startup cost.** Each start does a fat build: about 5–10 s, longer on
  the first run, which archives about 110 MB of dependency objects.
- **Cancellation.** Esc abandons a running native tool but does not kill it;
  a `bash` child runs until its timeout.
- **Trust.** BRP is unauthenticated on 127.0.0.1: any local process can
  prompt the agent or edit its world. `bash` has no sandbox or approval
  step. A remote tool call whose plugin died waits until Esc.
