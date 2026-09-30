# bevy-agent

A minimal coding agent in the spirit of pi: Rig (`rig-core`) for the model,
Bevy 0.20.0-rc.2 for the runtime, ratatui for the screen. Its native plugins
are hot-patched in place while it runs, without the Dioxus CLI, and external
processes extend it over the Bevy Remote Protocol (BRP).

## Run

```sh
cd bevy-agent
cargo run -p agent-hotpatch --bin hotbuild -- "$PWD"   # base build, with the hot-patch flags
target/debug/bevy-agent
```

`hotbuild` runs `cargo rustc --bin bevy-agent -- <flags>` with the flags the
patcher needs (see below). A plain `cargo build` also runs, and patches still
apply from it, but its base was dead-stripped, so a patch that references a
symbol the base dropped fails to load and is reported in the transcript.

Environment: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; `AGENT_PROVIDER`
(`anthropic` | `openai`, default: whichever key is set); `AGENT_MODEL` (default
`claude-haiku-4-5` / `gpt-5.6`); `AGENT_BRP_PORT` (default: a free port, shown
in the status bar).

Keys: type and Enter to prompt; `/patch` rebuilds and patches now; `/clear`;
`/quit` or Ctrl-C; PageUp/PageDown scroll.

## Shape

Everything is ECS state in one binary crate, `bevy-agent`, plus a small
library crate, `agent-hotpatch`.

- `agent.rs`: the conversation. `Conversation` holds the Rig `Message`
  history; `InFlight` is the one completion being streamed; each tool call
  the model makes is an entity (`ToolCallRequest`) that ends with `ToolDone`.
  Tools are entities too (`Tool`), whether a native plugin or a remote process
  registered them. Four chained systems drive the loop: start a completion,
  poll it, reject unknown tools, collect results and go again (up to 24 rounds).
- `model.rs`: Rig. Picks a provider client from the environment, erases the
  model to `DynModel<Completion>`, and streams one `CompletionRequest`,
  forwarding text fragments to the TUI through a channel. The future runs on
  Bevy's `AsyncComputeTaskPool`; `rig-reqwest` brings its own runtime, so the
  agent has no tokio dependency.
- `plugins/tools.rs`: the native plugin. Spawns the `Tool` entities
  (`read_file`, `list_dir`, `write_file`, `bash`) and answers their calls in a
  system; `bash` runs on the task pool. Also the system prompt and a status-bar
  hint. This file is what you edit to see a hot patch land.
- `tui.rs`: ratatui. `PreUpdate` reads keys, `Last` draws transcript, prompt
  line and status bar. Wrapping is done by hand to keep the view pinned to the
  bottom without extra crates.
- `remote.rs`: BRP. Adds `RemotePlugin` with custom methods and
  `RemoteHttpPlugin` on the chosen port. Methods:
  - `agent.register_tool {name, description, parameters}`: spawn a remote
    `Tool` (replacing one of the same name).
  - `agent.tool_calls+watch {name?}`: a watching method; streams one
    server-sent event per new call to a remote tool and marks it dispatched.
  - `agent.tool_result {call_id, content}`: answer a dispatched call.
  - `agent.prompt {text}`, `agent.transcript`, `agent.patch`.
  The built-in BRP methods (`world.query` and friends) are there as well.
  `scripts/brp_extension.py PORT` is a dependency-free example: it registers a
  `secret_number` tool and serves its calls.
- `hot.rs`: hot patching. Reads the running binary's symbol table at startup
  (off the main thread), watches `src/` every 500 ms, and on a change (or
  `/patch`, or `agent.patch`) runs the patcher on a thread. Subsecond's
  handler signals a channel; a `First` system then writes Bevy's `HotPatched`
  message and marks `HotPatchChanges`, so the executors refresh every system's
  function pointer that frame.
- `hotpatch/` (`agent-hotpatch`): the patcher, a minimal port of the Dioxus
  CLI's engine (`packages/cli/src/build/patch.rs` and `link.rs`).

## How a patch is made

The base must be built with four extra rustc flags for the tip crate:
`-Csave-temps` (keep the crate's `.rcgu.o` files), `-Clink-dead-code` (no
dead stripping), `-Clink-arg=-Wl,-all_load` (link every object of every
rlib, so a patch may reference any dependency symbol) and
`-Clink-arg=-Wl,-exported_symbol,_main` (`main` is subsecond's ASLR anchor).

On a change:

1. Remove the crate's old objects from `target/debug/deps` and run the same
   `cargo rustc` command. rustc copies every codegen unit it links, cached or
   not, so what is on disk afterwards is exactly the linked set. This also
   relinks the on-disk binary, so the next launch has the new code too.
2. Read the undefined symbols of those objects and write a stub object with
   the `object` crate: for each one defined in the base, a 16-byte
   trampoline (`ldr x16, #8; br x16; .quad addr` on arm64, `jmp [rip]` on
   x86_64) for functions, an absolute symbol for data, and a fresh
   thread-local slot initialised from the base's TLS image.
3. `cc -dylib -Wl,-undefined,dynamic_lookup` the objects and the stub into
   `target/hotpatch/libbevy-agent-patch-<ms>.dylib`. Symbols the stub does not
   cover (libSystem) resolve at load.
4. Build the `JumpTable`: every symbol present in both the base and the
   patch, base address to patch address, with `_main` on both sides as the
   ASLR references. `subsecond::apply_patch` dlopens the dylib and commits it.

Bevy's `FunctionSystem` calls through `subsecond::HotFn`, so after
`HotPatched` every system in this crate runs its patched body, including the
ratatui `render` system and the tool systems. On this machine a patch of the
agent takes about 1.7 s (cargo 1.6 s, link 50 ms) and remaps about 7700
symbols.

## Tried and abandoned

- Linking only the codegen units rustc rewrote. Cached objects are copied with
  their original mtimes, so an mtime filter gave 3 of 116 objects: the changed
  function was in the patch but the unchanged `HotFunction::call_it` wrapper
  that calls it was not, and the old code kept running. Deleting the crate's
  objects before the build and linking everything rustc writes back fixed it.
- `bevy_app`'s own `HotPatchPlugin`. It pulls `dioxus-devtools` and connects
  to a dx websocket for jump tables. The agent enables only
  `bevy_ecs/hotpatching` and talks to `subsecond` directly.
- Writing `HotPatched` from a `Last` system, as `bevy_app` does. Systems that
  ran later in the same schedule had a newer last-run tick than the change and
  never refreshed, so `Update` systems patched but the `Last` render system
  did not. The writer now runs in `First`.
- dx's linker interception (`RUSTC_WORKSPACE_WRAPPER`, a fake linker that
  records arguments and skips the link) to avoid relinking the base binary on
  every patch. The plain cargo relink costs about a second here and needs no
  wrapper binary or captured argument files, so it stayed.
- Passing the base's Mach-O symbol flags through to stub data symbols, as dx
  does on non-Android targets. `object` uses those flags in place of the
  symbol's section, which would turn the absolute symbols back into section
  symbols; the stub passes none on Mach-O and keeps ELF's `st_info`/`st_other`.

## Rig changes

None. `rig-core` 0.43 covered the agent end to end: a provider client from the
environment, `Model::erase` to `DynModel<Completion>`, `Model::stream` with
`StreamEvent::Text` fragments and `Streamed::finish` for the folded response,
`CompletionRequest::from(history)`, `CompletionResponse::message` and
`tool_calls`, `ToolCall::result` and `Message::tool_results`. No bugs surfaced
in the paths used, so there was nothing to fix, and adding API for one caller
would not have made the agent smaller.

## Verified

With `claude-haiku-4-5` on this machine, in the TUI:

- A prompt led to a `read_file` / `list_dir` tool call and a final answer.
- Editing `plugins/tools.rs` while the agent ran (a new status-bar hint and a
  count line in `list_dir`) was picked up within about two seconds: the status
  bar changed and the next tool call returned the new format. `dx` is not
  installed.
- `scripts/brp_extension.py` registered `secret_number` over BRP; the model
  called it, the script answered over BRP, and the TUI showed the answer.

## Known limitations

- macOS is the tested platform (arm64). The Linux path is written (ELF stub,
  `cc -shared`, `--export-dynamic-symbol,main`) but untested, and it has no
  equivalent of `-all_load`, so a patch that references a dependency symbol the
  base never used fails to load there. Windows is not supported.
- Only this crate is patched. Changing a dependency needs a restart.
- The usual subsecond rules: a patch that changes a struct layout used by
  live state will crash; statics and thread-locals defined in this crate are
  duplicated by the patch rather than shared; every patched function is
  considered new.
- The base binary is large (about 330 MB) because every dependency object is
  linked in, and its symbol table takes a moment to read at startup; the
  status bar shows `patch: loading` until then.
- A remote tool call waits forever if its process never answers. Prompts are
  rejected while a turn is in progress. There is no persistence, no
  cancellation and no approval step for `bash` or `write_file`.
- Tool results are shown truncated in the transcript and capped at 16 kB for
  `bash` output.
