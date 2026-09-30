# Sprout design

Standalone workspace: `tools/sprout`. Fork branch `agent/bevy-sprout-429f9f`,
[PR #3](https://github.com/gold-silver-copper/rig/pull/3), based on fork/main
`654567eb6`. No Rig library changes were needed: `rig-core` already provides the
completion request, response, tool definitions and correctly paired tool results.
Neither `rig-agent` nor `rig-ecs` is a dependency.

## Decisions

- Bevy owns native plugin state, BRP handlers, model event consumption and reload
  coordination. Ratatui drives `App::update` at terminal cadence. A worker owns
  the bounded Rig model/tool loop, so HTTP, shell and extension waits do not freeze
  Bevy or the terminal. One shell tool is enough to read, edit and test code.
- Native plugins are real `Plugin` implementations with scheduled systems. A tiny
  cdylib exports the exact `SystemParamFunction::run` trampoline address Bevy uses.
  The host builds with Cargo, discovers the new address, and installs a subsecond
  jump table at a frame boundary. Bevy's stock `HotPatchPlugin` refreshes cached
  system pointers. State stays in the host. No dx executable, ThinLink, websocket
  patch server or compiler wrapper is needed. Fingerprints guard compiler/ABI
  source/manifest/lockfile changes; only behavior bodies may hotpatch.
- Higher-level plugins are independent BRP clients: register JSON Schema tools,
  poll invocations, return results. No HTTP callback server or cross-language ABI.
  The sample Python tool is actually executed by a separate process.

## Research and alternatives

Crates.io's index identified **0.20.0-rc.2** as the latest Bevy, including
prereleases. Before building, checked out exact tag commit
`343a0233d6788a49f224a3684c8995e246663df3` into this worktree's ignored research
area. Read all four remote examples, `app/headless.rs`, `app/plugin.rs`, and
`ecs/hotpatching_systems.rs`, plus the relevant app/ECS/BRP implementations and
subsecond 0.7.10's patch API.

Abandoned whole-program dx/ThinLink embedding: unnecessary complexity for one
explicit plugin boundary. Did not substitute mere library pointer swapping:
patches go through Bevy's own hotpatch trampoline. The first attempt inherited
Rig's Rust 1.95, below this Bevy's MSRV. Reused the already installed 1.98.1 with a
local toolchain file; no global installs or upgrades. Also avoided Bevy facade's
`bevy_remote` feature, which enables rendering; use the exact-version BRP crate's
HTTP-only feature directly. BRP's current `bevy_dev_tools` dependency still brings
substantial UI/audio/shader dependencies even in a headless app.

## Limits

Trusted local development tool, not a sandbox. OpenAI only, nonstreaming replies,
no persistence, scrolling, approvals, schema validation engine or plugin unloading.
Native layout/registration changes require restarting; old code stays loaded.
Shell descendants may outlive a shell timeout. BRP is loopback-only but unauthenticated,
and free-port selection has a short bind handoff race. Linux is intended but not
claimed tested; live verification is on macOS arm64. See the agent README for
commands and protocol contracts.

## Verification

The real-model PTY test passed with `gpt-4.1-mini` and no dx on PATH. Agent PID
**38769** stayed constant: patch #1 changed the native label to `flourishing` at
529 ticks, then patch #2 restored `seedling` at 844 ticks. The model itself edited
the source via its shell tool. Invalid Rust was rejected without losing the old
plugin. External Python PID **38947** registered and executed `uppercase` over BRP
on selected port **49263**, returning `EXTERNAL_EXTENSION_OK` to the model/TUI.

The terminal decoder asserted these real rendered lines, not just internal logs:

```text
tool shell: {"command":"printf SPROUT_TOOL_OK"}
assistant: SPROUT_TOOL_OK
native patched #1 (same process)
tool uppercase: {"text":"external_extension_ok"}
assistant: EXTERNAL_EXTENSION_OK
```

Machine-readable evidence: `tools/sprout/evidence/live-report.json`. Original PTY
capture and decoded screens remain in ignored `tools/sprout/.sprout/`. Five focused
unit tests passed (native state, TUI rendering, shell bounds/errors, BRP lifecycle
and reserved names); workspace Clippy with `-D warnings`, rustfmt and whitespace
checks also passed. An agent-specific macOS CI workflow covers the standalone
workspace, which Rig's root CI does not select. Live API/PTY evidence is local,
not a secret-dependent CI job. Independent model review found no confirmed correctness issues.
Test compilation caught Bevy errors lacking `std::error::Error` and the cdylib's
`main` colliding with the test harness; both were fixed without runtime changes.
