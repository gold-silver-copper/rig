# Minimal Bevy coding agent

Worktree `rig-bevy-agent-k7`, branch `agent-bevy-20260608-k7`, fork PR #2.
Latest fork/main at creation: `654567eb6`. crates.io reports Bevy
`0.20.0-rc.2` as newest, including prereleases. Exact Bevy source tag checked
out locally at `343a0233d6788a49f224a3684c8995e246663df3` before building.
Read the headless, hotpatching_systems, and all four remote examples, plus
Bevy's HotPatchPlugin and Subsecond's loader/HotFunction implementation.

## Design and decisions

`examples/bevy_agent` is an independent workspace so Bevy's prerelease and
Rust >=1.96 do not change Rig's existing dependency graph/MSRV. An installed
1.98.1 toolchain builds it; nothing is globally installed or changed.
A Bevy app owns the UI, native patch state, and BRP methods. A Tokio worker
calls Rig's raw OpenAI chat completion model and executes a bounded tool
loop. No rig-agent or rig-ecs. ratatui runs on the main thread; worker events
and tools cross channels. Read/write/shell tools permit real coding.

NativePlugin implements Bevy Plugin and supplies an ECS bridge to one
reloadable UTF-8-buffer C ABI entry point. The running agent compiles source
snapshots to unique cdylibs using rustc, checks an explicit ABI version, and
installs a **real Subsecond jump table**, with runtime addresses eliminating
ASLR arithmetic. Bevy hotpatching is enabled; successful swaps emit
HotPatched and change HotPatchChanges. We replace the stock dx-connecting
HotPatchPlugin. This deliberately bounds unsafe code: no Rust World,
allocations, trait objects, or state layouts cross the plugin boundary.
Both automatic source detection and a model-callable patch_native tool
apply code without restarting the app. Invalid builds/ABI versions retain
the previous code. Libraries remain loaded for pointer safety.

External processes register JSON-schema tools through custom BRP methods;
the model calls their loopback HTTP callbacks and receives actual results.
Tools refresh every model round. Standard BRP methods remain available.
Ports are selected fresh, never a BRP default. This is a trusted local
extension mechanism, not a security boundary.

## Tried and abandoned

Bevy's published hotpatch example requires dx. Porting dx's whole-program
linker/symbol/state machinery would swamp a minimal agent, so use explicit
native exports and Subsecond's public loader instead. Bevy's top-level
bevy_remote feature pulled a GPU/render graph even without default features;
using bevy_remote directly with only http avoids that graph, although its
current dev-tools dependency still brings many non-render crates.
No dynamic Rust Plugin/World ABI: compiler/layout changes are too unsafe.
No UI framework, async Bevy runtime adapter, or generalized tool framework.
The first isolated-PATH trial omitted macOS linker/SDK tools; include only
rustc, sh and compiler tools, still no dx. A live model initially mistook
behavior v2 for ABI v2; explicit ABI=1 instructions and actionable rejection
messages fixed this without weakening ABI checks.

## Rig changes

No Rig library changes were necessary and no Rig bugs were found. The raw
completion and message APIs already support the loop. Root manifest only
excludes the standalone agent workspace; root ignore only isolates local
source clones/logs. These are agent integration changes, not Rig runtime
changes. Any subsequent Rig fixes would be separate commits.

## Verification and limitations

Implementation compiles with the exact Bevy prerelease and Rig path crates.
Four focused tests passed; the native regression passed again after improving
request error delivery. Final clippy --all-targets -D warnings, rustfmt,
Python syntax and git diff --check passed. Independent full-source model
review found no concrete correctness bug; generic unsafe warnings from an
earlier review were checked against the fixed ABI and main-thread boundary.
A separate scoped macOS CI workflow covers the otherwise-excluded workspace;
hosted CI is pending, not a local acceptance claim.

Live PTY acceptance passed with gpt-4o-mini, PID **63981**, BRP port **53231**:
- read_file(fixture.txt) returned LIVE_SENTINEL_K7 and the TUI showed the answer.
- The model read/wrote native.rs, automatically installed native-live-v2,
  then called patch_native and native(status). Generation 1 -> 3; same PID.
- An external Python process registered external_word_count over BRP; the
  model called it, received {words:3, plugin:external-process}, and answered.
- dx was unavailable in the process's private PATH throughout. No global
  tools were removed/installed; rustc compiled the actual cdylib.

Raw evidence and actual rendered frames are in
`examples/bevy_agent/verification-artifacts/1790793091319267000/` (ignored):
`evidence.json`, `terminal.ansi`, and `01/02/03-*.txt`. The third frame shows
both the completed native answer and the external extension answer. The
verification script reproduces this with fresh ports and API keys
read only from the environment. See the agent README for the full contract.
Debug-only, fixed native ABI, one in-memory conversation, no streaming,
approval/cancellation UI, persistence, or compaction. Native code, shell and
BRP peers have full user privileges; loopback is not authentication. Retained
patch libraries consume memory. Shell timeout kills its child, not arbitrary
descendants. The ephemeral port probe has a small bind race. This patches
native plugin logic through a Bevy bridge, not arbitrary Rust system types
or host code. No claims of schema/state migration or release-build patching.
