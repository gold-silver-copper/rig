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

## Rig changes

No Rig library changes were necessary and no Rig bugs were found. The raw
completion and message APIs already support the loop. Root manifest only
excludes the standalone agent workspace; root ignore only isolates local
source clones/logs. These are agent integration changes, not Rig runtime
changes. Any subsequent Rig fixes would be separate commits.

## Verification and limitations

Implementation compiles with the exact Bevy prerelease and Rig path crates.
Final unit/lint/live acceptance results will be recorded after execution.
See the agent README for usage, test commands, BRP contract and limitations.
Debug-only, fixed native ABI, one in-memory conversation, no streaming,
approval/cancellation UI, persistence, or compaction. Native code, shell and
BRP peers have full user privileges; loopback is not authentication. Retained
patch libraries consume memory. Shell timeout kills its child, not arbitrary
descendants. The ephemeral port probe has a small bind race. This patches
native plugin logic through a Bevy bridge, not arbitrary Rust system types
or host code. No claims of schema/state migration or release-build patching.
