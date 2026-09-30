# Sprout

A small trusted-local coding agent: Rig's model API, Bevy 0.20.0-rc.2, ratatui,
native Bevy system hotpatching, and BRP tools. No `rig-agent`, `rig-ecs`, or `dx`.

## Run

From this directory, with a Rust toolchain and your exported `OPENAI_API_KEY`:

```sh
cargo build --workspace --locked
cargo run --locked
```

The local `rust-toolchain.toml` pins Rust 1.98.1. `SPROUT_MODEL` defaults to
`gpt-4.1-mini`. Enter submits a prompt; Esc or Ctrl-C exits. The model can use
`shell` to read/edit files and run commands in the launch directory. It has
12 tool turns per prompt, a 90-second model deadline and a 60-second shell
deadline. stdout/stderr each retain at most 16 KiB. The TUI displays the prompt,
tool calls/results and model replies. History persists for the current session.

**This is not a sandbox.** Only run it in a trusted checkout. Model-directed shell
commands have your user permissions and environment. Native code has full process
access. Review generated edits before relying on them. No approval UI or durable
sessions are implemented. Exiting does not guarantee cancellation of detached
subprocesses; shell deadlines kill the direct child, not arbitrary descendants.

## Native hotpatching, without dx

Edit `native/src/behavior.rs` while the TUI runs, or ask the agent to do so:

> Use shell to replace `native: seedling` with `native: flourishing` in
> native/src/behavior.rs. Do not restart the agent.

The native plugin's Bevy system keeps updating its existing `NativeState` resource.
The host watches the source, builds the workspace off-thread using Cargo, copies
the patch cdylib to a unique filename, discovers exported addresses, and supplies
a single-entry `subsecond::JumpTable`. Bevy's `HotPatchPlugin` refreshes its own
scheduled system trampoline. The TUI label changes, but PID, conversation and tick
counter survive. Build errors retain the previous code; fix the source to retry.
See `.sprout/native-build.log` for compiler diagnostics.

Keep function signatures, system parameters, resources, registration, manifests
and lockfile unchanged. Only the behavior module is watched. The compiler and
fixed ABI source fingerprint must match; structural changes require a restart.
This is deliberately a narrow native-plugin hotpatch boundary, not a general
whole-program linker. Debug builds only. Loaded generations stay resident, as
required by subsecond. Long editing sessions eventually need a restart to reclaim
code memory. Do not run two instances against the same source/artifact directory.

## External BRP plugins

The app chooses a free loopback port and writes its URL/PID to
`.sprout/session.json`. It never uses Bevy's fixed default port. In another shell:

```sh
python3 examples/uppercase.py
```

Then ask: `Use uppercase with text hello from BRP, then report its result.`
The separate Python process implements the actual tool. Custom BRP JSON-RPC 2.0
methods are real Bevy systems registered through `RemotePlugin`:

| Method | Params | Result |
| --- | --- | --- |
| `sprout.register` | `name`, `description`, `parameters` (JSON Schema) | registration acknowledgement |
| `sprout.poll` | `name` | pending `{id, name, arguments}` calls |
| `sprout.result` | `id`, `text` | completion acknowledgement |
| `sprout.status` | none | PID, busy state, transcript, native label/ticks/generation |

Tool definitions are snapshotted at prompt submission. Results expire after 30
seconds. Polling is at-least-once until result submission; only the first result
is accepted. One worker per tool name, maximum 32 registrations. No unregister,
authentication or sandboxing: BRP is for trusted local processes only, including
Bevy's built-in reflection methods. Do not expose or forward its port. The short
bind-to-port handoff has an OS port-allocation race; check the published endpoint
before using it. The example uses only Python's standard library.

## Verify

```sh
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo fmt --all -- --check
# Optional: small real-model test, real pseudo-terminal, actual source edits.
# Terminal decoder is test-only and installed inside this checkout.
python3 -m pip install --target .sprout/python pyte
python3 tests/live.py
```

The dedicated `Sprout` CI workflow runs the key-free checks on macOS. The real-model
acceptance test is local and opt-in, not a CI job requiring provider secrets.

The live test requires `dx` absent from PATH, opens its own ephemeral BRP port,
asks a real model to call a tool and patch its plugin, verifies unchanged PID and
increasing ticks, calls an external BRP tool, tests broken-source rollback and a
second successful patch, and restores the original source. Logs and decoded TUI
screens go under ignored `.sprout/`; credentials are read only from the environment.
