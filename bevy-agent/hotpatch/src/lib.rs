//! Subsecond hot-patching hosted by the program being patched, with no
//! Dioxus CLI (`dx`) involved.
//!
//! `dx serve --hotpatch` does three jobs: it builds a "fat" binary that keeps
//! every dependency symbol, it recompiles the tip crate into a "thin" patch
//! library whose missing symbols are stubbed with addresses from the running
//! process, and it ships a [`subsecond::JumpTable`] to that process. This
//! crate ports those jobs from dioxus-cli 0.7.10 (`build/link.rs`,
//! `build/patch.rs`, `rustcwrapper.rs`, `cli/link.rs`) so a program can do
//! them for itself: [`build_fat`] and [`relaunch`] at startup, then
//! [`Patcher::build`] and [`subsecond::apply_patch`] whenever its source
//! changes. The same executable doubles as the rustc wrapper and linker shim
//! the builds need, which is why [`intercept`] must run first in `main`.
//!
//! Only native macOS and Linux targets on x86_64 and aarch64 are supported,
//! and only the tip (binary) crate is patched.

mod capture;
mod fat;
mod stub;
mod thin;

pub use capture::{RustcInvocation, intercept};
pub use fat::{FatBuild, FatOptions, build_fat, relaunch};
pub use thin::{Patch, Patcher};
