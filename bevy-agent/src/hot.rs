//! Self hot-patching. At startup the agent rebuilds itself as a fat binary
//! and relaunches into it; afterwards any change under `src/` (or a request
//! from a tool or command) recompiles the crate into a patch that subsecond
//! swaps in while the app keeps running. Bevy systems and observers pick up
//! the new code on the next run.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use bevy::prelude::*;
use bevy_ecs::{HotPatchChanges, HotPatched};
use crossbeam_channel::{Receiver, Sender};
use rigpi_hotpatch::{FatBuild, FatOptions, Patcher};

/// The agent's own crate, which it hot-patches.
pub const CRATE_DIR: &str = env!("CARGO_MANIFEST_DIR");

/// Relaunch into a fat build unless this process already is one. Returns
/// the fat build when hot-patching is available.
pub fn bootstrap() -> Option<FatBuild> {
    if !cfg!(debug_assertions) || std::env::var_os("RIGPI_NO_HOTPATCH").is_some() {
        return None;
    }
    if let Some(fat) = FatBuild::current() {
        return Some(fat);
    }
    let crate_dir = PathBuf::from(CRATE_DIR);
    let crate_root = crate_dir.join("src/main.rs");
    let exe = std::env::current_exe().ok()?;
    let target_dir = exe.parent()?.parent()?;
    if !crate_root.exists() {
        eprintln!("rigpi: source not found at {CRATE_DIR}; hot-patching disabled");
        return None;
    }
    eprintln!("rigpi: building a hot-patchable binary...");
    let options = FatOptions {
        manifest_dir: crate_dir,
        package: env!("CARGO_PKG_NAME").to_owned(),
        bin: env!("CARGO_BIN_NAME").to_owned(),
        crate_root,
        out_dir: target_dir.join("hotpatch"),
    };
    match rigpi_hotpatch::build_fat(&options) {
        Ok(fat) => eprintln!("rigpi: relaunch failed: {}", rigpi_hotpatch::relaunch(&fat)),
        Err(error) => eprintln!("rigpi: hot-patching disabled: {error:#}"),
    }
    None
}

/// What the last patch did.
#[derive(Debug, Clone, Default)]
pub enum PatchStatus {
    /// Hot-patching is off (release build, or the fat build failed).
    #[default]
    Unavailable,
    /// No patch built yet.
    Idle,
    /// The last patch applied.
    Applied { count: usize, elapsed: Duration },
    /// The last patch failed; the running code is unchanged.
    Failed(String),
}

/// Handle for requesting patches and reading their outcome.
#[derive(Resource)]
pub struct HotReload {
    requests: Option<Sender<()>>,
    results: Receiver<Result<rigpi_hotpatch::Patch, String>>,
    /// The latest outcome.
    pub status: PatchStatus,
    /// A patch is compiling.
    pub building: bool,
    applied: usize,
    /// Bumped whenever a build finishes, applied or not.
    pub generation: u64,
    /// Another build was requested while one was running.
    queued: bool,
    watched: HashMap<PathBuf, SystemTime>,
    last_scan: Option<Instant>,
    changed_at: Option<Instant>,
}

impl HotReload {
    /// Ask for a build of the source as it is now. Returns the generation
    /// whose outcome covers it, or `None` when hot-patching is unavailable.
    pub fn request(&mut self) -> Option<u64> {
        let requests = self.requests.as_ref()?;
        if self.building {
            // The running build may predate the change; build again after it.
            self.queued = true;
            return Some(self.generation + 2);
        }
        self.building = true;
        let _ = requests.send(());
        Some(self.generation + 1)
    }
}

/// Hot-patching for the running app. Replaces Bevy's `HotPatchPlugin`,
/// which needs the Dioxus CLI.
pub struct HotReloadPlugin(pub Option<FatBuild>);

impl Plugin for HotReloadPlugin {
    fn build(&self, app: &mut App) {
        let (result_tx, results) = crossbeam_channel::unbounded();
        let requests = self.0.clone().map(|fat| {
            let (tx, rx) = crossbeam_channel::unbounded::<()>();
            std::thread::spawn(move || {
                let mut patcher = Patcher::new(fat);
                while rx.recv().is_ok() {
                    let result = patcher.build().map_err(|error| format!("{error:#}"));
                    if result_tx.send(result).is_err() {
                        break;
                    }
                }
            });
            tx
        });

        let (patched_tx, patched_rx) = crossbeam_channel::bounded::<()>(1);
        subsecond::register_handler(Arc::new(move || {
            let _ = patched_tx.try_send(());
        }));

        let status = if requests.is_some() {
            PatchStatus::Idle
        } else {
            PatchStatus::Unavailable
        };
        app.insert_resource(HotReload {
            requests,
            results,
            status,
            building: false,
            applied: 0,
            generation: 0,
            queued: false,
            watched: HashMap::new(),
            last_scan: None,
            changed_at: None,
        })
        .init_resource::<HotPatchChanges>()
        .add_message::<HotPatched>()
        .add_systems(First, (watch_sources, apply_patches).chain())
        .add_systems(
            Last,
            move |mut writer: MessageWriter<HotPatched>, mut changes: ResMut<HotPatchChanges>| {
                if patched_rx.try_recv().is_ok() {
                    writer.write_default();
                    changes.set_changed();
                }
            },
        );
    }
}

/// Request a patch once source files stop changing for a moment, so a burst
/// of edits builds once.
fn watch_sources(mut hot: ResMut<HotReload>) {
    if hot.requests.is_none() || hot.last_scan.is_some_and(|at| at.elapsed() < Duration::from_millis(250)) {
        return;
    }
    hot.last_scan = Some(Instant::now());
    let mut current = HashMap::new();
    scan(&Path::new(CRATE_DIR).join("src"), &mut current);
    if !hot.watched.is_empty() && current != hot.watched {
        hot.changed_at = Some(Instant::now());
    }
    hot.watched = current;
    if hot.changed_at.is_some_and(|at| at.elapsed() > Duration::from_millis(600)) {
        hot.changed_at = None;
        hot.request();
    }
}

fn scan(dir: &Path, out: &mut HashMap<PathBuf, SystemTime>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs")
            && let Ok(modified) = entry.metadata().and_then(|meta| meta.modified())
        {
            out.insert(path, modified);
        }
    }
}

/// Apply finished patches. Exclusive, so no system runs mid-swap.
fn apply_patches(world: &mut World) {
    let mut hot = world.resource_mut::<HotReload>();
    let Ok(result) = hot.results.try_recv() else {
        return;
    };
    hot.generation += 1;
    hot.building = std::mem::take(&mut hot.queued);
    if hot.building
        && let Some(requests) = &hot.requests
    {
        let _ = requests.send(());
    }
    hot.status = match result {
        // SAFETY: the table was built from this process's own fat
        // executable and ASLR slide, and patches only change function
        // bodies, never the layout of types shared with running code.
        Ok(patch) => match unsafe { subsecond::apply_patch(patch.table) } {
            Ok(()) => {
                hot.applied += 1;
                PatchStatus::Applied {
                    count: hot.applied,
                    elapsed: patch.elapsed,
                }
            }
            Err(error) => PatchStatus::Failed(error.to_string()),
        },
        Err(error) => PatchStatus::Failed(error),
    };
}
