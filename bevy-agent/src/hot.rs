//! Hot patching of this crate's native plugins. A source change under `src/`
//! (or `/patch`, or `agent.patch` over BRP) rebuilds the crate off the main
//! thread and applies the patch; Bevy then re-resolves every system.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use agent_hotpatch::{PatchReport, Patcher};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::{HotPatchChanges, HotPatched};

use crate::agent::Transcript;

const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

pub struct HotPatchPlugin;

impl Plugin for HotPatchPlugin {
    fn build(&self, app: &mut App) {
        let (tx, rx) = async_channel::unbounded();
        let (patched_tx, patched_rx) = async_channel::bounded::<()>(1);
        subsecond::register_handler(Arc::new(move || {
            let _ = patched_tx.try_send(());
        }));
        let ready = tx.clone();
        std::thread::spawn(move || {
            let patcher = Patcher::new(MANIFEST_DIR, "bevy-agent", "bevy-agent");
            let _ = ready.send_blocking(HotEvent::Ready(patcher.map_err(|e| e.to_string())));
        });
        app.insert_resource(Hot {
            patcher: None,
            tx,
            rx,
            running: false,
            requested: false,
            status: "loading".to_owned(),
            last_seen: newest_source(),
            last_check: Instant::now(),
        })
        .init_resource::<HotPatchChanges>()
        .add_message::<HotPatched>()
        .add_systems(Update, (watch_sources, drive))
        // In `First`, so every system's last run is older than the change and all
        // of them re-resolve their function pointers this frame.
        .add_systems(
            First,
            move |mut patched: MessageWriter<HotPatched>, mut changes: ResMut<HotPatchChanges>| {
                if patched_rx.try_recv().is_ok() {
                    patched.write_default();
                    changes.set_changed();
                }
            },
        );
    }
}

enum HotEvent {
    Ready(Result<Patcher, String>),
    Log(String),
    Done(Result<PatchReport, String>),
}

#[derive(Resource)]
pub struct Hot {
    patcher: Option<Arc<Patcher>>,
    tx: async_channel::Sender<HotEvent>,
    rx: async_channel::Receiver<HotEvent>,
    running: bool,
    requested: bool,
    status: String,
    last_seen: SystemTime,
    last_check: Instant,
}

impl Hot {
    /// Rebuild and patch as soon as the patcher is free.
    pub fn request(&mut self) {
        self.requested = true;
    }

    pub fn status(&self) -> &str {
        if self.running {
            "building"
        } else {
            &self.status
        }
    }
}

fn watch_sources(mut hot: ResMut<Hot>) {
    if hot.last_check.elapsed() < Duration::from_millis(500) {
        return;
    }
    hot.last_check = Instant::now();
    let newest = newest_source();
    if newest > hot.last_seen {
        hot.last_seen = newest;
        hot.request();
    }
}

fn drive(mut hot: ResMut<Hot>, mut transcript: ResMut<Transcript>) {
    while let Ok(event) = hot.rx.try_recv() {
        match event {
            HotEvent::Ready(Ok(patcher)) => {
                hot.patcher = Some(Arc::new(patcher));
                hot.status = "ready".to_owned();
            }
            HotEvent::Ready(Err(error)) => {
                hot.status = "unavailable".to_owned();
                transcript.note(format!("hot patching unavailable: {error}"));
            }
            HotEvent::Log(line) => transcript.note(line),
            HotEvent::Done(Ok(report)) => {
                hot.running = false;
                hot.status = format!("ok {:.1?}", report.build + report.link);
                transcript.note(format!(
                    "hot patched {} symbols from {} (build {:.1?}, link {:.1?})",
                    report.mapped,
                    report
                        .dylib
                        .file_name()
                        .map(|n| n.to_string_lossy())
                        .unwrap_or_default(),
                    report.build,
                    report.link
                ));
            }
            HotEvent::Done(Err(error)) => {
                hot.running = false;
                hot.status = "failed".to_owned();
                transcript.note(format!("hot patch failed: {error}"));
            }
        }
    }
    if hot.requested && !hot.running {
        let Some(patcher) = hot.patcher.clone() else {
            return;
        };
        hot.requested = false;
        hot.running = true;
        let tx = hot.tx.clone();
        std::thread::spawn(move || {
            let report = patcher.patch(&mut |line| {
                let _ = tx.send_blocking(HotEvent::Log(line));
            });
            let _ = tx.send_blocking(HotEvent::Done(report.map_err(|e| e.to_string())));
        });
    }
}

/// The newest modification time under `src/`.
fn newest_source() -> SystemTime {
    fn walk(dir: &Path, newest: &mut SystemTime) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, newest);
            } else if path.extension().is_some_and(|e| e == "rs")
                && let Ok(modified) = entry.metadata().and_then(|m| m.modified())
            {
                *newest = (*newest).max(modified);
            }
        }
    }
    let mut newest = SystemTime::UNIX_EPOCH;
    walk(&Path::new(MANIFEST_DIR).join("src"), &mut newest);
    newest
}
