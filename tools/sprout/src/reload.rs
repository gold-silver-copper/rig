//! Build changed native behavior off-thread, then patch at the frame boundary.
use anyhow::{Result, ensure};
use bevy::prelude::*;
use crossbeam_channel as mpsc;
use libloading::Library;
use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{Duration, Instant},
};

use crate::Agent;

#[derive(Resource)]
pub struct Reloader {
    root: PathBuf,
    source: Vec<u8>,
    last_check: Instant,
    pending: Option<mpsc::Receiver<Result<PathBuf>>>,
    original: u64,
    pub generation: u64,
}

impl Reloader {
    pub fn new(root: PathBuf) -> Result<Self> {
        Ok(Self {
            source: fs::read(root.join("native/src/behavior.rs"))?,
            root,
            last_check: Instant::now(),
            pending: None,
            original: sprout_native::system_address(),
            generation: 0,
        })
    }

    fn install(&mut self, path: PathBuf) -> Result<()> {
        // SAFETY: only locally compiled, trusted code from our fixed patch crate is loaded.
        // The original system, parameter layout, manifest and lockfile must match. The
        // loaded library stays resident in subsecond; no function pointer is unloaded.
        unsafe {
            let lib = Library::new(&path)?;
            let abi = lib.get::<unsafe extern "C" fn() -> u64>(b"sprout_abi")?;
            ensure!(
                abi() == sprout_native::abi(),
                "native ABI changed; restart required"
            );
            let entry = lib.get::<unsafe extern "C" fn() -> u64>(b"sprout_system")?;
            let anchor = lib.get::<*const ()>(b"main")?;
            subsecond::apply_patch(subsecond::JumpTable {
                lib: path,
                map: [(self.original, entry())].into_iter().collect(),
                aslr_reference: subsecond::aslr_reference() as u64,
                new_base_address: *anchor as u64,
                ifunc_count: 0,
            })?;
        }
        self.generation += 1;
        Ok(())
    }
}

pub fn reload(mut reload: ResMut<Reloader>, mut agent: ResMut<Agent>) {
    if let Some(rx) = &reload.pending {
        match rx.try_recv() {
            Ok(result) => {
                reload.pending = None;
                match result.and_then(|path| reload.install(path)) {
                    Ok(()) => agent.log(format!(
                        "native patched #{} (same process)",
                        reload.generation
                    )),
                    Err(e) => agent.log(format!("native patch rejected: {e:#}")),
                }
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                reload.pending = None;
                agent.log("native build worker disconnected".into());
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }
    if reload.pending.is_some() || reload.last_check.elapsed() < Duration::from_millis(500) {
        return;
    }
    reload.last_check = Instant::now();
    let Ok(source) = fs::read(reload.root.join("native/src/behavior.rs")) else {
        return;
    };
    if source == reload.source {
        return;
    }
    reload.source = source;
    agent.log("native change detected; compiling without dx…".into());
    let root = reload.root.clone();
    let generation = reload.generation + 1;
    let (tx, rx) = mpsc::unbounded();
    reload.pending = Some(rx);
    std::thread::spawn(move || {
        let _ = tx.send(build(root, generation));
    });
}

fn build(root: PathBuf, generation: u64) -> Result<PathBuf> {
    let output = Command::new("cargo")
        .args(["build", "--locked", "--workspace", "--target-dir"])
        .arg(root.join("target"))
        .current_dir(&root)
        .output()
        .map_err(|e| anyhow::anyhow!("start cargo for native patch: {e}"))?;
    let log = root.join(".sprout/native-build.log");
    fs::write(&log, &output.stderr)?;
    ensure!(
        output.status.success(),
        "cargo failed; see {}",
        log.display()
    );
    let file = format!(
        "{}sprout_patch{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    let destination = root.join(format!(
        ".sprout/patch-{}-{generation}{}",
        std::process::id(),
        std::env::consts::DLL_SUFFIX
    ));
    fs::copy(root.join("target/debug").join(file), &destination)?;
    Ok(destination)
}
