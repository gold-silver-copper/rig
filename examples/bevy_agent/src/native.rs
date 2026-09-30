use anyhow::{Context, Result, bail};
use bevy::{
    ecs::{HotPatchChanges, HotPatched},
    prelude::{App, DetectChangesMut, MessageWriter, Plugin, ResMut, Resource, Update},
};
use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::mpsc,
};
use subsecond::{HotFn, HotFunction, JumpTable};
use tokio::sync::oneshot;

type Entry = unsafe extern "C" fn(*const u8, usize, *mut u8, usize) -> usize;
type Args = (*const u8, usize, *mut u8, usize);
struct NativeFn(Entry);
struct NativeMarker;

impl HotFunction<Args, NativeMarker> for NativeFn {
    type Return = usize;
    type Real = Entry;
    fn call_it(&mut self, args: Args) -> usize {
        unsafe { (self.0)(args.0, args.1, args.2, args.3) }
    }
    unsafe fn call_as_ptr(&mut self, args: Args) -> usize {
        let mut entry = self.0;
        // The table only maps our fixed, version-checked C ABI entry point.
        if let Some(table) = unsafe { subsecond::get_jump_table() }
            && let Some(ptr) = table.map.get(&(entry as usize as u64))
        {
            entry = unsafe { std::mem::transmute::<usize, Entry>(*ptr as usize) };
        }
        unsafe { entry(args.0, args.1, args.2, args.3) }
    }
}

unsafe extern "C" fn baseline(_: *const u8, _: usize, _: *mut u8, _: usize) -> usize {
    0
}

pub fn invoke(input: &str) -> Result<String> {
    let mut output = vec![0_u8; 8192];
    let count = HotFn::current(NativeFn(baseline)).call((
        input.as_ptr(),
        input.len(),
        output.as_mut_ptr(),
        output.len(),
    ));
    if count > output.len() {
        bail!("native plugin returned an invalid length");
    }
    output.truncate(count);
    Ok(String::from_utf8(output)?)
}

struct Built {
    path: PathBuf,
}
type BuildResult = Result<Built>;

#[derive(Resource)]
pub struct Native {
    pub source: PathBuf,
    pub status: String,
    pub generation: u64,
    observed: Vec<u8>,
    pending: Option<std::sync::Mutex<mpsc::Receiver<BuildResult>>>,
    replies: Vec<oneshot::Sender<Result<String>>>,
    // Keep the validating handle alive as well as Subsecond's leaked handle.
    libraries: Vec<libloading::Library>,
    artifact_dir: PathBuf,
}

impl Native {
    pub fn new(source: PathBuf) -> Result<Self> {
        if !cfg!(debug_assertions) {
            bail!("native hot patches require a debug build");
        }
        let artifact_dir = source
            .parent()
            .context("native source needs a parent")?
            .join(".patches");
        std::fs::create_dir_all(&artifact_dir)?;
        let observed = std::fs::read(&source)?;
        let mut this = Self {
            source,
            status: String::new(),
            generation: 0,
            observed: observed.clone(),
            pending: None,
            replies: Vec::new(),
            libraries: Vec::new(),
            artifact_dir,
        };
        let built = compile(&this.artifact_dir, 0, &observed)?;
        this.apply(built)?;
        Ok(this)
    }

    fn apply(&mut self, built: Built) -> Result<()> {
        // Trusted native code only. Loading a library runs its constructors.
        let lib = unsafe { libloading::Library::new(&built.path)? };
        let version = unsafe { lib.get::<unsafe extern "C" fn() -> u32>(b"agent_plugin_version")? };
        let version = unsafe { version() };
        if version != 1 {
            bail!(
                "unsupported native ABI {version}: agent_plugin_version must return 1, even when editing behavior. Restore that function to return 1; keeping previous patch"
            );
        }
        let entry = unsafe { *lib.get::<Entry>(b"agent_plugin")? };
        let anchor = unsafe { *lib.get::<unsafe extern "C" fn()>(b"main")? };
        let mut table = JumpTable {
            lib: built.path,
            map: Default::default(),
            aslr_reference: subsecond::aslr_reference() as u64,
            new_base_address: anchor as usize as u64,
            ifunc_count: 0,
        };
        table
            .map
            .insert(baseline as *const () as usize as u64, entry as usize as u64);
        // Runtime addresses make both ASLR offsets zero. Signature and ABI are fixed.
        unsafe { subsecond::apply_patch(table)? };
        self.libraries.push(lib);
        self.generation += 1;
        self.status = invoke("status")?;
        Ok(())
    }

    pub fn rebuild(&mut self, reply: Option<oneshot::Sender<Result<String>>>) -> Result<()> {
        let result = self.start_build();
        if let Some(reply) = reply {
            match &result {
                Ok(()) => self.replies.push(reply),
                Err(error) => {
                    let _ = reply.send(Err(anyhow::anyhow!("{error:#}")));
                }
            }
        }
        result
    }

    fn start_build(&mut self) -> Result<()> {
        let bytes = std::fs::read(&self.source)?;
        if self.pending.is_some() {
            if bytes != self.observed {
                bail!("source changed during native build; retry shortly");
            }
            return Ok(());
        }
        self.observed = bytes.clone();
        let dir = self.artifact_dir.clone();
        let generation = self.generation;
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(compile(&dir, generation, &bytes));
        });
        self.pending = Some(std::sync::Mutex::new(rx));
        Ok(())
    }
}

fn compile(dir: &Path, generation: u64, bytes: &[u8]) -> BuildResult {
    let stem = format!(
        "patch-{}-{generation}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    );
    let source = dir.join(format!("{stem}.rs"));
    let library = dir.join(format!("{stem}{}", std::env::consts::DLL_SUFFIX));
    std::fs::write(&source, bytes)?;
    let result = Command::new("rustc")
        .args([
            "--edition=2024",
            "--crate-type=cdylib",
            "--crate-name=agent_native",
            "-C",
            "panic=abort",
        ])
        .arg(&source)
        .arg("-o")
        .arg(&library)
        .output()
        .context("launch rustc (dx is never used)")?;
    if !result.status.success() {
        bail!(
            "native compile failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    Ok(Built { path: library })
}

pub struct NativePlugin;
impl Plugin for NativePlugin {
    fn build(&self, app: &mut App) {
        // Replace Bevy's dx-connecting HotPatchPlugin, retaining its ECS notifications.
        app.init_resource::<HotPatchChanges>()
            .add_message::<HotPatched>()
            .add_systems(Update, poll);
    }
}

fn poll(
    mut native: ResMut<Native>,
    mut changes: ResMut<HotPatchChanges>,
    mut patched: MessageWriter<HotPatched>,
    mut ui: ResMut<crate::UiState>,
) {
    if let Some(result) = native
        .pending
        .as_ref()
        .and_then(|rx| rx.lock().ok()?.try_recv().ok())
    {
        native.pending = None;
        let result = result
            .and_then(|built| native.apply(built))
            .and_then(|()| invoke("status"));
        match &result {
            Ok(label) => {
                changes.set_changed();
                patched.write_default();
                ui.lines
                    .push(format!("Hot patch #{}: {label}", native.generation));
            }
            Err(error) => ui.lines.push(format!("Hot patch rejected: {error:#}")),
        }
        for reply in std::mem::take(&mut native.replies) {
            let output = match &result {
                Ok(label) => Ok(label.clone()),
                Err(error) => Err(anyhow::anyhow!("{error:#}")),
            };
            let _ = reply.send(output);
        }
    }
    if native.pending.is_none()
        && let Ok(bytes) = std::fs::read(&native.source)
        && bytes != native.observed
        && let Err(error) = native.rebuild(None)
    {
        ui.lines.push(format!("Hot patch: {error:#}"));
    }
}

#[cfg(test)]
mod tests;
