//! The fat build: the tip crate built by cargo with the wrapper and shim in
//! place, then linked with every object of every dependency rlib force-loaded
//! so a later patch can call any dependency symbol.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::capture::{CAPTURE_ENV, RustcInvocation, flag_value, read_link_args, read_rustc};

/// Points a relaunched process at its [`FatBuild`] record.
const FAT_ENV: &str = "RIGPI_HOTPATCH_FAT";

/// What to build.
#[derive(Debug, Clone)]
pub struct FatOptions {
    /// Directory holding the package's `Cargo.toml`.
    pub manifest_dir: PathBuf,
    /// Cargo package name.
    pub package: String,
    /// Binary target name.
    pub bin: String,
    /// The binary's crate root. Its mtime is bumped so cargo always reruns
    /// rustc (and so the wrapper always captures) for the tip crate.
    pub crate_root: PathBuf,
    /// Where the fat executable, captures, and patches go.
    pub out_dir: PathBuf,
}

/// A finished fat build: the executable and how its tip crate was compiled
/// and linked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FatBuild {
    /// The fat executable.
    pub exe: PathBuf,
    /// The tip crate's rustc invocation, replayed for every patch.
    pub rustc: RustcInvocation,
    /// The tip crate's original linker arguments.
    pub link_args: Vec<String>,
    /// Where captures and patches go.
    pub out_dir: PathBuf,
}

impl FatBuild {
    /// The fat build this process was relaunched into, if any.
    pub fn current() -> Option<Self> {
        let path = std::env::var_os(FAT_ENV)?;
        let bytes = std::fs::read(path).ok()?;
        serde_json::from_slice(&bytes).ok()
    }
}

/// Build the fat executable. Cargo output goes to this process's stderr.
pub fn build_fat(options: &FatOptions) -> Result<FatBuild> {
    let out_dir = &options.out_dir;
    let capture = out_dir.join("capture-fat");
    let _ = std::fs::remove_dir_all(&capture);
    std::fs::create_dir_all(&capture)?;

    // Cargo replaces its build outputs while they may be running, so the
    // wrapper and shim run from a private copy of this executable.
    let shim = out_dir.join("rustc-shim");
    let _ = std::fs::remove_file(&shim);
    std::fs::copy(std::env::current_exe()?, &shim).context("copying the shim")?;

    std::fs::File::options()
        .append(true)
        .open(&options.crate_root)?
        .set_modified(SystemTime::now())?;

    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = Command::new(cargo)
        .current_dir(&options.manifest_dir)
        .args(["rustc", "-p", &options.package, "--bin", &options.bin, "--"])
        .args(["-Csave-temps=true", "-Clink-dead-code"])
        .arg(format!("-Clinker={}", shim.display()))
        .env("RUSTC_WORKSPACE_WRAPPER", &shim)
        .env(CAPTURE_ENV, &capture)
        .status()
        .context("running cargo")?;
    ensure!(status.success(), "the fat build failed");

    let rustc = read_rustc(&capture, &options.bin.replace('-', "_"))?;
    let link_args = read_link_args(&capture)?;
    let exe = out_dir.join(format!("{}-fat", options.bin));
    fat_link(&rustc, &link_args, &exe, out_dir)?;

    let fat = FatBuild {
        exe,
        rustc,
        link_args,
        out_dir: out_dir.clone(),
    };
    std::fs::write(out_dir.join("fat.json"), serde_json::to_vec(&fat)?)?;
    Ok(fat)
}

/// Replace this process with the fat executable, same arguments. Only
/// returns on failure.
pub fn relaunch(fat: &FatBuild) -> std::io::Error {
    Command::new(&fat.exe)
        .args(std::env::args_os().skip(1))
        .env(FAT_ENV, fat.out_dir.join("fat.json"))
        .exec()
}

fn fat_link(rustc: &RustcInvocation, args: &[String], exe: &Path, out_dir: &Path) -> Result<()> {
    // Dependency rlibs sit next to the tip crate's output; toolchain rlibs
    // (std and friends) are left to the normal link.
    let deps_dir = PathBuf::from(flag_value(&rustc.args, "--out-dir").context("no --out-dir")?);
    let rlibs: Vec<PathBuf> = args
        .iter()
        .filter(|arg| arg.ends_with(".rlib"))
        .map(PathBuf::from)
        .collect();

    let mut hasher = DefaultHasher::new();
    for rlib in &rlibs {
        rlib.hash(&mut hasher);
        if let Ok(meta) = rlib.metadata() {
            meta.len().hash(&mut hasher);
            meta.modified()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .hash(&mut hasher);
        }
    }
    let key = format!("{:016x}", hasher.finish());
    let archive = out_dir.join(format!("libdeps-{key}.a"));
    let kept_list = out_dir.join(format!("rlibs-{key}.json"));

    let kept: Vec<PathBuf> = match std::fs::read(&kept_list) {
        Ok(bytes) if archive.exists() => serde_json::from_slice(&bytes)?,
        _ => {
            let kept = write_fat_archive(&rlibs, &deps_dir, &archive)?;
            std::fs::write(&kept_list, serde_json::to_vec(&kept)?)?;
            kept
        }
    };

    let mut args = args.to_vec();
    args.retain(|arg| !arg.ends_with(".rlib"));
    if let Some(at) = args.iter().rposition(|arg| arg.ends_with(".o")) {
        let mut inserted = if cfg!(target_os = "macos") {
            vec!["-Wl,-force_load".to_owned(), archive.display().to_string()]
        } else {
            vec![
                "-Wl,--whole-archive".to_owned(),
                archive.display().to_string(),
                "-Wl,--no-whole-archive".to_owned(),
            ]
        };
        inserted.extend(kept.iter().map(|rlib| rlib.display().to_string()));
        args.splice(at..at, inserted);
    }
    // `main` anchors the ASLR slide for subsecond, so it must be exported.
    args.push(if cfg!(target_os = "macos") {
        "-Wl,-exported_symbol,_main".to_owned()
    } else {
        "-Wl,--export-dynamic-symbol,main".to_owned()
    });
    if let Some(at) = args.iter().position(|arg| arg == "-o") {
        args.drain(at..at + 2);
    }
    args.push("-o".to_owned());
    args.push(exe.display().to_string());

    let output = Command::new("cc").args(&args).output()?;
    ensure!(
        output.status.success(),
        "fat link failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    for object in args.iter().filter(|arg| arg.ends_with(".rcgu.o")) {
        let _ = std::fs::remove_file(object);
    }
    Ok(())
}

/// Pack every `.rcgu.o` of the dependency rlibs into one archive. Returns
/// the rlibs that must still be linked normally: toolchain rlibs, and rlibs
/// carrying other kinds of objects.
fn write_fat_archive(rlibs: &[PathBuf], deps_dir: &Path, archive: &Path) -> Result<Vec<PathBuf>> {
    let mut kept = Vec::new();
    let mut builder = ar::Builder::new(std::fs::File::create(archive)?);
    for rlib in rlibs {
        if !rlib.starts_with(deps_dir) {
            kept.push(rlib.clone());
            continue;
        }
        let mut reader = ar::Archive::new(std::fs::File::open(rlib)?);
        let mut keep = false;
        while let Some(entry) = reader.next_entry() {
            let mut entry = entry?;
            let name = String::from_utf8_lossy(entry.header().identifier()).into_owned();
            if name.ends_with(".rmeta") || entry.header().size() == 0 {
                continue;
            }
            if !name.ends_with(".rcgu.o") {
                keep = true;
                continue;
            }
            let header = entry.header().clone();
            let mut bytes = Vec::with_capacity(header.size() as usize);
            entry.read_to_end(&mut bytes)?;
            builder.append(&header, bytes.as_slice())?;
        }
        if keep {
            kept.push(rlib.clone());
        }
    }
    drop(builder);
    if cfg!(target_os = "macos") {
        // ld64 wants a symbol index; failure only costs link speed.
        let _ = Command::new("ranlib").arg(archive).output();
    }
    Ok(kept)
}
