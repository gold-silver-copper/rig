//! The executable acting as its own `RUSTC_WORKSPACE_WRAPPER` and linker.
//!
//! Cargo runs the wrapper as `<wrapper> <rustc> <args...>`; the wrapper
//! records the invocation so the tip crate can later be recompiled with the
//! exact same flags, then forwards to rustc. Rustc runs the linker with
//! `cc`-style arguments; the shim records them and writes an empty object
//! where the executable would go, because the real link happens afterwards
//! (see `fat.rs` and `thin.rs`).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Directory the wrapper and linker shim write their captures to. Its
/// presence is what turns an ordinary run into a wrapper or shim run.
pub(crate) const CAPTURE_ENV: &str = "RIGPI_HOTPATCH_CAPTURE";

/// One recorded rustc invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustcInvocation {
    /// The rustc cargo asked for.
    pub rustc: String,
    /// Every argument after the rustc path.
    pub args: Vec<String>,
    /// The full environment cargo gave rustc.
    pub envs: Vec<(String, String)>,
    /// The working directory cargo ran rustc in.
    pub cwd: PathBuf,
}

/// If this process was spawned as the rustc wrapper or the linker shim, do
/// that job and return the exit code the process must exit with. Returns
/// `None` for a normal run. Call it before anything else in `main`.
pub fn intercept() -> Option<i32> {
    let dir = PathBuf::from(std::env::var_os(CAPTURE_ENV)?);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let wrapping_rustc = args.first().is_some_and(|arg| {
        Path::new(arg)
            .file_stem()
            .is_some_and(|stem| stem == "rustc")
    });
    let result = if wrapping_rustc {
        wrap_rustc(&dir, args)
    } else {
        capture_link(&dir, args)
    };
    Some(result.unwrap_or_else(|error| {
        eprintln!("rigpi-hotpatch: {error:#}");
        1
    }))
}

fn wrap_rustc(dir: &Path, mut args: Vec<String>) -> Result<i32> {
    let rustc = args.remove(0);
    // Cargo probes the compiler with `--crate-name ___`; only real crates matter.
    if let Some(name) = flag_value(&args, "--crate-name").filter(|name| *name != "___") {
        let kind = match flag_value(&args, "--crate-type") {
            Some("bin") => "bin",
            _ => "lib",
        };
        let invocation = RustcInvocation {
            rustc: rustc.clone(),
            args: args.clone(),
            envs: std::env::vars().collect(),
            cwd: std::env::current_dir()?,
        };
        std::fs::create_dir_all(dir)?;
        std::fs::write(
            dir.join(format!("rustc-{name}.{kind}.json")),
            serde_json::to_vec(&invocation)?,
        )?;
    }
    let status = Command::new(&rustc)
        .args(&args)
        .status()
        .with_context(|| format!("running {rustc}"))?;
    Ok(status.code().unwrap_or(1))
}

fn capture_link(dir: &Path, args: Vec<String>) -> Result<i32> {
    let args = args
        .into_iter()
        .map(expand_response_file)
        .collect::<Result<Vec<_>>>()?
        .concat();
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("link-args.json"), serde_json::to_vec(&args)?)?;
    let out = flag_value(&args, "-o").context("linker invoked without -o")?;
    std::fs::write(out, empty_object()?)?;
    Ok(0)
}

pub(crate) fn read_link_args(dir: &Path) -> Result<Vec<String>> {
    let bytes = std::fs::read(dir.join("link-args.json"))
        .context("the linker shim did not run; was the tip crate linked?")?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub(crate) fn read_rustc(dir: &Path, crate_name: &str) -> Result<RustcInvocation> {
    let path = dir.join(format!("rustc-{crate_name}.bin.json"));
    let bytes = std::fs::read(&path)
        .with_context(|| format!("no captured rustc invocation at {}", path.display()))?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub(crate) fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let at = args.iter().position(|arg| arg == flag)?;
    args.get(at + 1).map(String::as_str)
}

fn expand_response_file(arg: String) -> Result<Vec<String>> {
    let Some(path) = arg.strip_prefix('@') else {
        return Ok(vec![arg]);
    };
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .lines()
        .map(|line| line.trim().trim_matches('"').to_owned())
        .filter(|line| !line.is_empty())
        .collect())
}

/// An object file with nothing in it, for the host's format.
fn empty_object() -> Result<Vec<u8>> {
    let (format, arch) = host_object_format()?;
    Ok(object::write::Object::new(format, arch, object::Endianness::Little).write()?)
}

pub(crate) fn host_object_format() -> Result<(object::BinaryFormat, object::Architecture)> {
    let format = if cfg!(target_os = "macos") {
        object::BinaryFormat::MachO
    } else if cfg!(target_os = "linux") {
        object::BinaryFormat::Elf
    } else {
        anyhow::bail!("hot-patching supports macOS and Linux only");
    };
    let arch = if cfg!(target_arch = "aarch64") {
        object::Architecture::Aarch64
    } else if cfg!(target_arch = "x86_64") {
        object::Architecture::X86_64
    } else {
        anyhow::bail!("hot-patching supports x86_64 and aarch64 only");
    };
    Ok((format, arch))
}
