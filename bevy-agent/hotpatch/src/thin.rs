//! Thin builds: replay the tip crate's rustc invocation, then link its fresh
//! objects plus a symbol stub into a patch library for the running process.

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result, bail};

use crate::capture::{CAPTURE_ENV, flag_value, read_link_args};
use crate::fat::FatBuild;
use crate::stub::SymbolCache;

/// Builds patches for the running fat executable.
pub struct Patcher {
    fat: FatBuild,
    cache: Option<SymbolCache>,
    count: usize,
}

/// A linked patch, ready for [`subsecond::apply_patch`].
pub struct Patch {
    /// The jump table to apply.
    pub table: subsecond::JumpTable,
    /// Compiler warnings, rendered.
    pub warnings: Vec<String>,
    /// How long the compile and link took.
    pub elapsed: std::time::Duration,
}

impl Patcher {
    /// A patcher for the process running `fat`.
    pub fn new(fat: FatBuild) -> Self {
        Self {
            fat,
            cache: None,
            count: 0,
        }
    }

    /// Recompile the tip crate and link a patch against this process. A
    /// compile error is returned as the compiler's rendered diagnostics.
    pub fn build(&mut self) -> Result<Patch> {
        let started = Instant::now();
        self.count += 1;
        let dir = self.fat.out_dir.join(format!("patch-{}", self.count));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;

        let rustc = &self.fat.rustc;
        let output = Command::new(&rustc.rustc)
            .current_dir(&rustc.cwd)
            .env_clear()
            .envs(
                rustc
                    .envs
                    .iter()
                    // The jobserver cargo handed out is gone.
                    .filter(|(key, _)| {
                        !matches!(key.as_str(), "CARGO_MAKEFLAGS" | "MAKEFLAGS" | "MFLAGS" | CAPTURE_ENV)
                    })
                    .map(|(key, value)| (key, value)),
            )
            .env(CAPTURE_ENV, &dir)
            .args(&rustc.args)
            .output()
            .context("running rustc")?;
        let (errors, warnings) = diagnostics(&output.stderr);
        if !output.status.success() {
            bail!("compile failed:\n{}", errors.join("\n"));
        }

        let link_args = read_link_args(&dir)?;
        let mut objects: Vec<PathBuf> = link_args
            .iter()
            .filter(|arg| arg.ends_with(".rcgu.o"))
            .map(PathBuf::from)
            .collect();
        objects.sort();
        // rustc's output here is the shim's empty object; a stale file at
        // that path confuses later dlopens.
        if let Some(out) = flag_value(&link_args, "-o") {
            let _ = std::fs::remove_file(out);
        }

        let cache = match &mut self.cache {
            Some(cache) => cache,
            empty => empty.insert(SymbolCache::new(&self.fat.exe)?),
        };
        let stub = dir.join("stub.o");
        std::fs::write(&stub, cache.stub(&objects, subsecond::aslr_reference() as u64)?)?;

        let extension = if cfg!(target_os = "macos") { "dylib" } else { "so" };
        let lib = dir.join(format!("patch-{}.{extension}", self.count));
        let output = Command::new("cc")
            .args(&objects)
            .arg(&stub)
            .args(link_args.iter().filter(|arg| arg.ends_with(".dylib") || arg.ends_with(".so")))
            .args(thin_link_args(&link_args))
            .arg("-o")
            .arg(&lib)
            .output()?;
        if !output.status.success() {
            bail!("patch link failed:\n{}", String::from_utf8_lossy(&output.stderr));
        }
        for object in &objects {
            let _ = std::fs::remove_file(object);
        }

        Ok(Patch {
            table: cache.jump_table(&lib)?,
            warnings,
            elapsed: started.elapsed(),
        })
    }
}

/// The original link arguments reduced to what a shared library needs:
/// system libraries, frameworks, search paths, and target flags. The rlibs
/// are dropped because the stub resolves their symbols in the running
/// process.
fn thin_link_args(original: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    if cfg!(target_os = "macos") {
        out.push("-Wl,-dylib".to_owned());
        for (i, arg) in original.iter().enumerate() {
            if matches!(arg.as_str(), "-framework" | "-arch" | "-L" | "-target")
                && let Some(value) = original.get(i + 1)
            {
                out.extend([arg.clone(), value.clone()]);
            }
            if arg.starts_with("-l") || arg.starts_with("-m") || arg.starts_with("-nodefaultlibs") {
                out.push(arg.clone());
            }
        }
    } else {
        out.extend(
            [
                "-shared",
                "-Wl,--eh-frame-hdr",
                "-Wl,-z,noexecstack",
                "-Wl,-z,relro,-z,now",
                "-nodefaultlibs",
                "-Wl,-Bdynamic",
            ]
            .map(str::to_owned),
        );
        for (i, arg) in original.iter().enumerate() {
            if arg == "-L"
                && let Some(value) = original.get(i + 1)
            {
                out.extend([arg.clone(), value.clone()]);
            }
            if arg.starts_with("-l")
                || arg.starts_with("-m")
                || arg.starts_with("-Wl,--target=")
                || arg.starts_with("-Wl,-fuse-ld")
                || arg.starts_with("-fuse-ld")
                || arg.starts_with("-B")
                || arg.contains("-ld-path")
            {
                out.push(arg.clone());
            }
        }
    }
    out
}

/// Split rustc's JSON diagnostics into rendered errors and warnings. Lines
/// that are not JSON diagnostics count as errors.
fn diagnostics(stderr: &[u8]) -> (Vec<String>, Vec<String>) {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    for line in String::from_utf8_lossy(stderr).lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            if !line.trim().is_empty() {
                errors.push(line.to_owned());
            }
            continue;
        };
        let Some(rendered) = value.get("rendered").and_then(|r| r.as_str()) else {
            continue;
        };
        let rendered = strip_ansi(rendered);
        match value.get("level").and_then(|l| l.as_str()) {
            Some("warning") => warnings.push(rendered),
            _ => errors.push(rendered),
        }
    }
    (errors, warnings)
}

fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // CSI sequences end at the first letter.
            chars.by_ref().find(char::is_ascii_alphabetic);
        } else {
            out.push(c);
        }
    }
    out
}
