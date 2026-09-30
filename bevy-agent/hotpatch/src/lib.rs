//! A minimal subsecond patcher, ported from the Dioxus CLI's hotpatch engine.
//!
//! The running binary is the "base". A patch rebuilds the tip crate with cargo,
//! links its fresh incremental objects into a dylib whose undefined symbols
//! resolve to the base's addresses in this very process, and hands subsecond a
//! jump table mapping every base function to its copy in the patch.
//!
//! The base must be built with [`RUSTC_FLAGS`] so its objects survive, nothing
//! is dead-stripped, every dependency object is loaded and `main` is exported
//! as the ASLR anchor. The rebuild uses the same flags so symbol names match.

use anyhow::{Context, Result, bail};
use object::write::{MachOBuildVersion, Object, StandardSection, Symbol, SymbolSection};
use object::{
    Architecture, BinaryFormat, Endianness, File, Object as _, ObjectSection, ObjectSymbol,
    SymbolFlags, SymbolKind, SymbolScope,
};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use subsecond::JumpTable;

/// Extra `rustc` flags for the tip crate, for the base build and every patch build.
#[cfg(target_os = "macos")]
pub const RUSTC_FLAGS: &[&str] = &[
    "-Csave-temps",
    "-Clink-dead-code",
    "-Clink-arg=-Wl,-all_load",
    "-Clink-arg=-Wl,-exported_symbol,_main",
];
#[cfg(not(target_os = "macos"))]
pub const RUSTC_FLAGS: &[&str] = &[
    "-Csave-temps",
    "-Clink-dead-code",
    "-Clink-arg=-Wl,--export-dynamic-symbol,main",
];

/// The `cargo rustc` invocation that builds `bin` of `package` with [`RUSTC_FLAGS`],
/// reporting as JSON messages when `json`.
pub fn build_command(manifest_dir: &Path, package: &str, bin: &str, json: bool) -> Command {
    let mut cmd = Command::new("cargo");
    cmd.current_dir(manifest_dir)
        .args(["rustc", "-p", package, "--bin", bin]);
    if json {
        cmd.arg("--message-format=json");
    }
    cmd.arg("--").args(RUSTC_FLAGS);
    cmd
}

/// What one applied patch did.
#[derive(Debug, Clone)]
pub struct PatchReport {
    pub dylib: PathBuf,
    pub mapped: usize,
    pub build: Duration,
    pub link: Duration,
}

struct BaseSymbol {
    address: u64,
    kind: SymbolKind,
    undefined: bool,
    weak: bool,
    size: u64,
    flags: SymbolFlags<object::SectionIndex, object::SymbolIndex>,
}

/// Rebuilds and patches one binary crate into the current process.
pub struct Patcher {
    manifest_dir: PathBuf,
    package: String,
    bin: String,
    symbols: HashMap<String, BaseSymbol>,
    tls_init: Vec<u8>,
    tls_sizes: HashMap<String, (u64, u64)>,
    aslr_reference: u64,
}

impl Patcher {
    /// Read the running executable's symbol table. Fails when the base was not
    /// built with [`RUSTC_FLAGS`], since then `main` is not exported.
    pub fn new(manifest_dir: impl Into<PathBuf>, package: &str, bin: &str) -> Result<Self> {
        let aslr_reference = subsecond::aslr_reference() as u64;
        if aslr_reference == 0 {
            bail!(
                "`main` is not exported from this binary; build it with `{}`",
                flags_hint()
            );
        }
        let exe = std::env::current_exe()?;
        let bytes = std::fs::read(&exe).with_context(|| format!("read {}", exe.display()))?;
        let file = File::parse(&*bytes)?;
        let symbols = file
            .symbols()
            .filter_map(|s| {
                let name = s.name().ok()?.to_owned();
                // Only ELF wants the base's flags back; Mach-O n_type would override
                // the absolute section the stub sets.
                let flags = match s.flags() {
                    SymbolFlags::Elf { st_info, st_other } => {
                        SymbolFlags::Elf { st_info, st_other }
                    }
                    _ => SymbolFlags::None,
                };
                Some((
                    name,
                    BaseSymbol {
                        address: s.address(),
                        kind: s.kind(),
                        undefined: s.is_undefined(),
                        weak: s.is_weak(),
                        size: s.size(),
                        flags,
                    },
                ))
            })
            .collect::<HashMap<_, _>>();
        if !symbols.contains_key(main_symbol()) {
            bail!(
                "no `{}` in {}; is it stripped?",
                main_symbol(),
                exe.display()
            );
        }

        // TLS init image and, on Mach-O (no symbol sizes), sizes from adjacent symbols.
        let tls = file
            .sections()
            .find(|s| matches!(s.name(), Ok(".tdata" | "__thread_data")));
        let tls_init = tls
            .as_ref()
            .and_then(|s| s.data().ok())
            .unwrap_or_default()
            .to_vec();
        let mut tls_sizes = HashMap::new();
        if let Some(section) = &tls {
            let mut syms: Vec<(u64, String)> = file
                .symbols()
                .filter(|s| s.section_index() == Some(section.index()))
                .filter_map(|s| Some((s.address() - section.address(), s.name().ok()?.to_owned())))
                .collect();
            syms.sort_by_key(|(offset, _)| *offset);
            syms.dedup_by_key(|(offset, _)| *offset);
            for (i, (offset, name)) in syms.iter().enumerate() {
                let end = syms.get(i + 1).map_or(section.size(), |(next, _)| *next);
                tls_sizes.insert(name.clone(), (*offset, end - offset));
            }
        }

        Ok(Self {
            manifest_dir: manifest_dir.into(),
            package: package.to_owned(),
            bin: bin.to_owned(),
            symbols,
            tls_init,
            tls_sizes,
            aslr_reference,
        })
    }

    /// Rebuild the crate, link a patch and apply it. `log` receives progress lines.
    pub fn patch(&self, log: &mut dyn FnMut(String)) -> Result<PatchReport> {
        let started = Instant::now();
        let objects = self.rebuild(log)?;
        let build = started.elapsed();
        if objects.is_empty() {
            bail!(
                "nothing to patch: the rebuild wrote no objects (unchanged source, or a base not built with `{}`)",
                flags_hint()
            );
        }

        let out_dir = self.manifest_dir.join("target").join("hotpatch");
        std::fs::create_dir_all(&out_dir)?;
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        let stub = out_dir.join(format!("stub-{millis}.o"));
        std::fs::write(&stub, self.stub(&objects)?)?;

        let linking = Instant::now();
        let dylib = out_dir.join(format!("lib{}-patch-{millis}.{}", self.bin, DYLIB_EXT));
        let mut cc = Command::new("cc");
        cc.args(LINK_FLAGS)
            .args(&objects)
            .arg(&stub)
            .arg("-o")
            .arg(&dylib);
        let output = cc.output().context("run cc")?;
        if !output.status.success() {
            bail!(
                "patch link failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let link = linking.elapsed();
        log(format!("linked {}", dylib.display()));

        let table = self.jump_table(&dylib)?;
        let mapped = table.map.len();
        // SAFETY: the map pairs symbols of the same name from the same source, built
        // with the same flags, so signatures and layouts agree.
        unsafe { subsecond::apply_patch(table) }
            .map_err(|e| anyhow::anyhow!("apply patch: {e}"))?;
        Ok(PatchReport {
            dylib,
            mapped,
            build,
            link,
        })
    }

    /// Run cargo and return every object of the tip crate. The crate's old objects
    /// are removed first: rustc copies each codegen unit it links, cached or not,
    /// so what remains after the build is exactly the linked set, and nothing
    /// when the crate was unchanged.
    fn rebuild(&self, log: &mut dyn FnMut(String)) -> Result<Vec<PathBuf>> {
        let deps = self.manifest_dir.join("target").join("debug").join("deps");
        let prefix = format!("{}-", self.bin.replace('-', "_"));
        let is_object = |name: &str| {
            name.strip_prefix(&prefix).is_some_and(|rest| {
                rest.len() > 17
                    && rest.as_bytes()[16] == b'.'
                    && rest[..16].bytes().all(|b| b.is_ascii_hexdigit())
            }) && (name.ends_with(".rcgu.o") || name.ends_with(".rcgu.bc"))
        };
        if deps.is_dir() {
            for entry in std::fs::read_dir(&deps)?.flatten() {
                if entry.file_name().to_str().is_some_and(is_object) {
                    std::fs::remove_file(entry.path())?;
                }
            }
        }

        let output = build_command(&self.manifest_dir, &self.package, &self.bin, true)
            .output()
            .context("run cargo")?;
        let mut diagnostics = String::new();
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let Ok(message) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            if message["reason"] == "compiler-message"
                && let Some(rendered) = message["message"]["rendered"].as_str()
            {
                if message["message"]["level"] == "error" {
                    diagnostics.push_str(rendered);
                } else {
                    log(rendered.trim_end().to_owned());
                }
            }
        }
        if !output.status.success() {
            if diagnostics.is_empty() {
                diagnostics = String::from_utf8_lossy(&output.stderr).into_owned();
            }
            bail!("build failed:\n{}", diagnostics.trim_end());
        }

        let mut objects: Vec<PathBuf> = std::fs::read_dir(&deps)?
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| is_object(n) && n.ends_with(".rcgu.o"))
            })
            .collect();
        objects.sort();
        if objects.is_empty() {
            log("cargo: nothing changed".to_owned());
        } else {
            log(format!("{} objects rebuilt", objects.len()));
        }
        Ok(objects)
    }

    /// An object defining every symbol the patch objects need from the base:
    /// trampolines for functions, absolute symbols for data, fresh TLS slots.
    fn stub(&self, objects: &[PathBuf]) -> Result<Vec<u8>> {
        let mut undefined = HashSet::new();
        let mut defined = HashSet::new();
        for path in objects {
            let bytes = std::fs::read(path)?;
            let file = File::parse(&*bytes)?;
            for symbol in file.symbols() {
                let name = symbol.name()?.to_owned();
                if symbol.is_undefined() {
                    undefined.insert(name);
                } else if symbol.is_global() {
                    defined.insert(name);
                }
            }
        }

        let mut obj = Object::new(FORMAT, ARCH, Endianness::Little);
        if FORMAT == BinaryFormat::MachO {
            let mut version = MachOBuildVersion::default();
            version.platform = object::macho::PLATFORM_MACOS;
            version.minos = object::macho::Version::new(11, 0, 0);
            version.sdk = object::macho::Version::new(11, 0, 0);
            obj.set_macho_build_version(version);
        }
        let base_main = self.symbols[main_symbol()].address;
        if self.aslr_reference < base_main {
            bail!(
                "ASLR reference {:#x} below main {:#x}",
                self.aslr_reference,
                base_main
            );
        }
        let slide = self.aslr_reference - base_main;
        let strip = usize::from(FORMAT == BinaryFormat::MachO);
        let text = obj.section_id(StandardSection::Text);

        let mut names: Vec<_> = undefined.difference(&defined).collect();
        names.sort();
        for name in names {
            let Some(sym) = self.symbols.get(name) else {
                continue;
            };
            if sym.undefined {
                continue;
            }
            let address = sym.address + slide;
            let symbol_name = name.as_bytes()[strip..].to_vec();
            match sym.kind {
                SymbolKind::Text => {
                    let code = trampoline(address);
                    let value = obj.append_section_data(text, &code, 8);
                    obj.add_symbol(Symbol {
                        name: symbol_name,
                        value,
                        size: code.len() as u64,
                        kind: SymbolKind::Text,
                        scope: SymbolScope::Linkage,
                        weak: false,
                        section: SymbolSection::Section(text),
                        flags: SymbolFlags::None,
                    });
                }
                SymbolKind::Tls => {
                    let (offset, size) = self
                        .tls_sizes
                        .get(&format!("{name}$tlv$init"))
                        .copied()
                        .or_else(|| (sym.size > 0).then_some((sym.address, sym.size)))
                        .unwrap_or((0, 8));
                    let (start, end) = (offset as usize, (offset + size) as usize);
                    let init = if end <= self.tls_init.len() {
                        self.tls_init[start..end].to_vec()
                    } else {
                        vec![0; size as usize]
                    };
                    let id = obj.add_symbol(Symbol {
                        name: symbol_name,
                        value: 0,
                        size: 0,
                        kind: SymbolKind::Tls,
                        scope: SymbolScope::Linkage,
                        weak: false,
                        section: SymbolSection::Undefined,
                        flags: SymbolFlags::None,
                    });
                    let tls = obj.section_id(StandardSection::Tls);
                    obj.add_symbol_data(id, tls, &init, size.min(8).next_power_of_two());
                }
                kind => {
                    let flags = match sym.flags {
                        SymbolFlags::Elf { st_info, st_other } => {
                            SymbolFlags::Elf { st_info, st_other }
                        }
                        _ => SymbolFlags::None,
                    };
                    obj.add_symbol(Symbol {
                        name: symbol_name,
                        value: address,
                        size: 0,
                        kind: if kind == SymbolKind::Unknown {
                            SymbolKind::Data
                        } else {
                            kind
                        },
                        scope: SymbolScope::Linkage,
                        weak: sym.weak,
                        section: SymbolSection::Absolute,
                        flags,
                    });
                }
            }
        }
        Ok(obj.write()?)
    }

    /// Pair every base symbol with its namesake in the patch.
    fn jump_table(&self, dylib: &Path) -> Result<JumpTable> {
        let bytes = std::fs::read(dylib)?;
        let file = File::parse(&*bytes)?;
        let mut map = subsecond::JumpTable {
            lib: dylib.to_path_buf(),
            map: Default::default(),
            aslr_reference: self.symbols[main_symbol()].address,
            new_base_address: 0,
            ifunc_count: 0,
        };
        for symbol in file.symbol_map().symbols() {
            if symbol.name() == main_symbol() {
                map.new_base_address = symbol.address();
            }
            if let Some(old) = self.symbols.get(symbol.name()) {
                map.map.insert(old.address, symbol.address());
            }
        }
        if map.new_base_address == 0 {
            bail!("no `{}` in {}", main_symbol(), dylib.display());
        }
        Ok(map)
    }
}

fn flags_hint() -> String {
    format!("cargo rustc --bin <bin> -- {}", RUSTC_FLAGS.join(" "))
}

const fn main_symbol() -> &'static str {
    if cfg!(target_os = "macos") {
        "_main"
    } else {
        "main"
    }
}

#[cfg(target_os = "macos")]
const FORMAT: BinaryFormat = BinaryFormat::MachO;
#[cfg(not(target_os = "macos"))]
const FORMAT: BinaryFormat = BinaryFormat::Elf;

#[cfg(target_arch = "aarch64")]
const ARCH: Architecture = Architecture::Aarch64;
#[cfg(target_arch = "x86_64")]
const ARCH: Architecture = Architecture::X86_64;

#[cfg(target_os = "macos")]
const DYLIB_EXT: &str = "dylib";
#[cfg(not(target_os = "macos"))]
const DYLIB_EXT: &str = "so";

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const LINK_FLAGS: &[&str] = &["-dylib", "-arch", "arm64", "-Wl,-undefined,dynamic_lookup"];
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const LINK_FLAGS: &[&str] = &["-dylib", "-arch", "x86_64", "-Wl,-undefined,dynamic_lookup"];
#[cfg(not(target_os = "macos"))]
const LINK_FLAGS: &[&str] = &["-shared", "-Wl,-z,now"];

/// Machine code that jumps to `address`.
fn trampoline(address: u64) -> Vec<u8> {
    let mut code = Vec::with_capacity(16);
    if ARCH == Architecture::Aarch64 {
        // ldr x16, #8 ; br x16 ; .quad address
        code.extend_from_slice(&[0x50, 0x00, 0x00, 0x58, 0x00, 0x02, 0x1F, 0xD6]);
    } else {
        // jmp [rip+0] ; .quad address
        code.extend_from_slice(&[0xFF, 0x25, 0x00, 0x00, 0x00, 0x00]);
    }
    code.extend_from_slice(&address.to_le_bytes());
    code
}
