//! Symbol work for native patches, ported from dioxus-cli's `build/patch.rs`:
//! the fat executable's symbol table, the stub object that satisfies a
//! patch's undefined symbols with addresses in the running process, and the
//! jump table from old to new function addresses.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail};
use object::write::{MachOBuildVersion, StandardSection, Symbol, SymbolSection};
use object::{Object, ObjectSection, ObjectSymbol, SymbolFlags, SymbolKind, SymbolScope};
use subsecond::JumpTable;

use crate::capture::host_object_format;

/// The anchor symbol subsecond measures the ASLR slide against.
const MAIN: &str = if cfg!(target_os = "macos") {
    "_main"
} else {
    "main"
};

struct CachedSymbol {
    address: u64,
    kind: SymbolKind,
    is_undefined: bool,
    is_weak: bool,
    size: u64,
    flags: SymbolFlags<object::SectionIndex, object::SymbolIndex>,
}

/// The fat executable's symbols, parsed once and reused for every patch.
pub(crate) struct SymbolCache {
    symbols: HashMap<String, CachedSymbol>,
    /// The TLS initialization image (`.tdata` / `__thread_data`).
    tls_init_data: Vec<u8>,
    /// Mach-O `$tlv$init` symbol name to (offset, size) in the TLS image;
    /// nlist entries carry no sizes, so they are computed from neighbours.
    tls_init_sizes: HashMap<String, (u64, u64)>,
}

impl SymbolCache {
    pub(crate) fn new(exe: &Path) -> Result<Self> {
        let bytes = std::fs::read(exe)?;
        let file = object::File::parse(bytes.as_slice())?;
        let symbols = file
            .symbols()
            .filter_map(|symbol| {
                let flags = match symbol.flags() {
                    SymbolFlags::Elf { st_info, st_other } => SymbolFlags::Elf { st_info, st_other },
                    SymbolFlags::MachO { n_desc } => SymbolFlags::MachO { n_desc },
                    _ => SymbolFlags::None,
                };
                Some((
                    symbol.name().ok()?.to_owned(),
                    CachedSymbol {
                        address: symbol.address(),
                        kind: symbol.kind(),
                        is_undefined: symbol.is_undefined(),
                        is_weak: symbol.is_weak(),
                        size: symbol.size(),
                        flags,
                    },
                ))
            })
            .collect();

        let tls = file
            .sections()
            .find(|section| matches!(section.name(), Ok(".tdata" | "__thread_data")));
        let tls_init_data = tls
            .as_ref()
            .and_then(|section| section.data().ok())
            .unwrap_or_default()
            .to_vec();
        let tls_address = tls.as_ref().map_or(0, |section| section.address());
        let tls_size = tls.as_ref().map_or(0, |section| section.size());
        let tls_index = tls.as_ref().map(|section| section.index());

        let mut inits: Vec<(u64, String)> = file
            .symbols()
            .filter(|symbol| tls_index.is_some() && symbol.section_index() == tls_index)
            .filter_map(|symbol| {
                let offset = symbol.address().saturating_sub(tls_address);
                Some((offset, symbol.name().ok()?.to_owned()))
            })
            .collect();
        inits.sort_by_key(|(offset, _)| *offset);
        inits.dedup_by_key(|(offset, _)| *offset);
        let tls_init_sizes = inits
            .iter()
            .enumerate()
            .map(|(i, (offset, name))| {
                let end = inits.get(i + 1).map_or(tls_size, |(next, _)| *next);
                (name.clone(), (*offset, end.saturating_sub(*offset)))
            })
            .collect();

        Ok(Self {
            symbols,
            tls_init_data,
            tls_init_sizes,
        })
    }

    /// An object that defines every symbol `objects` need but do not define,
    /// pointing at the running process: functions become jumps to the
    /// original address, data becomes absolute symbols, and thread-locals
    /// get fresh storage initialized from the original TLS image.
    /// `aslr_reference` is the runtime address of `main`.
    pub(crate) fn stub(&self, objects: &[impl AsRef<Path>], aslr_reference: u64) -> Result<Vec<u8>> {
        let mut undefined = HashSet::new();
        let mut defined = HashSet::new();
        for path in objects {
            let bytes = std::fs::read(path.as_ref())?;
            let file = object::File::parse(bytes.as_slice())?;
            for symbol in file.symbols() {
                if symbol.is_undefined() {
                    undefined.insert(symbol.name()?.to_owned());
                } else if symbol.is_global() {
                    defined.insert(symbol.name()?.to_owned());
                }
            }
        }

        let (format, arch) = host_object_format()?;
        let mut obj = object::write::Object::new(format, arch, object::Endianness::Little);
        if cfg!(target_os = "macos") {
            let mut version = MachOBuildVersion::default();
            version.platform = object::macho::PLATFORM_MACOS;
            version.minos = 11 << 16;
            version.sdk = 11 << 16;
            obj.set_macho_build_version(version);
        }

        let main = self.symbols.get(MAIN).context("no main in the fat executable")?;
        if aslr_reference < main.address {
            bail!("ASLR reference {aslr_reference:#x} is below main at {:#x}", main.address);
        }
        let slide = aslr_reference - main.address;
        // The Mach-O writer adds the leading underscore back.
        let strip = usize::from(cfg!(target_os = "macos"));
        let text = obj.section_id(StandardSection::Text);

        for name in undefined.difference(&defined) {
            let Some(symbol) = self.symbols.get(name) else {
                continue;
            };
            if symbol.is_undefined {
                // Imports stay imports; the patch links the same dylibs.
                continue;
            }
            let address = symbol.address + slide;
            let stub_name = name.as_bytes()[strip..].to_vec();
            match symbol.kind {
                SymbolKind::Text => {
                    let code = jump_to(address);
                    let offset = obj.append_section_data(text, &code, 8);
                    obj.add_symbol(Symbol {
                        name: stub_name,
                        value: offset,
                        size: code.len() as u64,
                        scope: SymbolScope::Linkage,
                        kind: SymbolKind::Text,
                        weak: false,
                        section: SymbolSection::Section(text),
                        flags: SymbolFlags::None,
                    });
                }
                SymbolKind::Tls => {
                    // Each patch gets its own copy, so patched thread-locals
                    // restart from their initial value.
                    let tls = obj.section_id(StandardSection::Tls);
                    let (offset, size) = match self.tls_init_sizes.get(&format!("{name}$tlv$init")) {
                        Some(&found) => found,
                        None if symbol.size > 0 => (symbol.address, symbol.size),
                        None if !self.tls_init_sizes.is_empty() => {
                            (0, self.tls_init_data.len() as u64)
                        }
                        None => (symbol.address, 8),
                    };
                    let start = offset as usize;
                    let end = start + size as usize;
                    let init = self
                        .tls_init_data
                        .get(start..end)
                        .map_or_else(|| vec![0; size as usize], <[u8]>::to_vec);
                    let id = obj.add_symbol(Symbol {
                        name: stub_name,
                        value: 0,
                        size: 0,
                        scope: SymbolScope::Linkage,
                        kind: SymbolKind::Tls,
                        weak: false,
                        section: SymbolSection::Undefined,
                        flags: SymbolFlags::None,
                    });
                    obj.add_symbol_data(id, tls, &init, size.clamp(1, 8).next_power_of_two());
                }
                kind => {
                    obj.add_symbol(Symbol {
                        name: stub_name,
                        value: address,
                        size: 0,
                        scope: SymbolScope::Linkage,
                        // Darwin statics show up as unknown.
                        kind: if kind == SymbolKind::Unknown {
                            SymbolKind::Data
                        } else {
                            kind
                        },
                        weak: symbol.is_weak,
                        section: SymbolSection::Absolute,
                        flags: match symbol.flags {
                            SymbolFlags::Elf { st_info, st_other } => {
                                SymbolFlags::Elf { st_info, st_other }
                            }
                            SymbolFlags::MachO { n_desc } => SymbolFlags::MachO { n_desc },
                            _ => SymbolFlags::None,
                        },
                    });
                }
            }
        }
        Ok(obj.write()?)
    }

    /// Map every function address of the fat executable to its address in
    /// `patch`, by symbol name.
    pub(crate) fn jump_table(&self, patch: &Path) -> Result<JumpTable> {
        let bytes = std::fs::read(patch)?;
        let file = object::File::parse(bytes.as_slice())?;
        let symbols = file.symbol_map();
        let mut map = subsecond::JumpTable {
            lib: patch.to_path_buf(),
            map: Default::default(),
            aslr_reference: 0,
            new_base_address: 0,
            ifunc_count: 0,
        };
        let mut new_main = None;
        for symbol in symbols.symbols() {
            if symbol.name() == MAIN {
                new_main = Some(symbol.address());
            }
            if let Some(old) = self.symbols.get(symbol.name()) {
                map.map.insert(old.address, symbol.address());
            }
        }
        map.new_base_address = new_main.context("no main in the patch")?;
        map.aslr_reference = self.symbols.get(MAIN).context("no main in the fat executable")?.address;
        Ok(map)
    }
}

/// Machine code that jumps to an absolute address.
fn jump_to(address: u64) -> Vec<u8> {
    let mut code = if cfg!(target_arch = "aarch64") {
        // ldr x16, #8 ; br x16
        vec![0x50, 0x00, 0x00, 0x58, 0x00, 0x02, 0x1F, 0xD6]
    } else {
        // jmp [rip+0]
        vec![0xFF, 0x25, 0x00, 0x00, 0x00, 0x00]
    };
    code.extend_from_slice(&address.to_le_bytes());
    code
}
