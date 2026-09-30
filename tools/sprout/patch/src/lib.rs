//! Stable symbol discovery replaces dx's whole-program linker machinery.
#[unsafe(no_mangle)]
pub extern "C" fn main() {}

#[unsafe(no_mangle)]
pub extern "C" fn sprout_system() -> u64 {
    sprout_native::system_address()
}

#[unsafe(no_mangle)]
pub extern "C" fn sprout_abi() -> u64 {
    sprout_native::abi()
}
