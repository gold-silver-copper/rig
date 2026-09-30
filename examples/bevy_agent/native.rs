//! Editable native plugin. Only UTF-8 byte buffers cross the host boundary.
//! Change the function body, not its signature or ABI version.

#[unsafe(no_mangle)]
pub extern "C" fn main() {}

#[unsafe(no_mangle)]
pub extern "C" fn agent_plugin_version() -> u32 {
    1
}

/// The host provides valid, non-overlapping buffers for this synchronous call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn agent_plugin(
    input: *const u8,
    len: usize,
    output: *mut u8,
    capacity: usize,
) -> usize {
    let input = unsafe { std::slice::from_raw_parts(input, len) };
    let input = std::str::from_utf8(input).unwrap_or("invalid UTF-8");
    let reply = if input == "status" {
        "native-v1".to_owned()
    } else {
        format!("native-v1: {input}")
    };
    let count = reply.len().min(capacity);
    unsafe { std::ptr::copy_nonoverlapping(reply.as_ptr(), output, count) };
    count
}
