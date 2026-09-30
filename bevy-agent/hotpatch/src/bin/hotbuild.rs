//! Build a binary crate with the flags the patcher needs:
//! `hotbuild [manifest-dir] [package] [bin]`.

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let manifest_dir = args
        .next()
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_default();
    let package = args.next().unwrap_or_else(|| "bevy-agent".to_owned());
    let bin = args.next().unwrap_or_else(|| package.clone());
    match agent_hotpatch::build_command(&manifest_dir, &package, &bin, false).status() {
        Ok(status) if status.success() => ExitCode::SUCCESS,
        Ok(_) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("cargo: {error}");
            ExitCode::FAILURE
        }
    }
}
