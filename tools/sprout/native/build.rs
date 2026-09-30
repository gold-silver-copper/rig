fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = std::process::Command::new(rustc)
        .arg("--version")
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other("rustc --version failed").into());
    }
    let version = String::from_utf8_lossy(&output.stdout);
    println!("cargo:rustc-env=SPROUT_RUSTC={}", version.trim());
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
