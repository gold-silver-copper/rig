use super::*;

#[test]
fn patches_preserve_host_and_reject_failed_builds_and_abi_changes() -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(".patches")
        .join(format!("test-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let source = dir.join("native.rs");
    let original = include_str!("../../native.rs");
    std::fs::write(&source, original)?;
    let mut native = Native::new(source)?;
    anyhow::ensure!(invoke("status")? == "native-v1");
    let patched = original.replace("native-v1", "native-test-v2");
    native.apply(compile(&dir, 2, patched.as_bytes())?)?;
    anyhow::ensure!(invoke("hello")? == "native-test-v2: hello");
    anyhow::ensure!(native.generation == 2);
    anyhow::ensure!(compile(&dir, 3, b"not valid Rust").is_err());
    let incompatible = patched.replace("    1\n", "    2\n");
    anyhow::ensure!(
        native
            .apply(compile(&dir, 4, incompatible.as_bytes())?)
            .is_err()
    );
    anyhow::ensure!(invoke("status")? == "native-test-v2");
    anyhow::ensure!(native.generation == 2);
    std::fs::remove_file(&native.source)?;
    let (tx, rx) = oneshot::channel();
    anyhow::ensure!(native.rebuild(Some(tx)).is_err());
    anyhow::ensure!(rx.blocking_recv()?.is_err());
    Ok(())
}
