use super::*;

#[test]
fn extension_validation_rejects_builtin_replacement_and_remote_callbacks() {
    let mut tool = Extension {
        name: "external_echo".into(),
        description: "echo".into(),
        parameters: json!({"type":"object"}),
        url: "http://127.0.0.1:43127/".into(),
    };
    assert!(validate_extension(&tool).is_ok());
    tool.name = "shell".into();
    assert!(validate_extension(&tool).is_err());
    tool.name = "external_echo".into();
    tool.url = "https://example.com:443/".into();
    assert!(validate_extension(&tool).is_err());
    tool.url = "http://127.0.0.1/".into();
    assert!(validate_extension(&tool).is_err());
}

#[tokio::test]
async fn tools_read_write_and_drain_large_shell_output() -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(".patches")
        .join(format!("tools-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let (host, _) = mpsc::channel();
    let tools = Tools {
        root: dir,
        host,
        extensions: Arc::new(RwLock::new(BTreeMap::new())),
    };
    tools
        .execute(
            "write_file",
            json!({"path":"sample.txt", "content":"hello"}),
        )
        .await?;
    anyhow::ensure!(
        tools
            .execute("read_file", json!({"path":"sample.txt"}))
            .await?
            == "hello"
    );
    let output = tools
        .execute(
            "shell",
            json!({"command":"head -c 100000 /dev/zero | tr '\\0' x"}),
        )
        .await?;
    anyhow::ensure!(output.starts_with("exit=exit status: 0"));
    anyhow::ensure!(output.len() < 17000);
    Ok(())
}
