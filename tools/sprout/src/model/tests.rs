use super::*;

#[test]
fn shell_reports_failure_and_bounds_output() -> Result<()> {
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        let cwd = std::env::current_dir()?;
        let output = shell_call(
            &json!({"command":"printf hello; printf error >&2; exit 7"}),
            &cwd,
        )
        .await?;
        assert!(output.contains("7"));
        assert!(output.contains("helloerror"));
        let output = shell_call(&json!({"command":"yes x | head -c 100000"}), &cwd).await?;
        assert!(output.len() < 17000);
        assert!(shell_call(&json!({}), &cwd).await.is_err());
        Ok(())
    })
}
