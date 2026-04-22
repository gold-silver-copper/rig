//! llama.cpp model listing smoke test.

use anyhow::Result;
use rig::client::ModelListingClient;

use super::support;

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn list_models_smoke() -> Result<()> {
    let client = support::client()?;
    let models = client.list_models().await?;

    assert!(
        !models.is_empty(),
        "expected llama.cpp to return at least one model\nModel list: {models:#?}"
    );
    Ok(())
}
