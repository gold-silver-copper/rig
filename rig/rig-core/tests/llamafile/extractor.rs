//! Llamafile extractor smoke test.

use anyhow::{Result, anyhow};
use rig::client::CompletionClient;

use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

use super::support;

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn extractor_smoke() -> Result<()> {
    if support::skip_if_server_unavailable() {
        return Ok(());
    }

    let client = support::client()?;
    let extractor = client
        .extractor::<SmokePerson>(support::model_name())
        .build();

    let response = extractor.extract_with_usage(EXTRACTOR_TEXT).await?;

    let Some(first_name) = response.data.first_name.as_deref() else {
        return Err(anyhow!("first_name should be present"));
    };
    let Some(last_name) = response.data.last_name.as_deref() else {
        return Err(anyhow!("last_name should be present"));
    };
    let Some(job) = response.data.job.as_deref() else {
        return Err(anyhow!("job should be present"));
    };

    assert_nonempty_response(first_name);
    assert_nonempty_response(last_name);
    assert_nonempty_response(job);
    assert!(response.usage.total_tokens > 0, "usage should be populated");
    Ok(())
}
