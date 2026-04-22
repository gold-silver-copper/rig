//! ChatGPT extractor smoke test.

use anyhow::{Result, anyhow};
use rig::client::CompletionClient;

use crate::chatgpt::{LIVE_MODEL, live_client};
use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn extractor_smoke() -> Result<()> {
    let extractor = live_client()?.extractor::<SmokePerson>(LIVE_MODEL).build();

    let response = extractor.extract_with_usage(EXTRACTOR_TEXT).await?;

    let first_name = response
        .data
        .first_name
        .as_deref()
        .ok_or_else(|| anyhow!("first_name should be present"))?;
    let last_name = response
        .data
        .last_name
        .as_deref()
        .ok_or_else(|| anyhow!("last_name should be present"))?;
    let job = response
        .data
        .job
        .as_deref()
        .ok_or_else(|| anyhow!("job should be present"))?;

    assert_nonempty_response(first_name);
    assert_nonempty_response(last_name);
    assert_nonempty_response(job);
    assert!(response.usage.total_tokens > 0, "usage should be populated");
    Ok(())
}
