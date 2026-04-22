//! DeepSeek extractor smoke test.

use anyhow::{Result, anyhow};
use rig::client::{CompletionClient, ProviderClient};
use rig::providers::deepseek;

use crate::support::{EXTRACTOR_TEXT, SmokePerson, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn extractor_smoke() -> Result<()> {
    let client = deepseek::Client::from_env()?;
    let extractor = client
        .extractor::<SmokePerson>(deepseek::DEEPSEEK_CHAT)
        .build();

    let person = extractor.extract(EXTRACTOR_TEXT).await?;

    let Some(first_name) = person.first_name.as_deref() else {
        return Err(anyhow!("first_name should be present"));
    };
    let Some(last_name) = person.last_name.as_deref() else {
        return Err(anyhow!("last_name should be present"));
    };
    let Some(job) = person.job.as_deref() else {
        return Err(anyhow!("job should be present"));
    };

    assert_nonempty_response(first_name);
    assert_nonempty_response(last_name);
    assert_nonempty_response(job);
    Ok(())
}
