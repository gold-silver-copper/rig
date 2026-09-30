//! Sessions an application opens with names chosen at run time: record
//! through the proxy, then replay the fixture with no upstream at all.

use super::*;
use serde_json::json;

fn client() -> rig_reqwest::reqwest::Client {
    rig_reqwest::reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("HTTP client")
}

async fn post(base_url: &str) -> Value {
    client()
        .post(format!("{base_url}/chat/completions"))
        .bearer_auth("sk-test")
        .json(&json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("exchange")
        .json()
        .await
        .expect("JSON reply")
}

#[tokio::test]
async fn a_session_named_at_run_time_records_and_replays() {
    let upstream = httpmock::MockServer::start_async().await;
    upstream
        .mock_async(|when, then| {
            when.method("POST").path("/v1/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"id": "chat-1", "choices": []}));
        })
        .await;
    let dir = assert_fs::TempDir::new().expect("fixture directory");
    // Names an application would only know at run time.
    let (provider, scenario) = (String::from("example"), format!("session-{}", 7));
    let fixture = cassette_path(dir.path(), &provider, &scenario);

    let recording = ProviderCassette::start_named(
        RecordVia::Proxy,
        &provider,
        &scenario,
        &format!("{}/v1", upstream.base_url()),
        CassetteMode::Record,
        fixture.clone(),
    )
    .await;
    let recorded = post(&recording.base_url()).await;
    recording.finish().await;
    assert!(fixture.exists());

    let replay = ProviderCassette::start_named(
        RecordVia::Proxy,
        &provider,
        &scenario,
        "https://unreachable.invalid/v1",
        CassetteMode::Replay,
        fixture,
    )
    .await;
    assert_eq!(post(&replay.base_url()).await, recorded);
    replay.finish().await;
}
