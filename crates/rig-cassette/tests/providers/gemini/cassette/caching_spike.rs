//! SPIKE (not for merge): the #2625 support-agent scenario through the
//! automatic caching transport (`gemini::caching`). Records under
//! `RIG_SPIKE_CASSETTE_ROOT` (default `/tmp/rig-gemini-caching/cassettes`),
//! never under `fixtures/`.

use std::path::PathBuf;

use rig::AgentBuilder;
use rig::cassette::gemini::Exchanges;
use rig::cassette::http::{CassetteSpec, ProviderCassette, cassette_path};
use rig::providers::gemini::caching::{AutoCache, CacheBook};
use rig::providers::gemini::{self, GeminiConfig};
use rig::tool::PortableTool;
use serde::Deserialize;
use serde_json::{Value, json};

const SUPPORT_PREAMBLE: &str = include_str!("support_preamble.md");
const STATUSES: [&str; 6] = ["refunded", "shipped", "delivered", "processing", "cancelled", "returned"];

#[derive(Deserialize)]
struct OrderArgs {
    order_id: String,
}

struct LookupOrder;

impl PortableTool for LookupOrder {
    const NAME: &'static str = "lookup_order";
    type Args = OrderArgs;
    type Output = Value;
    type Error = std::convert::Infallible;

    fn description(&self) -> String {
        "Status of an order by id.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": { "order_id": { "type": "string" } }, "required": ["order_id"] })
    }

    async fn call(&self, args: OrderArgs) -> Result<Value, Self::Error> {
        let i = args.order_id.bytes().map(usize::from).sum::<usize>() % STATUSES.len();
        Ok(json!({ "order_id": args.order_id, "status": STATUSES[i], "date": format!("2026-05-{:02}", (i * 5) % 27 + 1) }))
    }
}

#[tokio::test]
async fn support_agent_on_automatic_caching() {
    const SCENARIO: &str = "caching_spike/support_agent_auto";
    let root = PathBuf::from(
        std::env::var("RIG_SPIKE_CASSETTE_ROOT").unwrap_or_else(|_| "/tmp/rig-gemini-caching/cassettes".into()),
    );
    let cassette = ProviderCassette::start(&root, "gemini", CassetteSpec::new(SCENARIO), gemini::BASE_URL).await;
    let client = GeminiConfig::new(cassette.api_key(gemini::API_KEY_ENV))
        .with_base_url(cassette.base_url())
        .client();
    let book = CacheBook::new(AutoCache::default());
    let agent = AgentBuilder::new(client.completion_cached(gemini::GEMINI_3_8_FLASH, book.clone()))
        .preamble(SUPPORT_PREAMBLE)
        .tool(LookupOrder)
        .default_max_turns(4)
        .build();
    let mut history = Vec::new();
    for order in 1..=30 {
        agent
            .chat(format!("What happened to order A-{order}?"), &mut history)
            .await
            .expect("turn succeeds");
    }
    let caches = client.cached_contents();
    for lease in book.leases() {
        let _ = caches.delete(&lease.name).await;
    }
    cassette.finish().await;

    let exchanges = Exchanges::from_cassette(&cassette_path(&root, "gemini", SCENARIO)).expect("parses");
    let (mut input, mut cached) = (0u64, 0u64);
    for (turn, exchange) in exchanges.turns().iter().enumerate() {
        let usage = exchange.usage().expect("usage");
        input += usage.input_tokens.unwrap_or(0);
        cached += usage.cached_input_tokens.unwrap_or(0);
        println!(
            "turn {turn}: input {:?} cached {:?} cache {:?} contents {}",
            usage.input_tokens,
            usage.cached_input_tokens,
            exchange.request.cached_content,
            exchange.request.contents.len()
        );
        if exchange.request.cached_content.is_some() {
            assert!(
                exchange.request.system_instruction.is_none() && exchange.request.tools.is_empty(),
                "turn {turn} re-sent what its cache holds"
            );
        }
        assert!(usage.cached_input_tokens.unwrap_or(0) <= usage.input_tokens.unwrap_or(0));
    }
    println!("whole run: input {input}, cached {cached} ({:.0}%)", 100.0 * cached as f64 / input.max(1) as f64);
    println!("cache events: {:?}", book.events());
    assert!(cached * 2 > input, "the run is mostly cached");
}
