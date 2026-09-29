//! SPIKE (not for merge): automatic explicit caching for GenerateContent, as a
//! transport that wraps another.
//!
//! [`Caching`] sits between the GenerateContent wire and the HTTP transport.
//! Every request passes through it with its encoded body. It keeps a shared,
//! content-addressed [`CacheBook`] of live caches keyed by a chained SHA-256
//! over the rendered prefix (model, system instruction, tools, tool config,
//! then each content's exact bytes). For a request it:
//!
//! 1. finds the longest live cache whose digest the request starts with and
//!    strips what that cache holds (system instruction, tools, tool config and
//!    the covered contents), naming the cache in `cachedContent`;
//! 2. decides, by ski rental, whether to roll: create a cache of everything
//!    but the newest content once the input premium already paid for
//!    cacheable-but-uncached tokens covers the new cache's creation and
//!    storage. Implicit hits reduce the premium, so a conversation that
//!    implicit caching already serves never rolls;
//! 3. caches a prefix shared by two conversations right away (amortized);
//! 4. on a 403 for a cache it named (expired, deleted), forgets the cache and
//!    sends the original request inline, once;
//! 5. reads each reply's final usage to update the premium.
//!
//! Nothing here reads per-run state: a request's cache is a pure function of
//! its bytes and the book, so resumed runs, sub-agents and new processes find
//! each other's caches.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};

use super::api;
use super::cached_content::{CachedContentRequest, CachedContents};
use super::completion::GenerateContent;
use crate::driver::{Exchange, Model, Opening, Transport};
use crate::error::ProviderError;
use crate::wire::{Body, Encoded, WireFrame};

/// Seconds since the Unix epoch. Injected so replay is deterministic and
/// wasm32 never touches `SystemTime`.
pub type Clock = fn() -> u64;

/// The system clock.
pub fn system_clock() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

/// What the book optimizes for. Prices enter only as ratios to the input
/// price, so no model name is consulted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoCache {
    /// Lifetime of each cache the book creates.
    pub ttl_secs: u64,
    /// Cached-read price / input price (0.1 for Gemini 2.5 and later).
    pub cached_ratio: f64,
    /// Storage price per hour / input price (0.5 / 0.75 for gemini-3.8-flash).
    pub storage_ratio_per_hour: f64,
    /// Gemini's minimum cache size.
    pub min_tokens: u64,
    /// Smallest prompt implicit caching serves (4,096 on 3.x).
    pub implicit_floor: u64,
    /// Above this many cacheable tokens the book stops rolling and lets
    /// implicit caching serve, unless it has been seen failing (16,000:
    /// grown 3.8-flash prompts hit implicitly from about 16–24k tokens).
    pub implicit_ceiling: u64,
}

impl Default for AutoCache {
    fn default() -> Self {
        Self {
            ttl_secs: 3600,
            cached_ratio: 0.1,
            storage_ratio_per_hour: 0.5 / 0.75,
            min_tokens: 1024,
            implicit_floor: 4096,
            implicit_ceiling: 16_000,
        }
    }
}

/// One cache the book holds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lease {
    /// `cachedContents/<id>`.
    pub name: String,
    /// Hex digest of the prefix the cache holds; also the tail of its display name.
    pub digest: String,
    /// How many contents the cache holds.
    pub covers: usize,
    /// Its size, as Gemini counted it at creation.
    pub tokens: u64,
    /// Unix seconds.
    pub expires_at: u64,
}

/// What happened to caches, for accounting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CacheEvent {
    /// A cache was created: its size and when.
    Created { name: String, tokens: u64, at: u64 },
    /// A cache the book named answered 403 and was forgotten.
    Lost { name: String, at: u64 },
}

#[derive(Default)]
struct Line {
    /// Cacheable input tokens paid at full price since the last roll.
    premium: f64,
    /// EWMA of implicit coverage on inline calls big enough for it.
    implicit: Option<f64>,
    rolled_at: u64,
}

#[derive(Default)]
struct Book {
    leases: HashMap<String, Lease>,
    lineages: HashMap<String, HashSet<String>>,
    lines: HashMap<String, Line>,
    events: Vec<CacheEvent>,
}

/// The shared cache book. Clone it into every model that should share caches.
#[derive(Clone)]
pub struct CacheBook {
    inner: Arc<Mutex<Book>>,
    policy: AutoCache,
    clock: Clock,
}

impl CacheBook {
    /// A book with `policy` on the system clock.
    pub fn new(policy: AutoCache) -> Self {
        Self::with_clock(policy, system_clock)
    }

    /// A book on `clock`.
    pub fn with_clock(policy: AutoCache, clock: Clock) -> Self {
        Self {
            inner: Arc::default(),
            policy,
            clock,
        }
    }

    /// Every live lease, for a checkpoint.
    pub fn leases(&self) -> Vec<Lease> {
        let now = (self.clock)();
        let book = self.inner.lock().expect("book");
        book.leases.values().filter(|l| l.expires_at > now).cloned().collect()
    }

    /// Put checkpointed leases back. Callers prove them first (GET) or let a
    /// 403 drop them on first use.
    pub fn restore(&self, leases: impl IntoIterator<Item = Lease>) {
        let mut book = self.inner.lock().expect("book");
        for lease in leases {
            book.leases.insert(lease.digest.clone(), lease);
        }
    }

    /// Cache events so far.
    pub fn events(&self) -> Vec<CacheEvent> {
        self.inner.lock().expect("book").events.clone()
    }
}

/// A transport that caches GenerateContent requests through `inner`.
#[derive(Clone)]
pub struct Caching<T> {
    inner: T,
    config: super::GeminiConfig,
    book: CacheBook,
}

impl<T> Caching<T> {
    /// Cache through `inner` with `book`.
    pub fn new(inner: T, config: super::GeminiConfig, book: CacheBook) -> Self {
        Self { inner, config, book }
    }
}

/// The request body, contents kept as their exact bytes.
struct Parsed {
    top: serde_json::Map<String, serde_json::Value>,
    prefix: [Option<Box<RawValue>>; 3],
    contents: Vec<Box<RawValue>>,
}

const PREFIX_KEYS: [&str; 3] = ["systemInstruction", "tools", "toolConfig"];

fn parse(bytes: &[u8]) -> Option<Parsed> {
    let raw: HashMap<String, Box<RawValue>> = serde_json::from_slice(bytes).ok()?;
    let mut top = serde_json::Map::new();
    let mut prefix: [Option<Box<RawValue>>; 3] = [None, None, None];
    let mut contents = Vec::new();
    for (key, value) in raw {
        if key == "contents" {
            contents = serde_json::from_str(value.get()).ok()?;
        } else if let Some(i) = PREFIX_KEYS.iter().position(|k| *k == key) {
            prefix[i] = Some(value);
        } else {
            top.insert(key, serde_json::from_str(value.get()).ok()?);
        }
    }
    Some(Parsed { top, prefix, contents })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// d[k]: the digest of model + prefix + contents[..k].
fn digests(model: &str, parsed: &Parsed) -> Vec<String> {
    let mut hasher = Sha256::new();
    hasher.update(model.as_bytes());
    for part in &parsed.prefix {
        hasher.update([0u8]);
        if let Some(raw) = part {
            hasher.update(raw.get().as_bytes());
        }
    }
    let mut state = hasher.finalize().to_vec();
    let mut out = vec![hex(&state)];
    for content in &parsed.contents {
        let mut h = Sha256::new();
        h.update(&state);
        h.update(content.get().as_bytes());
        state = h.finalize().to_vec();
        out.push(hex(&state));
    }
    out
}

/// Visible tokens of a JSON fragment: bytes / 4 without thought signatures,
/// which Gemini expands into restored thoughts that no explicit cache holds.
fn visible_estimate(json: &str) -> u64 {
    let Ok(mut value) = serde_json::from_str::<serde_json::Value>(json) else {
        return (json.len() / 4) as u64;
    };
    fn scrub(v: &mut serde_json::Value) {
        match v {
            serde_json::Value::Object(map) => {
                map.remove("thoughtSignature");
                for (_, child) in map.iter_mut() {
                    scrub(child);
                }
            }
            serde_json::Value::Array(items) => items.iter_mut().for_each(scrub),
            _ => {}
        }
    }
    scrub(&mut value);
    (value.to_string().len() / 4) as u64
}

fn model_of(path: &str) -> Option<String> {
    let rest = path.split("/models/").nth(1)?;
    Some(rest.split(':').next()?.to_owned())
}

/// The plan for one request.
struct Plan {
    use_lease: Option<Lease>,
    create: Option<usize>,
    line: String,
    coverable: u64,
}

impl CacheBook {
    fn plan(&self, d: &[String], parsed: &Parsed) -> Plan {
        let p = self.policy;
        let now = (self.clock)();
        let n = parsed.contents.len();
        let prefix_est: u64 = parsed
            .prefix
            .iter()
            .flatten()
            .map(|r| visible_estimate(r.get()))
            .sum();
        let mut book = self.inner.lock().expect("book");
        if n >= 1 {
            book.lineages
                .entry(d[0].clone())
                .or_default()
                .insert(d[1].clone());
        }
        let shared = book.lineages.get(&d[0]).is_some_and(|s| s.len() >= 2);
        // Longest live lease the request starts with, never the whole request.
        let lease = (0..n.max(1))
            .rev()
            .find_map(|k| book.leases.get(&d[k]).filter(|l| l.expires_at > now + 5).cloned());
        let covered = lease.as_ref().map_or(0, |l| l.covers);
        let tail: u64 = parsed.contents[covered..n.saturating_sub(1).max(covered)]
            .iter()
            .map(|c| visible_estimate(c.get()))
            .sum();
        let coverable = lease.as_ref().map_or(prefix_est, |l| l.tokens) + tail;
        let line_key = lease.as_ref().map_or_else(|| d[1.min(n)].clone(), |l| l.digest.clone());
        let prefix_leased = book.leases.contains_key(&d[0]);
        let line = book.lines.entry(line_key.clone()).or_insert_with(|| Line {
            rolled_at: now,
            ..Line::default()
        });
        let mut create = None;
        let implicit_serving = line.implicit.is_none_or(|r| r >= 0.5);
        if coverable >= p.implicit_ceiling && implicit_serving {
            // Big enough for implicit caching: send inline and let it serve, or prove it fails.
            return Plan { use_lease: None, create: None, line: line_key, coverable };
        }
        if lease.is_none() && shared && prefix_est >= p.min_tokens && !prefix_leased {
            create = Some(0);
        } else if n >= 2 {
            let cached = lease.as_ref().map_or(0, |l| l.tokens);
            let gain = coverable.saturating_sub(cached);
            let held_h = (now.saturating_sub(line.rolled_at)).max(60) as f64 / 3600.0;
            let cost = coverable as f64 * (1.0 + p.storage_ratio_per_hour * held_h);
            if gain >= 256
                && coverable >= p.min_tokens
                && line.premium * (1.0 - p.cached_ratio) >= cost
            {
                create = Some(n - 1);
            }
        }
        Plan {
            use_lease: lease,
            create,
            line: line_key,
            coverable,
        }
    }

    fn observe(&self, line: &str, used_cache: bool, coverable: u64, cached: u64) {
        let p = self.policy;
        let mut book = self.inner.lock().expect("book");
        let entry = book.lines.entry(line.to_owned()).or_default();
        let mut missed = coverable.saturating_sub(cached) as f64;
        if !used_cache && coverable >= p.implicit_floor {
            let ratio = cached as f64 / coverable.max(1) as f64;
            let implicit = entry.implicit.map_or(ratio, |e| 0.5 * e + 0.5 * ratio);
            entry.implicit = Some(implicit);
            if implicit >= 0.5 {
                missed = 0.0;
            }
        }
        entry.premium += missed;
    }

    fn insert(&self, lease: Lease, from_line: &str) {
        let now = (self.clock)();
        let mut book = self.inner.lock().expect("book");
        let implicit = book.lines.get(from_line).and_then(|l| l.implicit);
        book.lines.insert(
            lease.digest.clone(),
            Line {
                premium: 0.0,
                implicit,
                rolled_at: now,
            },
        );
        book.events.push(CacheEvent::Created {
            name: lease.name.clone(),
            tokens: lease.tokens,
            at: now,
        });
        book.leases.insert(lease.digest.clone(), lease);
    }

    fn forget(&self, lease: &Lease) {
        let now = (self.clock)();
        let mut book = self.inner.lock().expect("book");
        book.leases.remove(&lease.digest);
        book.events.push(CacheEvent::Lost {
            name: lease.name.clone(),
            at: now,
        });
    }
}

fn body_bytes(payload: &Encoded) -> Option<Vec<u8>> {
    match payload.request.body() {
        Body::Bytes(bytes) => Some(bytes.clone()),
        Body::Multipart(_) => None,
    }
}

/// `payload` with `bytes` as its body.
fn with_body(payload: &Encoded, bytes: Vec<u8>) -> Result<Encoded, ProviderError> {
    let mut builder = http::Request::builder()
        .method(payload.request.method().clone())
        .uri(payload.request.uri().clone());
    for (name, value) in payload.request.headers() {
        builder = builder.header(name, value);
    }
    let request = builder
        .body(Body::Bytes(bytes))
        .map_err(|e| ProviderError::request(e.to_string()))?;
    Ok(Encoded {
        request,
        framing: payload.framing,
        request_id_header: payload.request_id_header,
        relaxed_content_type: payload.relaxed_content_type,
        route: payload.route,
        project: payload.project,
        analysis_only: payload.analysis_only,
    })
}

/// The body that reads `lease` instead of what it holds.
fn stripped(parsed: &Parsed, lease: &Lease) -> Result<Vec<u8>, ProviderError> {
    #[derive(Serialize)]
    struct Out<'a> {
        #[serde(rename = "cachedContent")]
        cached_content: &'a str,
        contents: &'a [Box<RawValue>],
        #[serde(flatten)]
        top: &'a serde_json::Map<String, serde_json::Value>,
    }
    serde_json::to_vec(&Out {
        cached_content: &lease.name,
        contents: &parsed.contents[lease.covers..],
        top: &parsed.top,
    })
    .map_err(|e| ProviderError::request(e.to_string()))
}

fn cache_request(
    model: &str,
    parsed: &Parsed,
    covers: usize,
    digest: &str,
    ttl_secs: u64,
) -> Result<api::CachedContent, ProviderError> {
    fn from_raw<T: serde::de::DeserializeOwned>(raw: &RawValue) -> Result<T, ProviderError> {
        serde_json::from_str(raw.get()).map_err(|e| ProviderError::request(e.to_string()))
    }
    Ok(api::CachedContent {
        model: Some(format!("models/{model}")),
        display_name: Some(format!("rig-caching-spike-rust-{}", &digest[..40])),
        system_instruction: parsed.prefix[0].as_deref().map(from_raw).transpose()?,
        tools: parsed.prefix[1].as_deref().map(from_raw).transpose()?.unwrap_or_default(),
        tool_config: parsed.prefix[2].as_deref().map(from_raw).transpose()?,
        contents: parsed.contents[..covers]
            .iter()
            .map(|c| from_raw(c))
            .collect::<Result<_, _>>()?,
        ttl: Some(format!("{ttl_secs}s")),
        ..Default::default()
    })
}

/// The last `usageMetadata` in a frame: `(promptTokenCount + toolUse, cachedContentTokenCount)`.
fn usage_of(frame: &WireFrame) -> Option<(u64, u64)> {
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Usage {
        #[serde(default)]
        cached_content_token_count: u64,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Reply {
        usage_metadata: Option<Usage>,
        #[serde(default)]
        candidates: Vec<serde_json::Value>,
    }
    let text = frame.as_str();
    let reply: Reply = serde_json::from_str(&text).ok()?;
    let finished = reply
        .candidates
        .iter()
        .any(|c| c.get("finishReason").is_some());
    let usage = reply.usage_metadata?;
    finished.then_some((0, usage.cached_content_token_count))
}

impl<T> Transport<GenerateContent> for Caching<T>
where
    T: Transport<GenerateContent> + Transport<CachedContents>,
{
    fn send(&self, payload: Encoded, exchange: Exchange) -> Opening<WireFrame> {
        let this = self.clone();
        Opening::new(async move {
            let Exchange { mode, observation } = exchange;
            let path = payload.request.uri().path().to_owned();
            let (Some(bytes), Some(model)) = (body_bytes(&payload), model_of(&path)) else {
                return Transport::<GenerateContent>::send(&this.inner, payload, Exchange { mode, observation }).await;
            };
            let Some(parsed) = parse(&bytes).filter(|p| !p.top.contains_key("cachedContent")) else {
                return Transport::<GenerateContent>::send(&this.inner, payload, Exchange { mode, observation }).await;
            };
            let d = digests(&model, &parsed);
            let plan = this.book.plan(&d, &parsed);
            let mut lease = plan.use_lease.clone();
            let mut line = plan.line.clone();
            if let Some(covers) = plan.create {
                let request = cache_request(&model, &parsed, covers, &d[covers], this.book.policy.ttl_secs)?;
                let caches = Model::new(this.config.cached_contents(), this.inner.clone());
                // A failed creation leaves the request as it was.
                if let Ok(reply) = caches.call(CachedContentRequest::Create(Box::new(request))).await {
                    if let Ok(resource) = reply.resource() {
                        let now = (this.book.clock)();
                        let new = Lease {
                            name: resource.name.clone().unwrap_or_default(),
                            digest: d[covers].clone(),
                            covers,
                            tokens: resource
                                .usage_metadata
                                .as_ref()
                                .and_then(|u| u.total_token_count)
                                .map_or(0, |t| t as u64),
                            expires_at: now + this.book.policy.ttl_secs,
                        };
                        this.book.insert(new.clone(), &line);
                        line = new.digest.clone();
                        lease = Some(new);
                    }
                }
            }
            let coverable = plan.coverable;
            let sent = match &lease {
                Some(lease) => with_body(&payload, stripped(&parsed, lease)?)?,
                None => with_body(&payload, bytes.clone())?,
            };
            let mut opened = Transport::<GenerateContent>::send(
                &this.inner,
                sent,
                Exchange { mode, observation: observation.clone() },
            )
            .await?;
            if opened.status == Some(http::StatusCode::FORBIDDEN) {
                if let Some(lost) = lease.take() {
                    // Expired or deleted: forget it and send the original inline, once.
                    this.book.forget(&lost);
                    opened = Transport::<GenerateContent>::send(
                        &this.inner,
                        with_body(&payload, bytes)?,
                        Exchange { mode, observation },
                    )
                    .await?;
                }
            }
            let book = this.book.clone();
            let used = lease.is_some();
            Ok(opened.map_frames(move |frames| {
                frames.inspect(move |frame| {
                    if let Ok(frame) = frame {
                        if let Some((_, cached)) = usage_of(frame) {
                            book.observe(&line, used, coverable, cached);
                        }
                    }
                })
            }))
        })
    }
}

#[cfg(test)]
mod tests;
