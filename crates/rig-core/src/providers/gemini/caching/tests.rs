use super::*;

fn parsed(body: serde_json::Value) -> Parsed {
    parse(&serde_json::to_vec(&body).expect("json")).expect("parses")
}

fn fixed_clock() -> u64 {
    1_000_000
}

fn body(turns: usize) -> serde_json::Value {
    let mut contents = Vec::new();
    for i in 0..turns {
        contents.push(serde_json::json!({"role": "user", "parts": [{"text": format!("question {i} {}", "x".repeat(4000))}]}));
        contents.push(serde_json::json!({"role": "model", "parts": [{"text": format!("answer {i}"), "thoughtSignature": "c2ln".repeat(500)}]}));
    }
    contents.push(serde_json::json!({"role": "user", "parts": [{"text": "next"}]}));
    serde_json::json!({
        "systemInstruction": {"parts": [{"text": "be brief ".repeat(600)}]},
        "tools": [{"functionDeclarations": [{"name": "lookup", "description": "d"}]}],
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 64}
    })
}

#[test]
fn a_request_that_extends_another_shares_its_digests() {
    let short = parsed(body(2));
    let long = parsed(body(3));
    let a = digests("m", &short);
    let b = digests("m", &long);
    assert_eq!(a[..4], b[..4], "the shared contents digest the same");
    assert_ne!(digests("other", &short)[0], a[0], "the model is part of the identity");
}

#[test]
fn stripping_a_lease_keeps_the_uncovered_contents_byte_for_byte() {
    let p = parsed(body(2));
    let lease = Lease { name: "cachedContents/x".into(), digest: "d".into(), covers: 3, tokens: 1, expires_at: u64::MAX };
    let out: serde_json::Value = serde_json::from_slice(&stripped(&p, &lease).expect("strips")).expect("json");
    assert_eq!(out["cachedContent"], "cachedContents/x");
    assert!(out.get("systemInstruction").is_none() && out.get("tools").is_none());
    assert_eq!(out["contents"].as_array().expect("contents").len(), 2);
    assert_eq!(out["generationConfig"]["maxOutputTokens"], 64);
}

#[test]
fn signatures_do_not_count_as_cacheable() {
    let with = visible_estimate(r#"{"parts":[{"text":"hi","thoughtSignature":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}]}"#);
    let without = visible_estimate(r#"{"parts":[{"text":"hi"}]}"#);
    assert_eq!(with, without);
}

#[test]
fn a_fresh_conversation_does_not_roll_until_the_premium_pays_for_the_cache() {
    let book = CacheBook::with_clock(AutoCache::default(), fixed_clock);
    let p = parsed(body(2));
    let d = digests("m", &p);
    let plan = book.plan(&d, &p);
    assert!(plan.create.is_none(), "nothing paid yet");
    // Two calls that missed everything they could have cached.
    book.observe(&plan.line, false, plan.coverable, 0);
    book.observe(&plan.line, false, plan.coverable, 0);
    let plan = book.plan(&d, &p);
    assert_eq!(plan.create, Some(p.contents.len() - 1), "rolls to everything but the newest content");
}

#[test]
fn implicit_hits_keep_a_conversation_inline() {
    let book = CacheBook::with_clock(AutoCache::default(), fixed_clock);
    let p = parsed(body(4));
    let d = digests("m", &p);
    let plan = book.plan(&d, &p);
    assert!(plan.coverable >= 4096, "big enough for implicit caching: {}", plan.coverable);
    for _ in 0..5 {
        book.observe(&plan.line, false, plan.coverable, plan.coverable * 9 / 10);
    }
    assert!(book.plan(&d, &p).create.is_none(), "implicit is serving");
}

#[test]
fn a_prefix_two_conversations_share_is_cached_at_once() {
    let book = CacheBook::with_clock(AutoCache::default(), fixed_clock);
    let mut a = body(0);
    a["contents"] = serde_json::json!([{"role": "user", "parts": [{"text": "agent a"}]}]);
    let mut b = a.clone();
    b["contents"] = serde_json::json!([{"role": "user", "parts": [{"text": "agent b"}]}]);
    let (a, b) = (parsed(a), parsed(b));
    assert!(book.plan(&digests("m", &a), &a).create.is_none());
    assert_eq!(book.plan(&digests("m", &b), &b).create, Some(0), "the shared prefix");
}

#[test]
fn a_lost_lease_is_forgotten() {
    let book = CacheBook::with_clock(AutoCache::default(), fixed_clock);
    let p = parsed(body(2));
    let d = digests("m", &p);
    let lease = Lease { name: "cachedContents/y".into(), digest: d[3].clone(), covers: 3, tokens: 5000, expires_at: u64::MAX };
    book.restore([lease.clone()]);
    assert_eq!(book.plan(&d, &p).use_lease.as_ref().map(|l| l.covers), Some(3));
    book.forget(&lease);
    assert!(book.plan(&d, &p).use_lease.is_none());
    assert!(matches!(book.events().last(), Some(CacheEvent::Lost { .. })));
}

#[test]
fn past_the_ceiling_the_book_goes_inline_until_implicit_fails() {
    let book = CacheBook::with_clock(AutoCache { implicit_ceiling: 2_000, ..AutoCache::default() }, fixed_clock);
    let p = parsed(body(4));
    let d = digests("m", &p);
    let lease = Lease { name: "cachedContents/z".into(), digest: d[3].clone(), covers: 3, tokens: 3000, expires_at: u64::MAX };
    book.restore([lease]);
    let plan = book.plan(&d, &p);
    assert!(plan.use_lease.is_none() && plan.create.is_none(), "inline past the ceiling");
    for _ in 0..3 {
        book.observe(&plan.line, false, plan.coverable, 0);
    }
    assert!(book.plan(&d, &p).use_lease.is_some(), "implicit failed: back to the cache");
}
