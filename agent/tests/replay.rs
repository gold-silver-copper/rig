//! Every eval in `evals/` replays its cassettes offline: the same tool calls
//! and final answers as the live session it was recorded from, with no API
//! keys and no network. CI runs this.

#[test]
fn every_eval_replays_offline() {
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"] {
        // SAFETY: set before this test starts any thread.
        unsafe { std::env::remove_var(key) };
    }
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("evals");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .expect("evals directory")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            (path.extension()? == "json").then(|| path.file_stem()?.to_str().map(str::to_owned))?
        })
        .collect();
    names.sort();
    assert!(names.len() >= 4, "one tool-calling eval per provider at least: {names:?}");

    let state = std::env::temp_dir().join(format!("rig-pi-replay-test-{}", std::process::id()));
    let results = rig_pi::plugins::evals::run_scripts(&names, &state);
    let _ = std::fs::remove_dir_all(&state);

    let failed: Vec<_> = results.iter().filter(|(_, failures)| !failures.is_empty()).collect();
    assert!(failed.is_empty(), "{failed:#?}");
    assert_eq!(results.len(), names.len());
}
