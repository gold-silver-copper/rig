//! Replays the recorded provider suite: every provider completes a turn
//! with a tool call, with no network and no API keys.

use std::process::Command;

#[test]
fn every_provider_replays_a_tool_calling_turn() {
    let output = Command::new(env!("CARGO_BIN_EXE_bevy-agent"))
        .args([
            "eval",
            concat!(env!("CARGO_MANIFEST_DIR"), "/evals/providers.json"),
        ])
        .env_remove("OPENAI_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("GEMINI_API_KEY")
        .env_remove("DEEPSEEK_API_KEY")
        .output()
        .expect("the agent runs");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("4/4 cases passed"), "{stdout}");
}
