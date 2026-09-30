use super::*;
use ratatui::{Terminal, backend::TestBackend};

#[test]
fn conversation_shows_prompt_tool_result_and_answer() -> Result<()> {
    let (prompts, _) = tokio::sync::mpsc::unbounded_channel();
    let state = UiState {
        lines: vec![
            "User: inspect code".into(),
            "Tool: read_file".into(),
            "Result: hello".into(),
            "Assistant: finished".into(),
        ],
        input: String::new(),
        busy: false,
        scroll: 0,
        prompts,
    };
    let mut terminal = Terminal::new(TestBackend::new(120, 24))?;
    let frame =
        terminal.draw(|frame| render(frame, &state, "native-v1", "http://127.0.0.1:43219/"))?;
    let text: String = frame
        .buffer
        .content
        .iter()
        .map(|cell| cell.symbol())
        .collect();
    for expected in [
        "User: inspect code",
        "Tool: read_file",
        "Result: hello",
        "Assistant: finished",
        "native-v1",
    ] {
        anyhow::ensure!(text.contains(expected));
    }
    Ok(())
}
