use super::*;

#[test]
fn native_plugin_preserves_resource_between_frames() {
    let mut app = App::new();
    app.add_plugins(sprout_native::NativePlugin);
    app.update();
    app.update();
    assert_eq!(
        app.world().resource::<sprout_native::NativeState>().ticks,
        2
    );
    assert_eq!(
        app.world().resource::<sprout_native::NativeState>().label,
        "native: seedling"
    );
}

#[test]
fn tui_displays_prompt_tool_and_final_answer() -> Result<()> {
    let (prompts, _) = crossbeam_channel::unbounded();
    let (_, events) = crossbeam_channel::unbounded();
    let agent = Agent {
        lines: vec![
            "you: compute".into(),
            "tool shell: printf 42".into(),
            "result shell: 42".into(),
            "assistant: 42".into(),
        ],
        input: "next λ".into(),
        busy: false,
        prompts,
        events,
    };
    let native = sprout_native::NativeState {
        ticks: 5,
        label: "test plugin".into(),
    };
    let mut terminal = ratatui::Terminal::new(ratatui::backend::TestBackend::new(100, 24))?;
    terminal.draw(|frame| draw(frame, &agent, &native, 12345))?;
    let screen = terminal
        .backend()
        .buffer()
        .content
        .iter()
        .map(|c| c.symbol())
        .collect::<String>();
    for expected in [
        "you: compute",
        "tool shell",
        "assistant: 42",
        "test plugin",
        "next λ",
    ] {
        assert!(screen.contains(expected), "missing {expected}");
    }
    Ok(())
}
