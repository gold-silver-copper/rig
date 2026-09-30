use crate::UiState;
use anyhow::Result;
use crossterm::{
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame,
    layout::{Constraint, Layout},
    widgets::{Block, Paragraph, Wrap},
};

pub struct TerminalGuard;
impl TerminalGuard {
    pub fn enter() -> Result<Self> {
        enable_raw_mode()?;
        let guard = Self;
        execute!(std::io::stdout(), EnterAlternateScreen)?;
        Ok(guard)
    }
}
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(std::io::stdout(), LeaveAlternateScreen);
    }
}

pub fn render(frame: &mut Frame, ui: &UiState, label: &str, endpoint: &str) {
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(2),
        Constraint::Length(3),
    ])
    .split(frame.area());
    let heading = format!(
        "{label} | {} | {endpoint} | Ctrl-C quit, PgUp/PgDn scroll",
        if ui.busy { "working" } else { "ready" }
    );
    let log = Paragraph::new(ui.lines.join("\n"))
        .wrap(Wrap { trim: false })
        .block(Block::bordered().title("Conversation"));
    let visible = areas[1].height.saturating_sub(2) as usize;
    let bottom = log.line_count(areas[1].width).saturating_sub(visible);
    let offset = bottom
        .saturating_sub(ui.scroll as usize)
        .min(u16::MAX as usize) as u16;
    frame.render_widget(
        Paragraph::new(heading).block(Block::bordered().title("Rig / Bevy")),
        areas[0],
    );
    frame.render_widget(log.scroll((offset, 0)), areas[1]);
    frame.render_widget(
        Paragraph::new(ui.input.as_str()).block(Block::bordered().title("Prompt (Enter to send)")),
        areas[2],
    );
}

#[cfg(test)]
mod tests;
