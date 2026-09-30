//! The terminal UI: a transcript, the prompt queue and an input line.

use std::time::Duration;

use bevy::prelude::*;
use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Position};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Wrap};

use crate::agent::{
    ActiveModel, Config, MODEL_ALIASES, PromptQueue, SwitchModel, Transcript, Turn,
    save_session_command,
};
use crate::reload::ReloadCommand;
use crate::session::EntryKind;

pub struct TuiPlugin;

struct Term(DefaultTerminal);

#[derive(Resource, Default)]
struct Input {
    text: String,
    /// Cursor position, in chars.
    cursor: usize,
    /// Lines scrolled up from the bottom of the transcript.
    scroll: usize,
    history: Vec<String>,
    browsing: Option<usize>,
}

impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        app.insert_non_send(Term(ratatui::init()))
            .init_resource::<Input>()
            .add_systems(PreUpdate, read_input)
            .add_systems(PostUpdate, draw);
    }
}

pub fn restore_terminal() {
    ratatui::restore();
}

const HELP: &str = "Enter sends (queued while busy) · /model [alias|vendor:model] · /reload · \
                    /quit · PgUp/PgDn scroll · ↑/↓ history";

#[allow(clippy::too_many_arguments)]
fn read_input(
    mut input: ResMut<Input>,
    mut queue: ResMut<PromptQueue>,
    mut transcript: ResMut<Transcript>,
    active: Res<ActiveModel>,
    config: Res<Config>,
    mut switch: MessageWriter<SwitchModel>,
    mut reload: MessageWriter<ReloadCommand>,
    mut exit: MessageWriter<AppExit>,
    mut commands: Commands,
) {
    while event::poll(Duration::ZERO).unwrap_or(false) {
        let key = match event::read() {
            Ok(Event::Key(key)) if key.kind != KeyEventKind::Release => key,
            Ok(Event::Paste(text)) => {
                for c in text.chars() {
                    input.insert(c);
                }
                continue;
            }
            _ => continue,
        };
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c' | 'd') if ctrl => {
                commands.queue(save_session_command(config.paths.clone()));
                exit.write(AppExit::Success);
            }
            KeyCode::Char('u') if ctrl => input.set(String::new()),
            KeyCode::Char(c) => input.insert(c),
            KeyCode::Backspace if input.cursor > 0 => {
                input.cursor -= 1;
                let at = input.byte_index();
                input.text.remove(at);
            }
            KeyCode::Delete if input.cursor < input.text.chars().count() => {
                let at = input.byte_index();
                input.text.remove(at);
            }
            KeyCode::Left => input.cursor = input.cursor.saturating_sub(1),
            KeyCode::Right => input.cursor = (input.cursor + 1).min(input.text.chars().count()),
            KeyCode::Home => input.cursor = 0,
            KeyCode::End => input.cursor = input.text.chars().count(),
            KeyCode::PageUp => input.scroll += 10,
            KeyCode::PageDown => input.scroll = input.scroll.saturating_sub(10),
            KeyCode::Up if !input.history.is_empty() => {
                let at = input.browsing.map_or(input.history.len(), |i| i).saturating_sub(1);
                input.browsing = Some(at);
                let text = input.history[at].clone();
                input.set(text);
            }
            KeyCode::Down => match input.browsing {
                Some(at) if at + 1 < input.history.len() => {
                    input.browsing = Some(at + 1);
                    let text = input.history[at + 1].clone();
                    input.set(text);
                }
                _ => {
                    input.browsing = None;
                    input.set(String::new());
                }
            },
            KeyCode::Esc => input.set(String::new()),
            KeyCode::Enter => {
                let text = std::mem::take(&mut input.text).trim().to_string();
                input.set(String::new());
                input.scroll = 0;
                if text.is_empty() {
                    continue;
                }
                input.history.push(text.clone());
                match text.split_once(' ').unwrap_or((&text, "")) {
                    ("/quit" | "/exit", _) => {
                        commands.queue(save_session_command(config.paths.clone()));
                        exit.write(AppExit::Success);
                    }
                    ("/reload", _) => {
                        reload.write(ReloadCommand);
                    }
                    ("/model", "") => {
                        let aliases: Vec<String> = MODEL_ALIASES
                            .iter()
                            .map(|(alias, vendor, model)| format!("  {alias} = {vendor}:{model}"))
                            .collect();
                        transcript.push(
                            EntryKind::Info,
                            format!(
                                "model: {}\n{}\nor any Rig provider reference `vendor[/format]:model`",
                                active.label(),
                                aliases.join("\n")
                            ),
                        );
                    }
                    ("/model", name) => {
                        switch.write(SwitchModel(name.trim().to_string()));
                    }
                    ("/help", _) => transcript.push(EntryKind::Info, HELP),
                    _ => queue.0.push_back(text),
                }
            }
            _ => {}
        }
    }
}

impl Input {
    fn byte_index(&self) -> usize {
        self.text
            .char_indices()
            .nth(self.cursor)
            .map_or(self.text.len(), |(i, _)| i)
    }

    fn insert(&mut self, c: char) {
        let at = self.byte_index();
        self.text.insert(at, c);
        self.cursor += 1;
    }

    fn set(&mut self, text: String) {
        self.cursor = text.chars().count();
        self.text = text;
    }
}

#[allow(clippy::too_many_arguments)]
fn draw(
    mut term: NonSendMut<Term>,
    input: Res<Input>,
    transcript: Res<Transcript>,
    queue: Res<PromptQueue>,
    turn: Res<Turn>,
    active: Res<ActiveModel>,
    config: Res<Config>,
    time: Res<Time>,
) {
    const SPINNER: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];
    let spin = SPINNER[(time.elapsed_secs() * 10.0) as usize % SPINNER.len()];
    let status = match *turn {
        Turn::Idle => "idle".to_string(),
        Turn::Thinking(_) => format!("{spin} thinking"),
        Turn::Tools => format!("{spin} running tools"),
    };

    let mut lines: Vec<Line> = Vec::new();
    for entry in &transcript.0 {
        let (label, style) = match entry.kind {
            EntryKind::User => ("you › ", Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            EntryKind::Assistant => ("", Style::new()),
            EntryKind::ToolCall => ("⚙ ", Style::new().fg(Color::Yellow)),
            EntryKind::ToolResult => ("  ", Style::new().fg(Color::DarkGray)),
            EntryKind::Info => ("• ", Style::new().fg(Color::Green)),
            EntryKind::Error => ("✗ ", Style::new().fg(Color::Red)),
        };
        let mut text: Vec<&str> = entry.text.lines().collect();
        let hidden = match entry.kind {
            EntryKind::ToolResult | EntryKind::ToolCall if text.len() > 8 => text.len() - 8,
            _ => 0,
        };
        text.truncate(text.len() - hidden);
        for (i, line) in text.iter().enumerate() {
            let prefix = if i == 0 { label } else { "  " };
            lines.push(Line::from(vec![Span::styled(prefix, style), Span::styled(*line, style)]));
        }
        if hidden > 0 {
            lines.push(Line::styled(format!("  … {hidden} more lines"), style));
        }
        lines.push(Line::default());
    }

    let queued: Vec<Line> = queue
        .0
        .iter()
        .map(|prompt| Line::styled(format!("queued › {prompt}"), Style::new().fg(Color::Magenta)))
        .collect();

    let _ = term.0.draw(|frame| {
        let width = frame.area().width.saturating_sub(2).max(1) as usize;
        let input_lines = (input.text.chars().count() / width + 1).min(6) as u16;
        let [header, body, pending, prompt] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(queued.len().min(4) as u16),
            Constraint::Length(input_lines + 2),
        ])
        .areas(frame.area());

        frame.render_widget(
            Line::from(format!(
                " bevy-agent │ {} │ {status} │ BRP 127.0.0.1:{} ",
                active.label(),
                config.brp_port
            ))
            .style(Style::new().reversed()),
            header,
        );

        let transcript = Paragraph::new(lines).wrap(Wrap { trim: false });
        let total = transcript.line_count(body.width);
        let bottom = total.saturating_sub(body.height as usize);
        let top = bottom.saturating_sub(input.scroll);
        frame.render_widget(transcript.scroll((top as u16, 0)), body);

        frame.render_widget(Paragraph::new(queued), pending);

        frame.render_widget(
            Paragraph::new(input.text.as_str())
                .wrap(Wrap { trim: false })
                .block(Block::bordered().title(" prompt · /help ")),
            prompt,
        );
        let before = input.cursor as u16;
        let w = prompt.width.saturating_sub(2).max(1);
        frame.set_cursor_position(Position::new(
            prompt.x + 1 + before % w,
            prompt.y + 1 + (before / w).min(input_lines.saturating_sub(1)),
        ));
    });
}
