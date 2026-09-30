//! The terminal: keys in, transcript out, one frame per tick.

use std::sync::Mutex;
use std::time::Duration;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::DefaultTerminal;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::Paragraph;

use crate::agent::{Conversation, Entry, InFlight, Llm, Transcript};
use crate::hot::Hot;
use crate::plugins;
use crate::remote::BrpPort;

pub struct TuiPlugin(Mutex<Option<DefaultTerminal>>);

impl TuiPlugin {
    pub fn new(terminal: DefaultTerminal) -> Self {
        Self(Mutex::new(Some(terminal)))
    }
}

impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        let terminal = self.0.lock().ok().and_then(|mut slot| slot.take());
        app.insert_resource(Tui {
            terminal,
            input: String::new(),
            scroll: 0,
        })
        .add_systems(PreUpdate, read_keys)
        .add_systems(Last, render);
    }
}

#[derive(Resource)]
pub struct Tui {
    terminal: Option<DefaultTerminal>,
    input: String,
    scroll: usize,
}

fn read_keys(
    mut tui: ResMut<Tui>,
    mut convo: ResMut<Conversation>,
    mut transcript: ResMut<Transcript>,
    mut hot: ResMut<Hot>,
    in_flight: Option<Res<InFlight>>,
    mut exit: MessageWriter<AppExit>,
) {
    while event::poll(Duration::ZERO).unwrap_or(false) {
        let Ok(Event::Key(key)) = event::read() else {
            continue;
        };
        if key.kind == KeyEventKind::Release {
            continue;
        }
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c' | 'd') if control => {
                exit.write(AppExit::Success);
            }
            KeyCode::Char(c) => tui.input.push(c),
            KeyCode::Backspace => {
                tui.input.pop();
            }
            KeyCode::PageUp => tui.scroll += 5,
            KeyCode::PageDown => tui.scroll = tui.scroll.saturating_sub(5),
            KeyCode::Enter => {
                let line = std::mem::take(&mut tui.input);
                tui.scroll = 0;
                match line.trim() {
                    "" => {}
                    "/quit" => {
                        exit.write(AppExit::Success);
                    }
                    "/patch" => hot.request(),
                    "/clear" => {
                        transcript.entries.clear();
                        convo.history.clear();
                    }
                    _ if convo.busy(in_flight.is_some()) => {
                        transcript.note("busy; wait for the turn to end")
                    }
                    text => convo.prompt(text.to_owned(), &mut transcript),
                }
            }
            _ => {}
        }
    }
}

fn render(
    mut tui: ResMut<Tui>,
    transcript: Res<Transcript>,
    convo: Res<Conversation>,
    in_flight: Option<Res<InFlight>>,
    llm: Res<Llm>,
    port: Res<BrpPort>,
    hot: Res<Hot>,
) {
    let Tui {
        terminal,
        input,
        scroll,
    } = &mut *tui;
    let Some(terminal) = terminal else {
        return;
    };
    let status = format!(
        " {} | brp :{} | patch: {} | {} | {}",
        llm.name,
        port.0,
        hot.status(),
        plugins::tools::status_hint(),
        if convo.busy(in_flight.is_some()) {
            "working"
        } else {
            "idle"
        }
    );
    let _ = terminal.draw(|frame| {
        let [body, prompt, bar] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .areas(frame.area());
        let lines = wrap(&transcript.entries, body.width.max(4) as usize);
        let end = lines.len().saturating_sub(*scroll);
        let start = end.saturating_sub(body.height as usize);
        frame.render_widget(Paragraph::new(lines[start..end].to_vec()), body);
        frame.render_widget(Paragraph::new(format!("> {input}")), prompt);
        frame.set_cursor_position((prompt.x + 2 + input.chars().count() as u16, prompt.y));
        frame.render_widget(
            Paragraph::new(status).style(Style::new().add_modifier(Modifier::REVERSED)),
            bar,
        );
    });
}

/// The transcript as styled lines no wider than `width` columns.
fn wrap(entries: &[Entry], width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for entry in entries {
        let (prefix, text, style) = match entry {
            Entry::User(text) => (
                "> ",
                text.clone(),
                Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
            Entry::Assistant(text) => ("", text.clone(), Style::new()),
            Entry::Tool { name, args, result } => {
                let mut text = format!("{name} {args}");
                if let Some(result) = result {
                    let short: String = result.chars().take(160).collect();
                    text.push_str(&format!(" → {}", short.replace('\n', " ")));
                }
                ("⚙ ", text, Style::new().fg(Color::Yellow))
            }
            Entry::Note(text) => ("· ", text.clone(), Style::new().fg(Color::DarkGray)),
        };
        for (i, raw) in text.lines().enumerate() {
            let mut row: String = if i == 0 {
                prefix.to_owned()
            } else {
                " ".repeat(prefix.len())
            };
            for ch in raw.chars() {
                if row.chars().count() >= width {
                    lines.push(Line::styled(
                        std::mem::replace(&mut row, " ".repeat(prefix.len())),
                        style,
                    ));
                }
                row.push(ch);
            }
            lines.push(Line::styled(row, style));
        }
        if text.is_empty() {
            lines.push(Line::styled(prefix.to_owned(), style));
        }
    }
    lines
}
