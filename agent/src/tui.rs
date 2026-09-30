//! The terminal UI: transcript, prompt queue, input line, status bar.
//!
//! Enter sends (or queues, while a turn runs). Esc cancels the turn. PageUp
//! and PageDown scroll. `/model [n|vendor:model]`, `/reload`, `/clear`,
//! `/quit`.

use std::time::Duration;

use bevy::prelude::*;
use ratatui::DefaultTerminal;
use ratatui::crossterm::cursor::Show;
use ratatui::crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEventKind, KeyModifiers,
};
use ratatui::crossterm::execute;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};

use crate::agent::{self, InFlight};
use crate::reload::UserReload;
use crate::session::{Kind, Phase, Session};
use crate::{Env, presets};

pub struct TuiPlugin;

impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        let terminal = ratatui::init();
        let _ = execute!(std::io::stdout(), EnableBracketedPaste);
        app.insert_non_send(Tui {
            terminal,
            input: String::new(),
            scroll: 0,
            frames: 0,
            dirty: true,
        })
        .add_systems(
            Update,
            (input.before(agent::drive), draw.after(agent::drive)),
        );
    }
}

pub fn restore() {
    let _ = execute!(std::io::stdout(), DisableBracketedPaste, Show);
    ratatui::restore();
}

struct Tui {
    terminal: DefaultTerminal,
    input: String,
    /// Lines scrolled up from the bottom.
    scroll: usize,
    frames: u64,
    /// Input or the terminal changed since the last draw.
    dirty: bool,
}

fn input(
    mut tui: NonSendMut<Tui>,
    mut session: ResMut<Session>,
    mut inflight: ResMut<InFlight>,
    mut reload: ResMut<UserReload>,
    env: Res<Env>,
    mut exit: MessageWriter<AppExit>,
) {
    while event::poll(Duration::ZERO).unwrap_or(false) {
        let Ok(event) = event::read() else { return };
        tui.dirty = true;
        let key = match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => key,
            Event::Paste(text) => {
                tui.input.push_str(&text);
                continue;
            }
            _ => continue,
        };
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c' | 'd') if ctrl => {
                exit.write(AppExit::Success);
            }
            KeyCode::Char(c) if !ctrl => tui.input.push(c),
            KeyCode::Backspace => {
                tui.input.pop();
            }
            KeyCode::PageUp | KeyCode::Up => tui.scroll += 10,
            KeyCode::PageDown | KeyCode::Down => tui.scroll = tui.scroll.saturating_sub(10),
            KeyCode::Esc if session.busy() => agent::cancel(&mut session, &mut inflight),
            KeyCode::Enter => {
                let line = std::mem::take(&mut tui.input);
                let line = line.trim();
                tui.scroll = 0;
                if line.is_empty() {
                    continue;
                }
                match line.split_once(' ').unwrap_or((line, "")) {
                    ("/quit", _) => {
                        exit.write(AppExit::Success);
                    }
                    ("/reload", _) => reload.start(&mut session, &env),
                    ("/clear", _) if session.busy() => {
                        session.log(Kind::Error, "/clear: a turn is running (Esc cancels it)");
                    }
                    ("/clear", _) => {
                        session.messages.clear();
                        session.transcript.clear();
                    }
                    ("/model", "") => {
                        let list: Vec<String> = presets()
                            .iter()
                            .enumerate()
                            .map(|(i, preset)| format!("{}. {preset}", i + 1))
                            .collect();
                        let current = session.model.to_string();
                        session.log(
                            Kind::Info,
                            format!(
                                "model: {current}\n{}\n/model <n|vendor:model>",
                                list.join("\n")
                            ),
                        );
                    }
                    ("/model", reference) => {
                        if let Err(error) = session.switch_model(reference.trim()) {
                            session.log(Kind::Error, error);
                        }
                    }
                    (command, _) if command.starts_with('/') && !command[1..].contains('/') => {
                        session.log(
                            Kind::Error,
                            format!("unknown command {command}: try /model /reload /clear /quit"),
                        );
                    }
                    _ => session.queue.push_back(line.to_owned()),
                }
            }
            _ => {}
        }
    }
}

/// Hard-wrap `text` to `width` columns under a hanging `prefix`, one styled
/// line per row.
fn wrap<'a>(prefix: &str, text: &str, width: usize, style: Style, out: &mut Vec<Line<'a>>) {
    let indent = " ".repeat(prefix.chars().count());
    let width = width.saturating_sub(indent.len()).max(1);
    let mut lead = prefix;
    for line in text.lines() {
        let chars: Vec<char> = line.chars().collect();
        let rows: Vec<String> = if chars.is_empty() {
            vec![String::new()]
        } else {
            chars
                .chunks(width)
                .map(|chunk| chunk.iter().collect())
                .collect()
        };
        for row in rows {
            out.push(Line::from(Span::styled(format!("{lead}{row}"), style)));
            lead = &indent;
        }
    }
}

fn draw(
    mut tui: NonSendMut<Tui>,
    session: Res<Session>,
    inflight: Res<InFlight>,
    reload: Res<UserReload>,
    env: Res<Env>,
) {
    let tui = &mut *tui;
    tui.frames += 1;
    let spinning = session.busy() || reload.building();
    if !(tui.dirty || session.is_changed() || spinning && tui.frames % 6 == 0) {
        return;
    }
    tui.dirty = false;
    let spinner =
        ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"][(tui.frames / 4 % 10) as usize];
    let state = match &session.phase {
        Phase::Idle => "idle".to_owned(),
        Phase::Model => format!("{spinner} thinking"),
        Phase::Tools { calls, next, .. } => match calls.get(*next) {
            Some(call) if inflight.tool_running() => format!("{spinner} {}", call.function.name),
            _ => format!("{spinner} tools"),
        },
    };
    let reloading = if reload.building() { " | building" } else { "" };
    let status = format!(
        " {} | {state}{reloading} | BRP :{} | Enter send · Esc cancel · /model /reload /clear /quit",
        session.model, env.port
    );
    let (input, scroll) = (tui.input.clone(), &mut tui.scroll);
    let _ = tui.terminal.draw(|frame| {
        let queue_height = session.queue.len().min(3) as u16;
        let [body, queue, prompt, bar] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(queue_height),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        let width = body.width as usize;
        let mut lines = Vec::new();
        for entry in &session.transcript {
            let (prefix, style) = match entry.kind {
                Kind::User => (
                    "> ",
                    Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                ),
                Kind::Assistant => ("", Style::new()),
                Kind::Call => ("→ ", Style::new().fg(Color::Yellow)),
                Kind::Result => ("  ", Style::new().fg(Color::DarkGray)),
                Kind::Info => ("· ", Style::new().fg(Color::Green)),
                Kind::Error => ("! ", Style::new().fg(Color::Red)),
            };
            let text = if entry.kind == Kind::Result {
                let shown: Vec<&str> = entry.text.lines().take(12).collect();
                let more = entry.text.lines().count().saturating_sub(shown.len());
                let mut text = shown.join("\n");
                if more > 0 {
                    text.push_str(&format!("\n… {more} more lines"));
                }
                text
            } else {
                entry.text.clone()
            };
            wrap(prefix, &text, width, style, &mut lines);
            lines.push(Line::default());
        }
        let height = body.height as usize;
        *scroll = (*scroll).min(lines.len().saturating_sub(height));
        let top = lines.len().saturating_sub(height + *scroll);
        let visible: Vec<Line> = lines.into_iter().skip(top).take(height).collect();
        frame.render_widget(Paragraph::new(visible), body);

        let queued: Vec<Line> = session
            .queue
            .iter()
            .map(|q| Line::styled(format!("queued: {q}"), Style::new().fg(Color::Magenta)))
            .collect();
        frame.render_widget(Paragraph::new(queued), queue);

        let inner = prompt.width.saturating_sub(2) as usize;
        let shown: String = {
            let chars: Vec<char> = input.chars().collect();
            chars[chars.len().saturating_sub(inner.saturating_sub(1))..]
                .iter()
                .collect()
        };
        frame.render_widget(
            Paragraph::new(shown.clone()).block(Block::bordered()),
            prompt,
        );
        frame.set_cursor_position((prompt.x + 1 + shown.chars().count() as u16, prompt.y + 1));
        frame.render_widget(
            Paragraph::new(status).style(Style::new().bg(Color::DarkGray).fg(Color::White)),
            bar,
        );
    });
}
