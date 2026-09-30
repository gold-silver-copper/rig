//! The terminal UI: a transcript, a status line, and a prompt.
//! Enter sends (queued while a turn runs), Alt+Enter adds a newline,
//! PgUp/PgDn scroll, Ctrl+C clears the prompt or quits.

use std::time::Duration;

use bevy::prelude::*;
use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Position};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};

use crate::agent::Runtime;
use crate::session::{Entry, PRESETS, Paths, Session, parse_model};
use crate::tools::Tools;

pub struct TuiPlugin;

impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Tui { terminal: ratatui::init(), input: String::new(), scroll: 0 })
            .add_systems(Update, (input, draw).chain());
    }
}

#[derive(Resource)]
pub struct Tui {
    terminal: DefaultTerminal,
    input: String,
    /// Lines scrolled up from the bottom.
    scroll: usize,
}

fn input(
    mut tui: ResMut<Tui>,
    mut session: ResMut<Session>,
    mut rt: ResMut<Runtime>,
    tools: Res<Tools>,
    paths: Res<Paths>,
    mut exit: MessageWriter<AppExit>,
) {
    while event::poll(Duration::ZERO).unwrap_or(false) {
        let Ok(event) = event::read() else { break };
        let key = match event {
            Event::Key(key) if key.kind == KeyEventKind::Press => key,
            Event::Paste(text) => {
                tui.input.push_str(&text);
                continue;
            }
            _ => continue,
        };
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c') if ctrl && !tui.input.is_empty() => tui.input.clear(),
            KeyCode::Char('c' | 'd') if ctrl => {
                session.save(&paths.session());
                exit.write(AppExit::Success);
            }
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::ALT) => tui.input.push('\n'),
            KeyCode::Enter => {
                let text = std::mem::take(&mut tui.input);
                tui.scroll = 0;
                submit(text.trim(), &mut session, &mut rt, &tools, &mut exit);
            }
            KeyCode::Backspace => {
                tui.input.pop();
            }
            KeyCode::PageUp => tui.scroll += 10,
            KeyCode::PageDown => tui.scroll = tui.scroll.saturating_sub(10),
            KeyCode::Up => tui.scroll += 1,
            KeyCode::Down => tui.scroll = tui.scroll.saturating_sub(1),
            KeyCode::Char(c) if !ctrl => tui.input.push(c),
            _ => {}
        }
    }
}

fn submit(text: &str, session: &mut Session, rt: &mut Runtime, tools: &Tools, exit: &mut MessageWriter<AppExit>) {
    let (command, rest) = text.split_once(' ').unwrap_or((text, ""));
    match command {
        "" => {}
        "/quit" | "/exit" => {
            exit.write(AppExit::Success);
        }
        "/reload" => {
            rt.user_reload = true;
            session.info("reload requested: it builds once the current turn is done");
        }
        "/model" if rest.trim().is_empty() => {
            let list: Vec<String> = PRESETS.iter().enumerate().map(|(i, p)| format!("  {}. {p}", i + 1)).collect();
            session.info(format!(
                "model: {}\n{}\nswitch with /model <number | name | vendor:model>",
                session.model,
                list.join("\n")
            ));
        }
        "/model" => match parse_model(rest) {
            Ok(model) => {
                session.info(format!("model: {model}"));
                session.model = model;
            }
            Err(error) => session.error(format!("unknown model `{}`: {error}", rest.trim())),
        },
        "/tools" => {
            let names: Vec<&str> = tools.0.keys().map(String::as_str).collect();
            session.info(format!("tools: {}", names.join(", ")));
        }
        "/help" => session.info(
            "/model [choice]  list or switch models\n/reload  rebuild and restart\n/tools  list tools\n/quit",
        ),
        _ => session.queue.push_back(text.to_owned()),
    }
}

fn style(entry: &Entry) -> (&'static str, Style, &str) {
    match entry {
        Entry::User(text) => ("› ", Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD), text),
        Entry::Assistant(text) => ("", Style::new(), text),
        Entry::ToolCall(text) => ("⚙ ", Style::new().fg(Color::Yellow), text),
        Entry::ToolResult(text) => ("  ↳ ", Style::new().fg(Color::DarkGray), text),
        Entry::Info(text) => ("· ", Style::new().fg(Color::Magenta), text),
        Entry::Error(text) => ("! ", Style::new().fg(Color::Red), text),
    }
}

/// Hard-wrap `text` to `width` columns (by character).
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.split('\n') {
        let chars: Vec<char> = line.replace('\t', "    ").chars().collect();
        if chars.is_empty() {
            out.push(String::new());
        }
        for chunk in chars.chunks(width.max(1)) {
            out.push(chunk.iter().collect());
        }
    }
    out
}

fn transcript(entries: &[Entry], width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for entry in entries {
        let (prefix, style, text) = style(entry);
        let indent = " ".repeat(prefix.chars().count());
        for (i, piece) in wrap(text, width.saturating_sub(prefix.len()).max(10)).into_iter().enumerate() {
            let lead = if i == 0 { prefix.to_owned() } else { indent.clone() };
            lines.push(Line::from(vec![Span::styled(lead, style), Span::styled(piece, style)]));
        }
        lines.push(Line::default());
    }
    lines
}

fn draw(mut tui: ResMut<Tui>, session: Res<Session>, rt: Res<Runtime>) {
    let Tui { terminal, input, scroll } = &mut *tui;
    let _ = terminal.draw(|frame| {
        let width = frame.area().width.saturating_sub(2) as usize;
        let prompt = wrap(&format!("› {input}"), width.max(1));
        let prompt_height = (prompt.len() as u16 + 2).min(frame.area().height / 2);
        let [body, status, prompt_area] =
            Layout::vertical([Constraint::Min(1), Constraint::Length(1), Constraint::Length(prompt_height)])
                .areas(frame.area());

        let lines = transcript(&session.transcript, body.width as usize);
        let height = body.height as usize;
        *scroll = (*scroll).min(lines.len().saturating_sub(height));
        let end = lines.len() - *scroll;
        let start = end.saturating_sub(height);
        frame.render_widget(Paragraph::new(lines[start..end].to_vec()), body);

        let queued = match session.queue.len() {
            0 => String::new(),
            n => format!(" │ {n} queued"),
        };
        let status_text = format!(
            " {} │ {}{queued} │ BRP 127.0.0.1:{} │ /help",
            session.model,
            rt.activity(),
            session.port
        );
        frame.render_widget(Paragraph::new(status_text).reversed(), status);

        let text: Vec<Line> = prompt.iter().map(|l| Line::from(l.clone())).collect();
        frame.render_widget(Paragraph::new(text).block(Block::bordered()), prompt_area);
        let last = prompt.last().map_or(0, |l| l.chars().count()) as u16;
        frame.set_cursor_position(Position::new(
            prompt_area.x + 1 + last,
            prompt_area.y + prompt.len().min(prompt_height as usize - 2) as u16,
        ));
    });
}
