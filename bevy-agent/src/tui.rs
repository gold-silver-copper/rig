//! The terminal UI: ratatui drawing and crossterm input, as Bevy systems.

use std::io::{Stdout, stdout};
use std::time::Duration;

use bevy::prelude::*;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEventKind, KeyModifiers,
};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::layout::{Constraint, Layout, Position};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Wrap};
use ratatui::Terminal;

use crate::agent::{Agent, Cancel, Clear, Entry, Submit, Transcript, Turn};
use crate::hot::HotReload;

/// A line of status under the transcript, written by the status plugin.
#[derive(Resource, Default)]
pub struct StatusLine(pub Line<'static>);

#[derive(Resource, Default)]
struct Editor {
    input: String,
    /// Lines scrolled up from the bottom of the transcript.
    scroll: u16,
}

struct Term(Terminal<CrosstermBackend<Stdout>>);

impl Drop for Term {
    fn drop(&mut self) {
        restore();
    }
}

fn restore() {
    let _ = disable_raw_mode();
    let _ = execute!(stdout(), DisableBracketedPaste, LeaveAlternateScreen);
}

pub struct TuiPlugin;

impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        let terminal = enable_raw_mode()
            .and_then(|()| execute!(stdout(), EnterAlternateScreen, EnableBracketedPaste))
            .and_then(|()| Terminal::new(CrosstermBackend::new(stdout())));
        let terminal = match terminal {
            Ok(terminal) => terminal,
            Err(error) => {
                restore();
                panic!("cannot open the terminal: {error}");
            }
        };
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            restore();
            hook(info);
        }));
        app.insert_non_send(Term(terminal))
            .init_resource::<StatusLine>()
            .init_resource::<Editor>()
            .add_systems(PreUpdate, read_input)
            .add_systems(PostUpdate, draw);
    }
}

fn read_input(
    mut editor: ResMut<Editor>,
    mut hot: ResMut<HotReload>,
    mut submit: MessageWriter<Submit>,
    mut cancel: MessageWriter<Cancel>,
    mut clear: MessageWriter<Clear>,
    mut exit: MessageWriter<AppExit>,
    mut transcript: ResMut<Transcript>,
) {
    while event::poll(Duration::ZERO).unwrap_or(false) {
        let Ok(event) = event::read() else {
            break;
        };
        let key = match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => key,
            Event::Paste(text) => {
                editor.input.push_str(&text.replace('\r', "\n"));
                continue;
            }
            _ => continue,
        };
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c' | 'd') if ctrl => {
                exit.write(AppExit::Success);
            }
            KeyCode::Char('j') if ctrl => editor.input.push('\n'),
            KeyCode::Enter if key.modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) => {
                editor.input.push('\n');
            }
            KeyCode::Enter => {
                let text = std::mem::take(&mut editor.input).trim().to_owned();
                editor.scroll = 0;
                match text.as_str() {
                    "" => {}
                    "/quit" => {
                        exit.write(AppExit::Success);
                    }
                    "/clear" => {
                        clear.write_default();
                    }
                    "/reload" => {
                        if !hot.request() {
                            transcript.push(Entry::Error { text: "hot-patching is unavailable".into() });
                        }
                    }
                    _ => {
                        submit.write(Submit(text));
                    }
                }
            }
            KeyCode::Esc => {
                cancel.write_default();
            }
            KeyCode::Backspace => {
                editor.input.pop();
            }
            KeyCode::PageUp => editor.scroll = editor.scroll.saturating_add(10),
            KeyCode::PageDown => editor.scroll = editor.scroll.saturating_sub(10),
            KeyCode::Up => editor.scroll = editor.scroll.saturating_add(1),
            KeyCode::Down => editor.scroll = editor.scroll.saturating_sub(1),
            KeyCode::Char(c) => editor.input.push(c),
            _ => {}
        }
    }
}

fn draw(
    mut term: NonSendMut<Term>,
    transcript: Res<Transcript>,
    editor: Res<Editor>,
    status: Res<StatusLine>,
    agent: Res<Agent>,
) {
    let mut lines = transcript_lines(&transcript);
    if matches!(agent.turn, Turn::Thinking { streamed: false, .. }) {
        lines.push(Line::styled("…", Style::new().fg(Color::DarkGray)));
    }
    let _ = term.0.draw(|frame| {
        let width = frame.area().width.saturating_sub(2).max(1);
        let input_rows = editor
            .input
            .split('\n')
            .map(|line| (line.chars().count() as u16 / width) + 1)
            .sum::<u16>()
            .clamp(1, 8);
        let [body, status_area, input_area] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(input_rows + 2),
        ])
        .areas(frame.area());

        let transcript = Paragraph::new(lines).wrap(Wrap { trim: false });
        let bottom = (transcript.line_count(body.width) as u16).saturating_sub(body.height);
        let top = bottom.saturating_sub(editor.scroll.min(bottom));
        frame.render_widget(transcript.scroll((top, 0)), body);
        frame.render_widget(Paragraph::new(status.0.clone()), status_area);

        let hint = if agent.busy() { " esc to stop " } else { " enter to send · /reload · /clear · /quit " };
        let input = Paragraph::new(editor.input.as_str())
            .wrap(Wrap { trim: false })
            .block(Block::bordered().border_style(Style::new().fg(Color::DarkGray)).title_bottom(hint));
        frame.render_widget(input, input_area);

        let last = editor.input.split('\n').last().unwrap_or_default().chars().count() as u16;
        let row = input_rows.saturating_sub(1).min(input_area.height.saturating_sub(3));
        frame.set_cursor_position(Position::new(
            input_area.x + 1 + last % width,
            input_area.y + 1 + row,
        ));
    });
}

fn transcript_lines(transcript: &Transcript) -> Vec<Line<'static>> {
    let dim = Style::new().fg(Color::DarkGray);
    let mut lines = Vec::new();
    for entry in &transcript.entries {
        match entry {
            Entry::User { text } => {
                for (i, line) in text.lines().enumerate() {
                    let prefix = if i == 0 { "› " } else { "  " };
                    lines.push(Line::from(vec![
                        Span::styled(prefix, Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                        Span::styled(line.to_owned(), Style::new().add_modifier(Modifier::BOLD)),
                    ]));
                }
            }
            Entry::Assistant { text } => lines.extend(text.lines().map(|line| Line::raw(line.to_owned()))),
            Entry::Tool { name, arguments, output, is_error } => {
                let mut args: String = arguments.chars().take(120).collect();
                if arguments.chars().count() > 120 {
                    args.push('…');
                }
                lines.push(Line::from(vec![
                    Span::styled(format!("⚙ {name} "), Style::new().fg(Color::Yellow)),
                    Span::styled(args, dim),
                ]));
                let style = if *is_error { Style::new().fg(Color::Red) } else { dim };
                match output {
                    None => lines.push(Line::styled("  running…", dim)),
                    Some(output) => {
                        let count = output.lines().count();
                        for line in output.lines().take(6) {
                            lines.push(Line::styled(format!("  {line}"), style));
                        }
                        if count > 6 {
                            lines.push(Line::styled(format!("  … {} more lines", count - 6), dim));
                        }
                    }
                }
            }
            Entry::Info { text } => lines.push(Line::styled(format!("· {text}"), dim)),
            Entry::Error { text } => {
                lines.extend(text.lines().map(|line| Line::styled(line.to_owned(), Style::new().fg(Color::Red))));
            }
        }
        lines.push(Line::default());
    }
    lines
}
