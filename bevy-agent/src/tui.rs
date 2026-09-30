//! The terminal interface: the transcript, a status line and the prompt,
//! drawn with ratatui every frame. Keys are read without blocking. A prompt
//! sent while the agent is busy waits in the queue.

use std::time::Duration;

use bevy::prelude::*;
use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::Boot;
use crate::agent::{self, Llm};
use crate::reload::Reload;
use crate::session::{Entry, Session, Turn};

/// Models offered by `/model`; any `vendor[/format]:model` Rig knows works.
const MODELS: [&str; 4] = [
    "openai:gpt-6.1-sol",
    "anthropic:claude-opus-5-5",
    "gemini:gemini-3.8-flash",
    "deepseek:deepseek-flash",
];

pub struct TuiPlugin;

impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Prompt>()
            .add_systems(Startup, open)
            .add_systems(PreUpdate, input)
            .add_systems(PostUpdate, draw);
    }
}

#[derive(Resource)]
struct Screen(DefaultTerminal);

#[derive(Resource, Default)]
struct Prompt {
    text: String,
    /// Lines scrolled up from the bottom of the transcript.
    scroll: usize,
}

fn open(mut commands: Commands, mut exit: MessageWriter<AppExit>) {
    match ratatui::try_init() {
        Ok(terminal) => commands.insert_resource(Screen(terminal)),
        Err(error) => {
            eprintln!("cannot open the terminal: {error}");
            exit.write(AppExit::error());
        }
    }
}

fn input(world: &mut World) {
    while let Ok(true) = event::poll(Duration::ZERO) {
        let Ok(Event::Key(key)) = event::read() else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        let mut prompt = world.resource_mut::<Prompt>();
        match key.code {
            KeyCode::Char('c' | 'd') if control => {
                if prompt.text.is_empty() {
                    world.write_message(AppExit::Success);
                } else {
                    prompt.text.clear();
                }
            }
            KeyCode::Char(c) => prompt.text.push(c),
            KeyCode::Backspace => {
                prompt.text.pop();
            }
            KeyCode::Up => prompt.scroll += 1,
            KeyCode::Down => prompt.scroll = prompt.scroll.saturating_sub(1),
            KeyCode::PageUp => prompt.scroll += 10,
            KeyCode::PageDown => prompt.scroll = prompt.scroll.saturating_sub(10),
            KeyCode::Esc => agent::cancel(world),
            KeyCode::Enter => {
                let text = std::mem::take(&mut prompt.text);
                prompt.scroll = 0;
                submit(world, text.trim());
            }
            _ => {}
        }
    }
}

fn submit(world: &mut World, text: &str) {
    let (command, argument) = text.split_once(' ').unwrap_or((text, ""));
    match command {
        "" => {}
        "/quit" => {
            world.write_message(AppExit::Success);
        }
        "/model" if argument.is_empty() => {
            let mut session = world.resource_mut::<Session>();
            let current = session.model.clone();
            session.log(Entry::Notice(format!(
                "Model: {current}. Switch with /model <vendor:model>, for example:\n{}",
                MODELS.join("\n")
            )));
        }
        "/model" => world.resource_scope(|world, mut llm: Mut<Llm>| {
            agent::switch_model(&mut world.resource_mut::<Session>(), &mut llm, argument.trim());
        }),
        _ => world
            .resource_mut::<Session>()
            .queue
            .push_back(text.to_owned()),
    }
}

fn draw(
    screen: Option<ResMut<Screen>>,
    session: Res<Session>,
    prompt: Res<Prompt>,
    reload: Res<Reload>,
    boot: Res<Boot>,
) {
    let Some(mut screen) = screen else {
        return;
    };
    let state = match &session.turn {
        _ if reload.building() => "building".to_owned(),
        Turn::Idle => "idle".to_owned(),
        Turn::Request => "thinking".to_owned(),
        Turn::Tools { calls, results } => calls
            .get(results.len())
            .map_or("tools".to_owned(), |call| format!("running {}", call.function.name)),
    };
    let status = format!(
        " {} · {state} · {} queued · brp :{} · gen {} · Esc cancel · /model /reload /quit",
        session.model,
        session.queue.len(),
        session.brp_port,
        boot.generation,
    );
    let _ = screen.0.draw(|frame| {
        let width = usize::from(frame.area().width.max(1));
        let input = wrap(&format!("> {}", prompt.text), width);
        let input_height = input.len().min(5);
        let [body, bar, field] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(input_height as u16),
        ])
        .areas(frame.area());

        let lines = transcript(&session, width);
        let height = usize::from(body.height);
        let end = lines.len() - prompt.scroll.min(lines.len().saturating_sub(height));
        let start = end.saturating_sub(height);
        frame.render_widget(Paragraph::new(lines[start..end].to_vec()), body);
        frame.render_widget(
            Paragraph::new(status).style(Style::new().add_modifier(Modifier::REVERSED)),
            bar,
        );
        let shown = &input[input.len() - input_height..];
        let cursor_x = shown.last().map_or(0, |line| line.chars().count()) as u16;
        frame.render_widget(
            Paragraph::new(shown.iter().map(|line| Line::raw(line.clone())).collect::<Vec<_>>()),
            field,
        );
        frame.set_cursor_position((
            field.x + cursor_x.min(field.width.saturating_sub(1)),
            field.y + input_height as u16 - 1,
        ));
    });
}

fn transcript(session: &Session, width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let entries = session.transcript.iter().map(|entry| match entry {
        Entry::User(text) => (Style::new().fg(Color::Cyan).bold(), format!("› {text}")),
        Entry::Assistant(text) => (Style::new(), text.clone()),
        Entry::Call(text) => (Style::new().fg(Color::Yellow), format!("⚙ {}", clip(text, 3))),
        Entry::Output(text) => (Style::new().fg(Color::DarkGray), clip(text, 8)),
        Entry::Notice(text) => (Style::new().fg(Color::Magenta), text.clone()),
        Entry::Error(text) => (Style::new().fg(Color::Red), text.clone()),
    });
    let queued = session
        .queue
        .iter()
        .map(|text| (Style::new().fg(Color::DarkGray), format!("› (queued) {text}")));
    for (style, text) in entries.chain(queued) {
        for line in wrap(&text, width) {
            lines.push(Line::from(Span::styled(line, style)));
        }
        lines.push(Line::default());
    }
    lines
}

/// The first `max` lines of `text`, noting how many were left out.
fn clip(text: &str, max: usize) -> String {
    let total = text.lines().count();
    let mut shown: Vec<&str> = text.lines().take(max).collect();
    let more = format!("… {} more lines", total.saturating_sub(max));
    if total > max {
        shown.push(&more);
    }
    shown.join("\n")
}

/// Splits `text` into lines of at most `width` characters.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for line in text.split('\n') {
        let chars: Vec<char> = line.replace('\t', "    ").chars().collect();
        if chars.is_empty() {
            lines.push(String::new());
        }
        for chunk in chars.chunks(width) {
            lines.push(chunk.iter().collect());
        }
    }
    lines
}
