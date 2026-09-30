//! The terminal interface for the TUI session: its transcript, the prompt
//! editor and a footer, drawn with ratatui every frame. Keys follow pi:
//! Enter sends (queued while the agent is busy), Alt+Enter adds a line, Esc
//! cancels the turn, Ctrl+C clears the editor and quits when it is empty,
//! Ctrl+D quits, Ctrl+P switches to the next model and Ctrl+O expands tool
//! output.

use std::time::Duration;

use bevy::prelude::*;
use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::{BrpPort, State};
use crate::glue::{self, Model, Turn};
use crate::reload::Reload;
use crate::session::{self, Entry, MODELS, Origin, Prompts, Transcript, Usage};

pub struct TuiPlugin;

impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Editor>()
            .add_systems(Startup, open)
            .add_systems(PreUpdate, input)
            .add_systems(PostUpdate, draw);
    }
}

#[derive(Resource)]
struct Screen(DefaultTerminal);

#[derive(Resource, Default)]
struct Editor {
    text: String,
    /// Lines scrolled up from the bottom of the transcript.
    scroll: usize,
    expanded: bool,
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

fn tui_agent(world: &mut World) -> Option<Entity> {
    world
        .query::<(Entity, &Origin)>()
        .iter(world)
        .find(|(_, origin)| **origin == Origin::Tui)
        .map(|(entity, _)| entity)
}

fn input(world: &mut World) {
    while let Ok(true) = event::poll(Duration::ZERO) {
        let Ok(Event::Key(key)) = event::read() else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        let Some(agent) = tui_agent(world) else {
            return;
        };
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let mut editor = world.resource_mut::<Editor>();
        match key.code {
            KeyCode::Char('c') if control => {
                if editor.text.is_empty() {
                    world.write_message(AppExit::Success);
                } else {
                    editor.text.clear();
                }
            }
            KeyCode::Char('d') if control && editor.text.is_empty() => {
                world.write_message(AppExit::Success);
            }
            KeyCode::Char('o') if control => editor.expanded = !editor.expanded,
            KeyCode::Char('p') if control => {
                let current = world.get::<Model>(agent).map(|model| model.spec().to_owned());
                let next = MODELS
                    .iter()
                    .position(|spec| Some(*spec) == current.as_deref())
                    .map_or(0, |index| (index + 1) % MODELS.len());
                session::switch_model(world, agent, MODELS[next]);
            }
            KeyCode::Char(c) if !control => editor.text.push(c),
            KeyCode::Enter if alt => editor.text.push('\n'),
            KeyCode::Enter => {
                let text = std::mem::take(&mut editor.text);
                editor.scroll = 0;
                session::submit(world, agent, &text);
            }
            KeyCode::Backspace => {
                editor.text.pop();
            }
            KeyCode::Up => editor.scroll += 1,
            KeyCode::Down => editor.scroll = editor.scroll.saturating_sub(1),
            KeyCode::PageUp => editor.scroll += 10,
            KeyCode::PageDown => editor.scroll = editor.scroll.saturating_sub(10),
            KeyCode::Esc => glue::cancel(world, agent),
            _ => {}
        }
    }
}

fn draw(
    screen: Option<ResMut<Screen>>,
    agents: Query<(&Origin, &Transcript, &Prompts, &Turn, Option<&Model>, Option<&Usage>)>,
    editor: Res<Editor>,
    reload: Res<Reload>,
    state: Res<State>,
    port: Option<Res<BrpPort>>,
) {
    let Some(mut screen) = screen else {
        return;
    };
    let Some((_, transcript, prompts, turn, model, usage)) =
        agents.iter().find(|(origin, ..)| **origin == Origin::Tui)
    else {
        return;
    };
    let activity = match turn {
        _ if reload.building() => "building".to_owned(),
        Turn::Idle => "idle".to_owned(),
        Turn::Request => "thinking".to_owned(),
        Turn::Tools { calls, results } => calls
            .get(results.len())
            .map_or("tools".to_owned(), |call| format!("running {}", call.function.name)),
    };
    let mut footer = format!(
        " {} · {} · {activity}",
        state.cwd_label,
        model.map_or("no model", Model::spec),
    );
    if let Some(usage) = usage {
        footer.push_str(&format!(" · ↑{} ↓{}", usage.input, usage.output));
    }
    if !prompts.queue.is_empty() {
        footer.push_str(&format!(" · {} queued", prompts.queue.len()));
    }
    if let Some(port) = port {
        footer.push_str(&format!(" · brp :{}", port.0));
    }
    footer.push_str(&format!(" · gen {} · /help", state.generation));
    let _ = screen.0.draw(|frame| {
        let width = usize::from(frame.area().width.max(1));
        let input = wrap(&format!("> {}", editor.text), width);
        let input_height = input.len().min(8);
        let [body, bar, field] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(input_height as u16),
        ])
        .areas(frame.area());

        let lines = render(transcript, prompts, width, editor.expanded);
        let height = usize::from(body.height);
        let end = lines.len() - editor.scroll.min(lines.len().saturating_sub(height));
        let start = end.saturating_sub(height);
        frame.render_widget(Paragraph::new(lines[start..end].to_vec()), body);
        frame.render_widget(
            Paragraph::new(footer.clone()).style(Style::new().add_modifier(Modifier::REVERSED)),
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

fn render(
    transcript: &Transcript,
    prompts: &Prompts,
    width: usize,
    expanded: bool,
) -> Vec<Line<'static>> {
    let output_lines = if expanded { usize::MAX } else { 8 };
    let entries = transcript.0.iter().map(|entry| match entry {
        Entry::User(text) => (Style::new().fg(Color::Cyan).bold(), format!("› {text}")),
        Entry::Assistant(text) => (Style::new(), text.clone()),
        Entry::Call(text) => (Style::new().fg(Color::Yellow), format!("⚙ {}", clip(text, 3))),
        Entry::Output(text) => (Style::new().fg(Color::DarkGray), clip(text, output_lines)),
        Entry::Notice(text) => (Style::new().fg(Color::Magenta), text.clone()),
        Entry::Error(text) => (Style::new().fg(Color::Red), text.clone()),
    });
    let queued = prompts
        .queue
        .iter()
        .map(|text| (Style::new().fg(Color::DarkGray), format!("› (queued) {text}")));
    let mut lines = Vec::new();
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
    let more = format!("… {} more lines (Ctrl+O)", total.saturating_sub(max));
    if total > max {
        shown.push(&more);
    }
    shown.join("\n")
}

/// Splits `text` into lines of at most `width` characters, at spaces where it can.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for source in text.split('\n') {
        let mut line: Vec<char> = Vec::new();
        for c in source.replace('\t', "    ").chars() {
            if line.len() == width {
                let cut = line
                    .iter()
                    .rposition(|&c| c == ' ')
                    .map_or(width, |space| space + 1);
                let rest = line.split_off(cut);
                lines.push(line.iter().collect());
                line = rest;
            }
            line.push(c);
        }
        lines.push(line.iter().collect());
    }
    lines
}

/// Shows `entry` in the TUI session's transcript, when there is one.
pub fn notify(world: &mut World, entry: Entry) {
    if let Some(agent) = tui_agent(world) {
        session::log(world, agent, entry);
    }
}
