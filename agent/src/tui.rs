//! The terminal UI. It shows and drives the [`Focused`] agent.
//!
//! Keys follow pi: Enter sends (steering the running turn if there is one),
//! Alt+Enter queues a follow-up, Esc interrupts, Ctrl+C clears the input,
//! Ctrl+D quits on an empty input, Ctrl+P cycles models, Ctrl+O expands tool
//! output, PageUp/PageDown scroll. Commands: `/model [n|vendor:model]`,
//! `/reload`, `/new`, `/sessions`, `/focus <n>`, `/quit`, plus any a plugin
//! registers in [`SlashCommands`].

use std::collections::BTreeMap;
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
use rig_core::providers::registry::ProviderRef;

use crate::glue::{Agent, AgentSet, Inbox, Interrupt, Model, Session, Turn};
use crate::reload::{ReloadRequest, Reloading};
use crate::prompt::Prompt;
use crate::glue::Tools;
use crate::session::{EntryKind, Focused, Kind, Transcript, new_session_id, spawn_agent};
use crate::{Env, presets};

/// Slash commands plugins handle, by name, with a description.
#[derive(Resource, Default)]
pub struct SlashCommands(pub BTreeMap<String, String>);

/// A registered slash command the user typed, for the plugin that owns it.
#[derive(Message, Clone, Debug)]
pub struct SlashCommand {
    pub agent: Entity,
    pub name: String,
    pub args: String,
}

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
            expanded: false,
        })
        .init_resource::<SlashCommands>()
        .add_message::<SlashCommand>()
        .add_systems(Update, (input.in_set(AgentSet::Input), draw.after(AgentSet::Run)));
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
    /// Show whole tool results.
    expanded: bool,
}

type FocusedAgent<'a> = (
    Entity,
    &'a Model,
    &'a mut Inbox,
    &'a Turn,
    &'a mut Transcript,
);

#[allow(clippy::too_many_arguments)]
fn input(
    mut tui: NonSendMut<Tui>,
    mut focused: Query<FocusedAgent, With<Focused>>,
    others: Query<(Entity, &Name, &Session, &Model, &Turn, &Kind), With<Agent>>,
    registered: Res<SlashCommands>,
    prompt: Res<Prompt>,
    tools: Res<Tools>,
    mut slash: MessageWriter<SlashCommand>,
    mut interrupts: MessageWriter<Interrupt>,
    mut reloads: MessageWriter<ReloadRequest>,
    mut exit: MessageWriter<AppExit>,
    mut commands: Commands,
) {
    let Ok((agent, model, mut inbox, turn, mut transcript)) = focused.single_mut() else {
        return;
    };
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
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        match key.code {
            KeyCode::Char('c') if ctrl => tui.input.clear(),
            KeyCode::Char('d') if ctrl && tui.input.is_empty() => {
                exit.write(AppExit::Success);
            }
            KeyCode::Char('p') if ctrl => {
                let presets = presets();
                let current = model.0.to_string();
                let at = presets
                    .iter()
                    .position(|p| ProviderRef::parse(p).is_ok_and(|p| p.to_string() == current));
                let next = &presets[at.map_or(0, |i| (i + 1) % presets.len())];
                if let Ok(next) = ProviderRef::parse(next) {
                    transcript.log(EntryKind::Info, format!("model: {next}"));
                    commands.entity(agent).insert(Model(next));
                }
            }
            KeyCode::Char('o') if ctrl => tui.expanded = !tui.expanded,
            KeyCode::Char(c) if !ctrl => tui.input.push(c),
            KeyCode::Backspace => {
                tui.input.pop();
            }
            KeyCode::PageUp | KeyCode::Up => tui.scroll += 10,
            KeyCode::PageDown | KeyCode::Down => tui.scroll = tui.scroll.saturating_sub(10),
            KeyCode::Esc if !turn.is_idle() => {
                interrupts.write(Interrupt(agent));
            }
            KeyCode::Enter => {
                let line = std::mem::take(&mut tui.input);
                let line = line.trim().to_owned();
                tui.scroll = 0;
                if line.is_empty() {
                    continue;
                }
                let first = line.split_whitespace().next().unwrap_or_default();
                if !first.starts_with('/') || first[1..].contains('/') {
                    if alt || turn.is_idle() {
                        inbox.follow_ups.push_back(line);
                    } else {
                        inbox.steering.push(line);
                    }
                    continue;
                }
                let (name, args) = line.split_once(' ').unwrap_or((&line, ""));
                let args = args.trim();
                match name {
                    "/quit" => {
                        exit.write(AppExit::Success);
                    }
                    "/reload" => {
                        reloads.write(ReloadRequest);
                    }
                    "/model" if args.is_empty() => {
                        let list: Vec<String> = presets()
                            .iter()
                            .enumerate()
                            .map(|(i, preset)| format!("{}. {preset}", i + 1))
                            .collect();
                        transcript.log(
                            EntryKind::Info,
                            format!("model: {}\n{}\n/model <n|vendor:model>", model.0, list.join("\n")),
                        );
                    }
                    "/model" => match parse_model(args) {
                        Ok(next) => {
                            transcript.log(EntryKind::Info, format!("model: {next}"));
                            commands.entity(agent).insert(Model(next));
                        }
                        Err(error) => transcript.log(EntryKind::Error, error),
                    },
                    "/new" if !turn.is_idle() => {
                        transcript.log(EntryKind::Error, "/new: a turn is running (Esc interrupts it)")
                    }
                    "/new" => {
                        let preamble = prompt.build(&tools);
                        let fresh = spawn_agent(
                            &mut commands,
                            preamble,
                            Kind::Tui,
                            "tui",
                            model.0.clone(),
                            new_session_id(Kind::Tui),
                        );
                        commands.entity(fresh).insert(Focused);
                        commands.entity(agent).despawn();
                    }
                    "/sessions" => {
                        let list: Vec<String> = others
                            .iter()
                            .enumerate()
                            .map(|(i, (entity, name, session, model, turn, kind))| {
                                format!(
                                    "{}. {} {:?} {} {} {}{}",
                                    i + 1,
                                    name,
                                    kind,
                                    session.0,
                                    model.0,
                                    state(turn),
                                    if entity == agent { " (shown)" } else { "" }
                                )
                            })
                            .collect();
                        transcript.log(EntryKind::Info, format!("{}\n/focus <n> shows one", list.join("\n")));
                    }
                    "/focus" => match args
                        .parse::<usize>()
                        .ok()
                        .and_then(|n| others.iter().nth(n.wrapping_sub(1)))
                    {
                        Some((entity, ..)) => {
                            commands.entity(agent).remove::<Focused>();
                            commands.entity(entity).insert(Focused);
                        }
                        None => transcript.log(EntryKind::Error, "/focus <n>: see /sessions"),
                    },
                    other if registered.0.contains_key(&other[1..]) => {
                        slash.write(SlashCommand {
                            agent,
                            name: other[1..].to_owned(),
                            args: args.to_owned(),
                        });
                    }
                    other => {
                        let mut known: Vec<String> =
                            ["model", "reload", "new", "sessions", "focus", "quit"].map(String::from).to_vec();
                        known.extend(registered.0.keys().cloned());
                        transcript.log(
                            EntryKind::Error,
                            format!("unknown command {other}; try /{}", known.join(" /")),
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

pub fn parse_model(reference: &str) -> Result<ProviderRef, String> {
    let presets = presets();
    let reference = match reference.parse::<usize>() {
        Ok(n) if (1..=presets.len()).contains(&n) => presets[n - 1].as_str(),
        _ => reference,
    };
    ProviderRef::parse(reference).map_err(|error| error.to_string())
}

fn state(turn: &Turn) -> &'static str {
    match turn {
        Turn::Idle => "idle",
        Turn::Ready | Turn::Thinking(_) => "thinking",
        Turn::Acting => "tools",
    }
}

/// Hard-wrap `text` to `width` columns under a hanging `prefix`.
fn wrap<'a>(prefix: &str, text: &str, width: usize, style: Style, out: &mut Vec<Line<'a>>) {
    let indent = " ".repeat(prefix.chars().count());
    let width = width.saturating_sub(indent.len()).max(1);
    let mut lead = prefix;
    for line in text.lines() {
        let chars: Vec<char> = line.chars().collect();
        let rows: Vec<String> = if chars.is_empty() {
            vec![String::new()]
        } else {
            chars.chunks(width).map(|chunk| chunk.iter().collect()).collect()
        };
        for row in rows {
            out.push(Line::from(Span::styled(format!("{lead}{row}"), style)));
            lead = &indent;
        }
    }
}

fn draw(
    mut tui: NonSendMut<Tui>,
    focused: Query<(Ref<Transcript>, &Model, &Inbox, &Turn, &Session), With<Focused>>,
    changed: Query<(), Or<(Changed<Model>, Changed<Inbox>, Changed<Turn>, Added<Focused>)>>,
    agents: Query<(), With<Agent>>,
    reloading: Res<Reloading>,
    env: Res<Env>,
) {
    let tui = &mut *tui;
    tui.frames += 1;
    let Ok((transcript, model, inbox, turn, session)) = focused.single() else {
        return;
    };
    let spinning = !turn.is_idle() || reloading.busy();
    if !(tui.dirty || transcript.is_changed() || !changed.is_empty() || spinning && tui.frames % 6 == 0) {
        return;
    }
    tui.dirty = false;
    let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"][(tui.frames / 6 % 10) as usize];
    let busy = if spinning { format!("{spinner} ") } else { String::new() };
    let building = if reloading.busy() { " | building" } else { "" };
    let status = format!(
        " {} | {busy}{}{building} | {} | {} agents | BRP :{} | Esc interrupt · /model /reload /sessions /quit",
        model.0,
        state(turn),
        session.0,
        agents.iter().count(),
        env.port
    );
    let (input, expanded, scroll) = (tui.input.clone(), tui.expanded, &mut tui.scroll);
    let _ = tui.terminal.draw(|frame| {
        let queued: Vec<Line> = inbox
            .steering
            .iter()
            .map(|q| Line::styled(format!("steering: {q}"), Style::new().fg(Color::Magenta)))
            .chain(
                inbox
                    .follow_ups
                    .iter()
                    .map(|q| Line::styled(format!("queued: {q}"), Style::new().fg(Color::Magenta))),
            )
            .collect();
        let [body, queue, prompt, bar] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(queued.len().min(4) as u16),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        let width = body.width as usize;
        let mut lines = Vec::new();
        for entry in &transcript.0 {
            let (prefix, style) = match entry.kind {
                EntryKind::User => ("> ", Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                EntryKind::Assistant => ("", Style::new()),
                EntryKind::Call => ("→ ", Style::new().fg(Color::Yellow)),
                EntryKind::Result => ("  ", Style::new().fg(Color::DarkGray)),
                EntryKind::Info => ("· ", Style::new().fg(Color::Green)),
                EntryKind::Error => ("! ", Style::new().fg(Color::Red)),
            };
            let text = if entry.kind == EntryKind::Result && !expanded {
                let shown: Vec<&str> = entry.text.lines().take(8).collect();
                let more = entry.text.lines().count().saturating_sub(shown.len());
                let mut text = shown.join("\n");
                if more > 0 {
                    text.push_str(&format!("\n… {more} more lines (Ctrl+O)"));
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
        frame.render_widget(Paragraph::new(queued), queue);

        let inner = prompt.width.saturating_sub(2) as usize;
        let chars: Vec<char> = input.chars().collect();
        let shown: String = chars[chars.len().saturating_sub(inner.saturating_sub(1))..].iter().collect();
        frame.render_widget(Paragraph::new(shown.clone()).block(Block::bordered()), prompt);
        frame.set_cursor_position((prompt.x + 1 + shown.chars().count() as u16, prompt.y + 1));
        frame.render_widget(
            Paragraph::new(status).style(Style::new().bg(Color::DarkGray).fg(Color::White)),
            bar,
        );
    });
}
