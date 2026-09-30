//! Sprout: a small, trusted-local coding agent hosted by Bevy.
mod model;
mod reload;
mod remote;

use anyhow::{Result, ensure};
use bevy::prelude::*;
use bevy_remote::http::RemoteHttpPlugin;
use crossbeam_channel::{Receiver, Sender};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::{
    layout::{Constraint, Layout},
    widgets::{Block, Paragraph, Wrap},
};
use std::{
    fs,
    net::{Ipv4Addr, TcpListener},
    path::PathBuf,
    time::Duration,
};

#[derive(Resource)]
struct Agent {
    lines: Vec<String>,
    input: String,
    busy: bool,
    prompts: Sender<String>,
    events: Receiver<model::Event>,
}

impl Agent {
    fn log(&mut self, text: String) {
        self.lines.push(text);
        if self.lines.len() > 200 {
            self.lines.remove(0);
        }
    }

    fn submit(&mut self) -> Result<()> {
        if self.busy || self.input.trim().is_empty() {
            return Ok(());
        }
        let prompt = std::mem::take(&mut self.input);
        self.log(format!("you: {prompt}"));
        self.prompts.send(prompt)?;
        self.busy = true;
        Ok(())
    }
}

fn receive(mut agent: ResMut<Agent>) {
    while let Ok(event) = agent.events.try_recv() {
        match event {
            model::Event::Line(text) => agent.log(text),
            model::Event::Done => agent.busy = false,
        }
    }
}

fn draw(frame: &mut ratatui::Frame, agent: &Agent, native: &sprout_native::NativeState, port: u16) {
    let chunks = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(3),
        Constraint::Length(3),
    ])
    .split(frame.area());
    frame.render_widget(Paragraph::new(format!("Sprout | {} | ticks {} | BRP 127.0.0.1:{port}\nEnter: send · Esc/Ctrl-C: quit · trusted shell + native code", native.label, native.ticks)), chunks[0]);
    let text = agent.lines.join("\n");
    // Use the same wrapped paragraph for line counting and drawing.
    let transcript = Paragraph::new(text)
        .wrap(Wrap { trim: false })
        .block(Block::bordered().title("conversation"));
    let scroll = transcript
        .line_count(chunks[1].width.saturating_sub(2))
        .saturating_sub(chunks[1].height.saturating_sub(2) as usize);
    frame.render_widget(
        transcript.scroll((scroll.min(u16::MAX as usize) as u16, 0)),
        chunks[1],
    );
    frame.render_widget(
        Paragraph::new(agent.input.as_str())
            .wrap(Wrap { trim: false })
            .block(Block::bordered().title(if agent.busy {
                "working… (input retained)"
            } else {
                "prompt"
            })),
        chunks[2],
    );
}

fn main() -> Result<()> {
    ensure!(
        cfg!(debug_assertions),
        "hotpatching requires a debug build; run cargo run, not --release"
    );
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::create_dir_all(root.join(".sprout"))?;
    let cwd = std::env::current_dir()?;
    // Never claim the fixed BRP default; choose an available loopback port per process.
    let reservation = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let port = reservation.local_addr()?.port();
    let extensions = remote::Extensions::default();
    let (prompt_tx, prompt_rx) = crossbeam_channel::unbounded();
    let (event_tx, event_rx) = crossbeam_channel::unbounded();
    model::start(prompt_rx, event_tx, extensions.clone(), cwd);
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        bevy::app::hotpatch::HotPatchPlugin,
        sprout_native::NativePlugin,
    ))
    .insert_resource(Agent {
        lines: vec![
            "Ready. OPENAI_API_KEY from environment; SPROUT_MODEL defaults to gpt-4.1-mini.".into(),
        ],
        input: String::new(),
        busy: false,
        prompts: prompt_tx,
        events: event_rx,
    })
    .insert_resource(extensions)
    .insert_resource(reload::Reloader::new(root.clone())?)
    .add_plugins((
        remote::plugin(),
        RemoteHttpPlugin::default()
            .with_address(Ipv4Addr::LOCALHOST)
            .with_port(port),
    ))
    .add_systems(Update, (receive, reload::reload).chain());
    app.finish();
    app.cleanup();
    drop(reservation);
    app.update();
    fs::write(
        root.join(".sprout/session.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"url":format!("http://127.0.0.1:{port}"), "pid":std::process::id()}),
        )?,
    )?;
    let mut terminal = ratatui::init();
    let result = (|| -> Result<()> {
        loop {
            app.update();
            terminal.draw(|frame| {
                draw(
                    frame,
                    app.world().resource::<Agent>(),
                    app.world().resource::<sprout_native::NativeState>(),
                    port,
                )
            })?;
            if event::poll(Duration::from_millis(33))?
                && let Event::Key(key) = event::read()?
            {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                if key.code == KeyCode::Esc
                    || (key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL))
                {
                    break;
                }
                let mut agent = app.world_mut().resource_mut::<Agent>();
                match key.code {
                    KeyCode::Enter => agent.submit()?,
                    KeyCode::Backspace => {
                        agent.input.pop();
                    }
                    KeyCode::Char(c) => agent.input.push(c),
                    _ => {}
                }
            }
        }
        Ok(())
    })();
    ratatui::restore();
    result
}

#[cfg(test)]
mod tests;
