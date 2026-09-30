mod model;
mod native;
mod remote;
mod tools;
mod ui;

use anyhow::{Context, Result, bail};
use bevy::prelude::{App, MinimalPlugins, Res, ResMut, Resource, Update};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::{
    path::PathBuf,
    sync::{Arc, RwLock, mpsc},
    time::Duration,
};

#[derive(Resource)]
pub struct UiState {
    pub lines: Vec<String>,
    pub input: String,
    pub busy: bool,
    pub scroll: u16,
    prompts: tokio::sync::mpsc::UnboundedSender<String>,
}
impl UiState {
    fn submit(&mut self, text: String) -> Result<()> {
        if self.busy {
            bail!("agent is already working");
        }
        if text.trim().is_empty() {
            bail!("empty prompt");
        }
        self.prompts.send(text.clone())?;
        self.lines.push(format!("User: {text}"));
        self.busy = true;
        self.scroll = 0;
        Ok(())
    }
}

#[derive(Resource)]
struct Inbox {
    events: std::sync::Mutex<mpsc::Receiver<model::Event>>,
    host: std::sync::Mutex<mpsc::Receiver<tools::HostCommand>>,
}

fn receive(inbox: Res<Inbox>, mut ui: ResMut<UiState>, mut native: ResMut<native::Native>) {
    if let Ok(events) = inbox.events.lock() {
        for event in events.try_iter() {
            match event {
                model::Event::Line(line) => ui.lines.push(line),
                model::Event::Done => ui.busy = false,
            }
        }
    }
    if let Ok(host) = inbox.host.lock() {
        for command in host.try_iter() {
            match command {
                tools::HostCommand::Native(input, reply) => {
                    let _ = reply.send(native::invoke(&input));
                }
                tools::HostCommand::Patch(reply) => {
                    if let Err(error) = native.rebuild(Some(reply)) {
                        ui.lines.push(format!("Hot patch request: {error:#}"));
                    }
                }
            }
        }
    }
}

struct Options {
    root: PathBuf,
    native: PathBuf,
    model: String,
    port: u16,
    snapshot: Option<PathBuf>,
    endpoint_file: Option<PathBuf>,
}
fn options() -> Result<Options> {
    let mut options = Options {
        root: std::env::current_dir()?,
        native: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("native.rs"),
        model: std::env::var("BEVY_AGENT_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into()),
        port: 0,
        snapshot: None,
        endpoint_file: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!(
                "bevy-coding-agent [--root DIR] [--native FILE] [--model NAME] [--port FREE_PORT] [--snapshot FILE] [--endpoint-file FILE]\nRequires OPENAI_API_KEY, rustc and a terminal. BRP binds loopback; omitted port picks a free port. Native code and BRP peers are trusted."
            );
            std::process::exit(0);
        }
        let value = args.next().context("missing option value")?;
        match arg.as_str() {
            "--root" => options.root = PathBuf::from(value).canonicalize()?,
            "--native" => options.native = PathBuf::from(value).canonicalize()?,
            "--model" => options.model = value,
            "--port" => options.port = value.parse()?,
            "--snapshot" => options.snapshot = Some(value.into()),
            "--endpoint-file" => options.endpoint_file = Some(value.into()),
            _ => bail!("unknown option {arg}"),
        }
    }
    let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, options.port))?;
    options.port = listener.local_addr()?.port();
    Ok(options)
}

fn main() -> Result<()> {
    let options = options()?;
    let endpoint = format!("http://127.0.0.1:{}/", options.port);
    let native = native::Native::new(options.native)?;
    let extensions: tools::Extensions = Arc::new(RwLock::new(Default::default()));
    let (host_tx, host_rx) = mpsc::channel();
    let tools = tools::Tools {
        root: options.root,
        native_source: native.source.clone(),
        host: host_tx,
        extensions: extensions.clone(),
    };
    let (prompts, events, runtime) = model::start(tools, options.model)?;
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(native::NativePlugin).add_plugins(remote::plugins(options.port))
        .insert_resource(native).insert_resource(remote::Registry(extensions))
        .insert_resource(Inbox { events: std::sync::Mutex::new(events), host: std::sync::Mutex::new(host_rx) })
        .insert_resource(UiState { lines: vec!["Ready. Native source edits are compiled automatically. BRP peers can register model tools.".into()], input: String::new(), busy: false, scroll: 0, prompts })
        .add_systems(Update, receive);
    app.finish();
    app.cleanup();
    let _guard = ui::TerminalGuard::enter()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(std::io::stdout()))?;
    if let Some(path) = options.endpoint_file {
        std::fs::write(path, &endpoint)?;
    }
    loop {
        app.update();
        let state = app.world().resource::<UiState>();
        let native = app.world().resource::<native::Native>();
        let frame = terminal.draw(|frame| ui::render(frame, state, &native.status, &endpoint))?;
        if let Some(path) = &options.snapshot {
            let mut text = String::new();
            for y in 0..frame.area.height {
                for x in 0..frame.area.width {
                    text.push_str(frame.buffer[(x, y)].symbol());
                }
                text.push('\n');
            }
            std::fs::write(path, text)?;
        }
        if event::poll(Duration::from_millis(33))?
            && let Event::Key(key) = event::read()?
        {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                break;
            }
            let mut state = app.world_mut().resource_mut::<UiState>();
            match key.code {
                KeyCode::Enter => {
                    let text = std::mem::take(&mut state.input);
                    if let Err(error) = state.submit(text) {
                        state.lines.push(format!("Error: {error:#}"));
                    }
                }
                KeyCode::Backspace => {
                    state.input.pop();
                }
                KeyCode::PageUp => state.scroll = state.scroll.saturating_add(10),
                KeyCode::PageDown => state.scroll = state.scroll.saturating_sub(10),
                KeyCode::Char(c) => state.input.push(c),
                _ => {}
            }
        }
    }
    terminal.show_cursor()?;
    // Cancel network/tool futures rather than waiting indefinitely at shutdown.
    runtime.shutdown_timeout(Duration::from_secs(1));
    Ok(())
}
