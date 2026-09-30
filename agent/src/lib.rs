//! rig-pi: a minimal coding agent. Bevy runs it, Rig talks to the models,
//! ratatui draws the terminal, and BRP lets other processes plug in.
//!
//! The core is the agent loop ([`glue`]), the tools ([`tools`]), the TUI
//! ([`tui`]) and reload ([`reload`], [`supervisor`]). Everything else is a
//! plugin in [`plugins`] that `--disable` turns off.

pub mod brp;
pub mod glue;
pub mod plugins;
pub mod prompt;
pub mod reload;
pub mod session;
pub mod supervisor;
pub mod tools;
pub mod tui;

use std::path::PathBuf;
use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::log::LogPlugin;
use bevy::prelude::*;

/// Exit code the agent uses to ask the supervisor for a restart into
/// `<state>/next-binary`.
pub const RELOAD_EXIT: i32 = 75;

/// The models `/model` offers by number; the first is the default. Any
/// `vendor[/format]:model` that rig's provider registry resolves also works.
pub fn presets() -> [String; 4] {
    use rig_core::providers::{anthropic, deepseek, gemini, openai};
    [
        format!("openai:{}", openai::GPT_6_1_SOL),
        format!("anthropic:{}", anthropic::completion::CLAUDE_OPUS_5_5),
        format!("{}:{}", gemini::PROVIDER_NAME, gemini::completion::GEMINI_3_8_FLASH),
        format!("deepseek:{}", deepseek::DEEPSEEK_V4_1_FLASH),
    ]
}

/// Where this agent lives, fixed for the life of one process.
#[derive(Resource, Clone, Debug)]
pub struct Env {
    /// The agent's own crate: what `reload` builds.
    pub source: PathBuf,
    /// Sessions, binaries, logs.
    pub state: PathBuf,
    /// BRP HTTP port, stable across reloads.
    pub port: u16,
    /// What the supervisor wants the agent told on startup.
    pub note: Option<String>,
    /// Marker file the agent creates once it is up.
    pub ready_file: Option<PathBuf>,
    /// This process replaces one that reloaded or crashed.
    pub restarted: bool,
}

impl Env {
    /// The environment the supervisor hands a child.
    pub fn from_process() -> Self {
        let var = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
        Self {
            source: var("RIG_PI_SOURCE")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR"))),
            state: var("RIG_PI_STATE")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(".rig-pi")),
            port: var("RIG_PI_PORT").and_then(|port| port.parse().ok()).unwrap_or(0),
            note: var("RIG_PI_NOTE"),
            ready_file: var("RIG_PI_READY").map(PathBuf::from),
            restarted: var("RIG_PI_RESTARTED").is_some(),
        }
    }
}

/// Command-line options the supervisor passes on to the agent.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// The model the TUI session starts on.
    pub model: Option<String>,
    /// Start a new TUI session instead of resuming the last one.
    pub fresh: bool,
    /// Record the TUI session's provider traffic as cassette NAME.
    pub record: Option<String>,
    /// Run the TUI session against cassette NAME, offline.
    pub replay: Option<String>,
    /// MCP servers to connect, as a JSON file.
    pub mcp: Option<PathBuf>,
    /// OTLP/HTTP collector to send spans to.
    pub otlp: Option<String>,
    /// Plugins to turn off, by name.
    pub disable: Vec<String>,
}

pub const USAGE: &str = "usage: rig-pi [--model vendor:model] [--new] [--port N] [--state DIR]
              [--record NAME | --replay NAME] [--mcp FILE] [--otlp URL] [--disable a,b]
       rig-pi eval [NAME...]

  --model    start on this model (default: the saved one, else openai:gpt-6.1-sol)
  --new      start a new session instead of resuming the last one
  --port     BRP HTTP port (default: the saved one, else a free port)
  --state    sessions, binaries and logs (default: ./.rig-pi)
  --record   record this session's provider traffic as cassette NAME
  --replay   replay cassette NAME offline, feeding its recorded prompts
  --mcp      connect the MCP servers listed in FILE (default: ./mcp.json if present)
  --otlp     send spans to this OTLP/HTTP collector (default: $OTEL_EXPORTER_OTLP_ENDPOINT)
  --disable  turn plugins off: greet brp-tools remote mcp durable telemetry cassette evals codemode
  eval       run the evals in ./evals (or the named ones) against their cassettes, offline";

impl Options {
    /// Parse agent options; `extra` receives the ones it does not know.
    pub fn parse(args: &[String], extra: &mut Vec<(String, String)>) -> Result<Self, String> {
        let mut options = Options::default();
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            if arg == "--new" {
                options.fresh = true;
                continue;
            }
            let value = args.next().ok_or_else(|| format!("{arg} needs a value"))?.clone();
            match arg.as_str() {
                "--model" => options.model = Some(value),
                "--record" => options.record = Some(value),
                "--replay" => options.replay = Some(value),
                "--mcp" => options.mcp = Some(PathBuf::from(value)),
                "--otlp" => options.otlp = Some(value),
                "--disable" => options.disable.extend(value.split(',').map(str::to_owned)),
                _ => extra.push((arg.clone(), value)),
            }
        }
        Ok(options)
    }

    /// The options as arguments. A restart drops the ones that only apply to
    /// a launch: the model and session were saved, and a recording ends when
    /// its process does.
    pub fn to_args(&self, launch: bool) -> Vec<String> {
        let mut args = Vec::new();
        let mut push = |flag: &str, value: String| {
            args.push(flag.to_owned());
            args.push(value);
        };
        if launch {
            if let Some(model) = &self.model {
                push("--model", model.clone());
            }
            if let Some(record) = &self.record {
                push("--record", record.clone());
            }
        }
        if let Some(replay) = &self.replay {
            push("--replay", replay.clone());
        }
        if let Some(mcp) = &self.mcp {
            push("--mcp", mcp.display().to_string());
        }
        if let Some(otlp) = &self.otlp {
            push("--otlp", otlp.clone());
        }
        if !self.disable.is_empty() {
            push("--disable", self.disable.join(","));
        }
        if launch && self.fresh {
            args.push("--new".into());
        }
        args
    }

    pub fn enabled(&self, plugin: &str) -> bool {
        !self.disable.iter().any(|name| name == plugin)
    }
}

/// The agent app: the core plugins, then every feature plugin not disabled.
/// Without `tui` it runs headless, for evals and tests.
pub fn app(env: Env, options: &Options, tui: bool) -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_millis(16))))
        .insert_resource(env);
    if options.enabled("telemetry") {
        plugins::telemetry::configure(&mut app, options.otlp.clone());
    }
    app.add_plugins(LogPlugin {
        // `rig` holds Rig's model spans; nothing logs under it.
        filter: "warn,rig_pi=info,rig=info".into(),
        custom_layer: plugins::telemetry::layer,
        ..default()
    })
    .add_plugins((
        glue::RigPlugin,
        prompt::PromptPlugin,
        tools::ToolsPlugin,
        session::SessionPlugin {
            model: options.model.clone(),
            fresh: options.fresh,
        },
        reload::ReloadPlugin,
        brp::BrpPlugin,
    ));
    if tui {
        app.add_plugins((tui::TuiPlugin, supervisor::ReadyPlugin));
    } else {
        app.init_resource::<tui::SlashCommands>().add_message::<tui::SlashCommand>();
    }
    app.add_plugins(plugins::features(options));
    app
}

/// Run the agent app under a supervisor (`--child`) and return its exit code.
pub fn run_child(args: &[String]) -> i32 {
    let options = match Options::parse(args, &mut Vec::new()) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("{error}\n{USAGE}");
            return 2;
        }
    };
    let exit = app(Env::from_process(), &options, true).run();
    tui::restore();
    match exit {
        AppExit::Success => 0,
        AppExit::Error(code) => i32::from(code.get()),
    }
}
