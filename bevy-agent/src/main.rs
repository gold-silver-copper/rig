//! rigpi: a minimal coding agent on Rig and Bevy that hot-patches its own
//! native plugins and takes out-of-process plugins over BRP.

mod agent;
mod hot;
mod plugins;
mod remote;
mod tools;
mod tui;

use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;
use rig_core::providers::registry::{Format, ProviderRef};

const USAGE: &str = "usage: rigpi [--model provider:model] [--brp-port PORT]

  --model     a Rig provider reference (default: $RIGPI_MODEL, else
              anthropic:claude-haiku-4-5 or openai:gpt-5-mini by available key)
  --brp-port  Bevy Remote Protocol port on 127.0.0.1 (default: $RIGPI_BRP_PORT,
              else a free port)
  RIGPI_NO_HOTPATCH=1 runs without hot-patching.";

fn main() -> AppExit {
    if let Some(code) = rigpi_hotpatch::intercept() {
        std::process::exit(code);
    }
    let options = match Options::parse() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    let model = ProviderRef::parse(&options.model)
        .map_err(|error| error.to_string())
        .and_then(|reference| {
            let anthropic = reference
                .id()
                .is_some_and(|id| id.format() == Format::Anthropic);
            let model = reference
                .completion_model()
                .map_err(|error| error.to_string())?;
            Ok((model, anthropic))
        });
    let (model, anthropic) = match model {
        Ok(model) => model,
        Err(error) => {
            eprintln!("rigpi: cannot use model `{}`: {error}", options.model);
            std::process::exit(1);
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("rigpi: cannot start tokio: {error}");
            std::process::exit(1);
        }
    };
    let fat = hot::bootstrap();

    App::new()
        .add_plugins(
            MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
                1.0 / 30.0,
            ))),
        )
        .insert_resource(agent::Agent::new(model, options.model, anthropic, runtime))
        .add_plugins((
            hot::HotReloadPlugin(fat),
            tools::ToolsPlugin,
            agent::AgentPlugin,
            remote::BrpPlugin(remote::pick_port(options.brp_port)),
            tui::TuiPlugin,
            plugins::NativePlugins,
        ))
        .run()
}

struct Options {
    model: String,
    brp_port: Option<u16>,
}

impl Options {
    fn parse() -> Result<Self, String> {
        let default_model = if std::env::var_os("ANTHROPIC_API_KEY").is_some() {
            "anthropic:claude-haiku-4-5"
        } else {
            "openai:gpt-5-mini"
        };
        let mut options = Options {
            model: std::env::var("RIGPI_MODEL").unwrap_or_else(|_| default_model.to_owned()),
            brp_port: std::env::var("RIGPI_BRP_PORT")
                .ok()
                .and_then(|port| port.parse().ok()),
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => options.model = args.next().ok_or("--model needs a value")?,
                "--brp-port" => {
                    let port = args.next().ok_or("--brp-port needs a value")?;
                    options.brp_port = Some(port.parse().map_err(|_| format!("bad port {port}"))?);
                }
                "-h" | "--help" => return Err("rigpi".into()),
                other => return Err(format!("unknown argument {other}")),
            }
        }
        Ok(options)
    }
}
