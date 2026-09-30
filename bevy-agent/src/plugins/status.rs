//! The status line under the transcript.

use bevy::prelude::*;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

use crate::agent::{Agent, Turn};
use crate::hot::{HotReload, PatchStatus};
use crate::remote::BrpPort;
use crate::tui::StatusLine;

pub struct StatusPlugin;

impl Plugin for StatusPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_status);
    }
}

fn update_status(
    agent: Res<Agent>,
    hot: Res<HotReload>,
    port: Res<BrpPort>,
    mut status: ResMut<StatusLine>,
) {
    let state = match agent.turn {
        Turn::Idle => "ready",
        Turn::Thinking { .. } => "thinking",
        Turn::Tools { .. } => "running tools",
    };
    let patch = match &hot.status {
        _ if hot.building => "patching...".to_owned(),
        PatchStatus::Unavailable => "hot-patch off".to_owned(),
        PatchStatus::Idle => "hot-patch ready".to_owned(),
        PatchStatus::Applied { count, .. } => format!("patched x{count}"),
        PatchStatus::Failed(_) => "patch failed".to_owned(),
    };
    let usage = agent.usage;
    let tokens = format!(
        "{} in ({} cached) · {} out",
        kilo(usage.input_tokens),
        kilo(usage.cached_input_tokens),
        kilo(usage.output_tokens)
    );
    let dim = Style::new().fg(Color::DarkGray);
    status.0 = Line::from(vec![
        Span::styled(" rigpi ", Style::new().fg(Color::Black).bg(Color::Cyan)),
        Span::styled(
            format!(
                " {} · {state} · {tokens} · {patch} · brp :{} ",
                agent.model_name, port.0
            ),
            dim,
        ),
    ]);
}

fn kilo(tokens: Option<u64>) -> String {
    format!("{:.1}k", tokens.unwrap_or(0) as f64 / 1000.0)
}
