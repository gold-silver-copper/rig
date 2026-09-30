//! Telemetry: exports the agent's tracing spans over OTLP/HTTP (JSON). The
//! glue opens an `invoke_agent` span per turn and an `execute_tool` span per
//! tool call; Rig's own GenAI spans for each model request nest under them.
//! Without this plugin those spans go nowhere.

use bevy::prelude::*;
use opentelemetry::trace::TracerProvider;
use opentelemetry_otlp::{Protocol, WithExportConfig};
use opentelemetry_sdk::Resource as OtelResource;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

pub struct TelemetryPlugin {
    /// The collector's base URL, such as `http://127.0.0.1:4318`.
    pub endpoint: String,
}

#[derive(Resource)]
struct Exporter(SdkTracerProvider);

impl Plugin for TelemetryPlugin {
    fn build(&self, app: &mut App) {
        let exporter = match opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_protocol(Protocol::HttpJson)
            .with_endpoint(format!("{}/v1/traces", self.endpoint.trim_end_matches('/')))
            .build()
        {
            Ok(exporter) => exporter,
            Err(error) => {
                eprintln!("telemetry is off: {error}");
                return;
            }
        };
        let provider = SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .with_resource(
                OtelResource::builder()
                    .with_service_name("bevy-agent")
                    .build(),
            )
            .build();
        let layer = tracing_opentelemetry::layer().with_tracer(provider.tracer("bevy-agent"));
        let filter = tracing_subscriber::EnvFilter::try_from_env("BEVY_AGENT_TRACE")
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
        if tracing_subscriber::registry()
            .with(filter)
            .with(layer)
            .try_init()
            .is_err()
        {
            eprintln!("telemetry is off: another tracing subscriber is installed");
            return;
        }
        app.insert_resource(Exporter(provider))
            .add_systems(Last, flush_on_exit);
    }
}

fn flush_on_exit(exits: MessageReader<AppExit>, exporter: Res<Exporter>) {
    if !exits.is_empty() {
        let _ = exporter.0.force_flush();
        let _ = exporter.0.shutdown();
    }
}
