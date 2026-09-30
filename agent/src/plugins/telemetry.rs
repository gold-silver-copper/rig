//! Telemetry: every turn is an `invoke_agent` span holding Rig's model
//! request spans (`rig_core::telemetry`, GenAI semantic conventions) and an
//! `execute_tool` span per tool call. Spans go over OTLP/HTTP (JSON) to the
//! collector named by `--otlp` or `$OTEL_EXPORTER_OTLP_ENDPOINT`.

use bevy::log::BoxedLayer;
use bevy::log::tracing_subscriber::Layer;
use bevy::log::tracing_subscriber::filter::Targets;
use bevy::prelude::*;
use opentelemetry::trace::TracerProvider;
use opentelemetry_otlp::{Protocol, WithExportConfig};
use opentelemetry_sdk::Resource as OtelResource;
use opentelemetry_sdk::trace::SdkTracerProvider;

/// The collector, and the provider exporting to it once the layer exists.
#[derive(Resource)]
pub struct TelemetryConfig {
    endpoint: String,
    provider: Option<SdkTracerProvider>,
}

/// Call before `LogPlugin` is added: telemetry is on if a collector is named.
pub fn configure(app: &mut App, endpoint: Option<String>) {
    let endpoint = endpoint.or_else(|| std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok());
    if let Some(endpoint) = endpoint.filter(|e| !e.is_empty()) {
        app.insert_resource(TelemetryConfig {
            endpoint: endpoint.trim_end_matches('/').to_owned(),
            provider: None,
        });
    }
}

/// `LogPlugin::custom_layer`: the OpenTelemetry layer, if configured.
pub fn layer(app: &mut App) -> Option<BoxedLayer> {
    let mut config = app.world_mut().get_resource_mut::<TelemetryConfig>()?;
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpJson)
        .with_endpoint(format!("{}/v1/traces", config.endpoint))
        .build()
        .map_err(|error| eprintln!("telemetry: {error}"))
        .ok()?;
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(OtelResource::builder().with_service_name("rig-pi").build())
        .build();
    let tracer = provider.tracer("rig-pi");
    config.provider = Some(provider);
    let spans = Targets::new()
        .with_target("rig_pi::agent", bevy::log::Level::INFO)
        .with_target("rig", bevy::log::Level::INFO);
    Some(Box::new(tracing_opentelemetry::layer().with_tracer(tracer).with_filter(spans)))
}

pub struct TelemetryPlugin;

impl Plugin for TelemetryPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Last, flush_on_exit.run_if(resource_exists::<TelemetryConfig>));
    }
}

/// Export what is buffered before the process exits (quit or reload).
fn flush_on_exit(mut exits: MessageReader<AppExit>, config: Res<TelemetryConfig>) {
    if exits.read().next().is_some()
        && let Some(provider) = &config.provider
    {
        let _ = provider.force_flush();
    }
}
