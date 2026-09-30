fn main() {
    let _ = bevy_app::App::new();
    let _ = bevy_remote::RemotePlugin::default();
    let _ = ratatui::text::Text::raw("");
    let _ = rig_core::completion::CompletionRequest::new("x");
}
