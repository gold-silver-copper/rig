use crate::tools::Tools;
use anyhow::{Result, bail};
use rig_core::{
    completion::CompletionRequest,
    message::{AssistantContent, Message},
    providers::openai::OpenAI,
};
use tokio::sync::mpsc;

pub enum Event {
    Line(String),
    Done,
}

pub fn start(
    tools: Tools,
    model_name: String,
) -> Result<(
    mpsc::UnboundedSender<String>,
    std::sync::mpsc::Receiver<Event>,
    tokio::runtime::Runtime,
)> {
    let provider = OpenAI::from_env()?;
    let model = provider.chat(model_name);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let (prompts, mut rx) = mpsc::unbounded_channel::<String>();
    let (events, output) = std::sync::mpsc::channel();
    runtime.spawn(async move {
        let mut history = Vec::new();
        while let Some(prompt) = rx.recv().await {
            history.push(Message::user(prompt));
            let turn = async {
                for _ in 0..12 {
                    let Some(current) = history.pop() else { bail!("empty conversation"); };
                    let request = CompletionRequest::new(current.clone())
                        .preamble(format!("You are a minimal coding agent. Working directory: {}. Use tools to inspect, edit and test code. Tools run with the user's privileges. Never expose environment credentials. Native plugin source: {}. Edit behavior only; agent_plugin_version MUST keep returning 1 and all C ABI signatures MUST stay unchanged. patch_native applies it in this running process. Be concise.", tools.root.display(), tools.native_source.display()))
                        .messages(history.clone()).tools(tools.definitions()?).max_tokens(2048);
                    // Keep the conversation valid even when a provider call fails.
                    history.push(current);
                    let response = tokio::time::timeout(std::time::Duration::from_secs(90), model.call(request)).await??;
                    history.extend(response.message());
                    let mut calls = Vec::new();
                    for content in &response.choice {
                        match content {
                            AssistantContent::Text(text) if !text.text.is_empty() => { let _ = events.send(Event::Line(format!("Assistant: {}", text.text))); }
                            AssistantContent::ToolCall(call) => calls.push(call.clone()),
                            _ => {}
                        }
                    }
                    if calls.is_empty() { return Ok::<_, anyhow::Error>(()); }
                    for call in calls {
                        let name = call.function.name.as_str();
                        let _ = events.send(Event::Line(format!("Tool: {name}({})", call.function.arguments)));
                        let result = match tools.execute(name, call.function.arguments).await {
                            Ok(output) => output,
                            Err(error) => format!("Tool error: {error:#}"),
                        };
                        let _ = events.send(Event::Line(format!("Result: {result}")));
                        history.push(Message::tool_result(call.id, call.function.name, result));
                    }
                }
                bail!("tool round limit reached (12)")
            }.await;
            if let Err(error) = turn { let _ = events.send(Event::Line(format!("Error: {error:#}"))); }
            let _ = events.send(Event::Done);
        }
    });
    Ok((prompts, output, runtime))
}
