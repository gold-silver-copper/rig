//! The system prompt, built like pi's: a preamble, then tagged sections for
//! the tools, the rules, and whatever plugins add. It holds no absolute
//! paths, times or random values, so the same setup sends the same request
//! and recorded sessions replay.

use std::collections::BTreeMap;

use bevy::prelude::*;

use crate::glue::{Agent, AgentSet, Tools};
use crate::tools::CONTRIBUTIONS;

/// What plugins add to the prompt.
#[derive(Resource, Default)]
pub struct Prompt {
    /// Tool name to its one-line snippet and guidelines. Only tools that are
    /// registered are listed.
    pub tools: BTreeMap<String, (String, Vec<String>)>,
    /// Extra tagged sections, in order.
    pub sections: Vec<(String, String)>,
}

impl Prompt {
    pub fn tool(&mut self, name: &str, snippet: &str, guidelines: &[&str]) {
        self.tools.insert(
            name.to_owned(),
            (snippet.to_owned(), guidelines.iter().map(|g| g.to_string()).collect()),
        );
    }

    pub fn section(&mut self, name: &str, text: impl Into<String>) {
        self.sections.push((name.to_owned(), text.into()));
    }

    pub fn build(&self, tools: &Tools) -> String {
        let listed: Vec<(&String, &(String, Vec<String>))> = ["read", "bash", "edit", "write"]
            .iter()
            .filter_map(|name| self.tools.get_key_value(*name))
            .chain(
                self.tools
                    .iter()
                    .filter(|(name, _)| !["read", "bash", "edit", "write"].contains(&name.as_str())),
            )
            .filter(|(name, _)| tools.0.contains_key(name.as_str()))
            .collect();
        let tool_lines: Vec<String> = listed
            .iter()
            .map(|(name, (snippet, _))| format!("- {name}: {snippet}"))
            .collect();
        let mut rules: Vec<String> = Vec::new();
        let mut rule = |text: &str| {
            if !rules.iter().any(|r| r == text) {
                rules.push(text.to_owned());
            }
        };
        for (_, (_, guidelines)) in &listed {
            for guideline in guidelines {
                rule(guideline);
            }
        }
        rule("Be concise in your responses");
        rule("Show file paths clearly when working with files");

        let mut prompt = String::from(
            "You are an expert coding assistant operating inside rig-pi, a coding agent harness. \
             You help users by reading files, executing commands, editing code, and writing new files.",
        );
        let tools_text = format!(
            "{}\n\nIn addition to the tools above, you may have access to other custom tools depending on the project.",
            if tool_lines.is_empty() { "(none)".to_owned() } else { tool_lines.join("\n") }
        );
        let rules_text = rules.iter().map(|r| format!("- {r}")).collect::<Vec<_>>().join("\n");
        for (name, text) in [("tools".to_owned(), tools_text), ("rules".to_owned(), rules_text)]
            .into_iter()
            .chain(self.sections.iter().cloned())
        {
            prompt.push_str(&format!("\n\n<{name}>\n{text}\n</{name}>"));
        }
        prompt
    }
}

pub struct PromptPlugin;

impl Plugin for PromptPlugin {
    fn build(&self, app: &mut App) {
        let mut prompt = app.world_mut().get_resource_or_init::<Prompt>();
        for tool in CONTRIBUTIONS {
            prompt.tool(tool.name, tool.snippet, tool.guidelines);
        }
        // After PreUpdate, where plugins change the registry, and before
        // any request of this frame is built.
        app.add_systems(
            Update,
            refresh
                .run_if(|tools: Res<Tools>, prompt: Res<Prompt>| tools.is_changed() || prompt.is_changed())
                .in_set(AgentSet::Route),
        );
    }
}

/// Keep every agent's preamble in step with the registered tools.
fn refresh(prompt: Res<Prompt>, tools: Res<Tools>, mut agents: Query<&mut Agent>) {
    let preamble = prompt.build(&tools);
    for mut agent in &mut agents {
        if agent.preamble != preamble {
            agent.preamble = preamble.clone();
        }
    }
}

/// The prompt for an agent spawned now.
pub fn current(world: &World) -> String {
    match (world.get_resource::<Prompt>(), world.get_resource::<Tools>()) {
        (Some(prompt), Some(tools)) => prompt.build(tools),
        _ => String::new(),
    }
}
