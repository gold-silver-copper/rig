//! pi's edit: several exact replacements in one file, each matched against
//! the original content. A match may ignore trailing whitespace, curly
//! quotes, dashes and unusual spaces when it covers whole lines.

use bevy::prelude::*;
use rig_core::tool::{DynamicTool, ToolExecutionError, ToolOutput};
use serde::Deserialize;
use serde_json::{Value, json};

use super::arguments;
use crate::glue::AddTool;

pub struct EditPlugin;

impl Plugin for EditPlugin {
    fn build(&self, app: &mut App) {
        app.add_rig_tool(DynamicTool::new(
            "edit",
            "Edit a single file using exact text replacement. Every edits[].oldText must match a \
             unique, non-overlapping region of the original file. If two changes affect the same \
             block or nearby lines, merge them into one edit instead of emitting overlapping \
             edits. Do not include large unchanged regions just to connect distant changes.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
                    "edits": {
                        "type": "array",
                        "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": {"type": "string", "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."},
                                "newText": {"type": "string", "description": "Replacement text for this targeted edit."}
                            },
                            "required": ["oldText", "newText"]
                        }
                    }
                },
                "required": ["path", "edits"]
            }),
            |args: Value| Box::pin(async move { edit(arguments(args)?) }),
        ));
    }
}

#[derive(Deserialize)]
struct Args {
    path: String,
    edits: Vec<Replacement>,
}

#[derive(Deserialize)]
struct Replacement {
    #[serde(rename = "oldText")]
    old_text: String,
    #[serde(rename = "newText")]
    new_text: String,
}

fn edit(args: Args) -> Result<ToolOutput, ToolExecutionError> {
    let fail = |message: String| Err(ToolExecutionError::invalid_args(message));
    let raw = std::fs::read_to_string(&args.path)
        .map_err(|error| ToolExecutionError::not_found(format!("{}: {error}", args.path)))?;
    let crlf = raw.contains("\r\n");
    let content = raw.replace("\r\n", "\n");
    let total = args.edits.len();
    let name = |index: usize| {
        if total == 1 {
            "the text".to_owned()
        } else {
            format!("edits[{index}]")
        }
    };
    let mut spans: Vec<(usize, usize, String)> = Vec::new();
    for (index, replacement) in args.edits.iter().enumerate() {
        let old = replacement.old_text.replace("\r\n", "\n");
        let new = replacement.new_text.replace("\r\n", "\n");
        if old.is_empty() {
            return fail(format!("{}.oldText must not be empty in {}.", name(index), args.path));
        }
        let matches: Vec<usize> = content.match_indices(&old).map(|(at, _)| at).collect();
        let span = match matches.as_slice() {
            [at] => (*at, at + old.len()),
            [] => match fuzzy_lines(&content, &old) {
                Ok(span) => span,
                Err(0) => {
                    return fail(format!(
                        "Could not find {} in {}. The old text must match exactly including all \
                         whitespace and newlines.",
                        name(index),
                        args.path
                    ));
                }
                Err(count) => return fail(duplicate(&name(index), &args.path, count)),
            },
            many => return fail(duplicate(&name(index), &args.path, many.len())),
        };
        spans.push((span.0, span.1, new));
    }
    spans.sort_by_key(|span| span.0);
    for pair in spans.windows(2) {
        if pair[0].1 > pair[1].0 {
            return fail(format!(
                "Two edits overlap in {}. Merge them into one edit or target disjoint regions.",
                args.path
            ));
        }
    }
    let mut edited = content.clone();
    for (start, end, new) in spans.iter().rev() {
        edited.replace_range(*start..*end, new);
    }
    if edited == content {
        return fail(format!(
            "No changes made to {}. The replacement produced identical content.",
            args.path
        ));
    }
    if crlf {
        edited = edited.replace('\n', "\r\n");
    }
    std::fs::write(&args.path, edited).map_err(ToolExecutionError::from_error)?;
    Ok(ToolOutput::text(format!(
        "Successfully replaced {total} block(s) in {}.",
        args.path
    )))
}

fn duplicate(name: &str, path: &str, count: usize) -> String {
    format!(
        "Found {count} occurrences of {name} in {path}. The text must be unique. Please provide \
         more context to make it unique."
    )
}

/// Finds `old` as a run of whole lines, comparing lines loosely. Returns
/// the byte span of those lines, or how many runs matched when not one.
fn fuzzy_lines(content: &str, old: &str) -> Result<(usize, usize), usize> {
    let wanted: Vec<String> = old.trim_end_matches('\n').split('\n').map(loose).collect();
    let mut offset = 0;
    let lines: Vec<(usize, &str)> = content
        .split('\n')
        .map(|line| {
            let start = offset;
            offset += line.len() + 1;
            (start, line)
        })
        .collect();
    let mut found = Vec::new();
    for (index, window) in lines.windows(wanted.len()).enumerate() {
        if window
            .iter()
            .zip(&wanted)
            .all(|((_, line), wanted)| loose(line) == *wanted)
        {
            found.push(index);
        }
    }
    match found.as_slice() {
        [index] => {
            let (start, _) = lines[*index];
            let (last_start, last) = lines[index + wanted.len() - 1];
            let mut end = last_start + last.len();
            // Keep the newline when the old text ended with one.
            if old.ends_with('\n') && end < content.len() {
                end += 1;
            }
            Ok((start, end))
        }
        found => Err(found.len()),
    }
}

fn loose(line: &str) -> String {
    line.trim_end()
        .chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
            '\u{2010}'..='\u{2015}' | '\u{2212}' => '-',
            '\u{00A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
            c => c,
        })
        .collect()
}
