//! The core tools, after pi's: `read`, `bash`, `edit` and `write`, with pi's
//! names, arguments, descriptions and limits. Each is a Rig `DynamicTool`.
//! Paths are relative to the working directory.

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bevy::prelude::*;
use rig_core::tool::{DynamicTool, ToolExecutionError, ToolOutput};
use serde_json::{Value, json};

use crate::glue::{Tool, Tools};

pub const MAX_LINES: usize = 2000;
pub const MAX_BYTES: usize = 50 * 1024;

/// How each tool appears in the system prompt: a one-line snippet and the
/// guidelines it contributes, as in pi.
pub struct PromptContribution {
    pub name: &'static str,
    pub snippet: &'static str,
    pub guidelines: &'static [&'static str],
}

pub const CONTRIBUTIONS: &[PromptContribution] = &[
    PromptContribution {
        name: "read",
        snippet: "Read file contents",
        guidelines: &["Use read to examine files instead of cat or sed."],
    },
    PromptContribution {
        name: "bash",
        snippet: "Execute bash commands (ls, grep, find, etc.)",
        guidelines: &[],
    },
    PromptContribution {
        name: "edit",
        snippet: "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
        guidelines: &[
            "Use edit for precise changes (edits[].oldText must match exactly)",
            "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
            "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
            "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
        ],
    },
    PromptContribution {
        name: "write",
        snippet: "Create or overwrite files",
        guidelines: &["Use write only for new files or complete rewrites."],
    },
];

pub struct ToolsPlugin;

impl Plugin for ToolsPlugin {
    fn build(&self, app: &mut App) {
        let mut tools = app.world_mut().get_resource_or_init::<Tools>();
        for tool in [read(), bash(), edit(), write()] {
            tools.add(Tool::Task(tool));
        }
    }
}

/// A tool from a blocking function, run on the async compute pool.
pub fn blocking(
    name: &str,
    description: &str,
    parameters: Value,
    run: impl Fn(Value) -> Result<String, String> + Send + Sync + 'static,
) -> DynamicTool {
    let run = Arc::new(run);
    DynamicTool::new(name, description, parameters, move |args| {
        let run = run.clone();
        Box::pin(async move {
            run(args)
                .map(ToolOutput::text)
                .map_err(ToolExecutionError::other)
        })
    })
}

fn arg<'a>(args: &'a Value, name: &str) -> Result<&'a str, String> {
    args.get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string argument `{name}`"))
}

fn read() -> DynamicTool {
    blocking(
        "read",
        &format!(
            "Read the contents of a file. For text files, output is truncated to {MAX_LINES} lines or {}KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
            MAX_BYTES / 1024
        ),
        json!({"type": "object", "properties": {
            "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
            "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
            "limit": {"type": "number", "description": "Maximum number of lines to read"}
        }, "required": ["path"]}),
        |args| {
            let path = arg(&args, "path")?;
            let text = std::fs::read_to_string(path).map_err(|e| format!("Could not read {path}: {e}"))?;
            let lines: Vec<&str> = text.lines().collect();
            let start = args.get("offset").and_then(Value::as_u64).unwrap_or(1).max(1) as usize;
            if start > lines.len().max(1) {
                return Err(format!("Offset {start} is beyond end of file ({} lines total)", lines.len()));
            }
            let limit = args.get("limit").and_then(Value::as_u64).map(|n| n as usize);
            let wanted = &lines[start - 1..lines.len().min(start - 1 + limit.unwrap_or(usize::MAX))];
            let (mut shown, mut bytes) = (Vec::new(), 0);
            for line in wanted.iter().take(MAX_LINES) {
                if bytes + line.len() + 1 > MAX_BYTES && !shown.is_empty() {
                    break;
                }
                bytes += line.len() + 1;
                shown.push(*line);
            }
            let end = start - 1 + shown.len();
            let mut out = shown.join("\n");
            if shown.len() < wanted.len() {
                out.push_str(&format!(
                    "\n\n[Showing lines {start}-{end} of {}. Use offset={} to continue.]",
                    lines.len(),
                    end + 1
                ));
            }
            Ok(out)
        },
    )
}

fn write() -> DynamicTool {
    blocking(
        "write",
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
        json!({"type": "object", "properties": {
            "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
            "content": {"type": "string", "description": "Content to write to the file"}
        }, "required": ["path", "content"]}),
        |args| {
            let path = arg(&args, "path")?;
            let content = arg(&args, "content")?;
            if let Some(parent) = std::path::Path::new(path).parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            std::fs::write(path, content).map_err(|e| format!("Could not write {path}: {e}"))?;
            Ok(format!("Successfully wrote {} bytes to {path}", content.len()))
        },
    )
}

fn edit() -> DynamicTool {
    blocking(
        "edit",
        "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.",
        json!({"type": "object", "properties": {
            "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
            "edits": {"type": "array", "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.",
                "items": {"type": "object", "properties": {
                    "oldText": {"type": "string", "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."},
                    "newText": {"type": "string", "description": "Replacement text for this targeted edit."}
                }, "required": ["oldText", "newText"]}}
        }, "required": ["path", "edits"]}),
        |args| {
            let path = arg(&args, "path")?;
            let edits: Vec<(String, String)> = args
                .get("edits")
                .and_then(Value::as_array)
                .ok_or("missing array argument `edits`")?
                .iter()
                .map(|e| Ok((arg(e, "oldText")?.to_owned(), arg(e, "newText")?.to_owned())))
                .collect::<Result<_, String>>()?;
            let raw = std::fs::read_to_string(path).map_err(|e| format!("Could not edit file: {path}. {e}."))?;
            let updated = apply_edits(&raw, &edits, path)?;
            std::fs::write(path, updated).map_err(|e| e.to_string())?;
            Ok(format!("Successfully replaced {} block(s) in {path}.", edits.len()))
        },
    )
}

/// Apply every edit against the original text: each `old` must occur once
/// and no two may overlap. Line endings and a BOM are preserved.
pub fn apply_edits(raw: &str, edits: &[(String, String)], path: &str) -> Result<String, String> {
    let (bom, body) = match raw.strip_prefix('\u{feff}') {
        Some(body) => ("\u{feff}", body),
        None => ("", raw),
    };
    let crlf = body.contains("\r\n");
    let text = body.replace("\r\n", "\n");
    let many = edits.len() > 1;
    let which = |i: usize| if many { format!("edits[{i}]") } else { "the text".into() };
    let mut spans = Vec::new();
    for (i, (old, new)) in edits.iter().enumerate() {
        let old = old.replace("\r\n", "\n");
        if old.is_empty() {
            return Err(format!("oldText must not be empty in {path}."));
        }
        let found: Vec<usize> = text.match_indices(&old).map(|(at, _)| at).collect();
        match found.as_slice() {
            [at] => spans.push((*at, *at + old.len(), new.replace("\r\n", "\n"))),
            [] => return Err(format!("Could not find {} in {path}. The old text must match exactly including all whitespace and newlines.", which(i))),
            many_found => return Err(format!("Found {} occurrences of {} in {path}. The text must be unique. Please provide more context to make it unique.", many_found.len(), which(i))),
        }
    }
    spans.sort_by_key(|(start, _, _)| *start);
    if spans.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(format!("Edits overlap in {path}. Merge nearby changes into one edit."));
    }
    let mut out = text.clone();
    for (start, end, new) in spans.into_iter().rev() {
        out.replace_range(start..end, &new);
    }
    if out == text {
        return Err(format!("No changes made to {path}. The replacement produced identical content."));
    }
    Ok(format!("{bom}{}", if crlf { out.replace('\n', "\r\n") } else { out }))
}

fn bash() -> DynamicTool {
    blocking(
        "bash",
        &format!(
            "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last {MAX_LINES} lines or {}KB (whichever is hit first). Optionally provide a timeout in seconds.",
            MAX_BYTES / 1024
        ),
        json!({"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"}
        }, "required": ["command"]}),
        |args| {
            let command = arg(&args, "command")?;
            let timeout = args.get("timeout").and_then(Value::as_f64).map(Duration::from_secs_f64);
            run_bash(command, timeout)
        },
    )
}

fn run_bash(command: &str, timeout: Option<Duration>) -> Result<String, String> {
    let mut child = Command::new("bash")
        .arg("-c")
        .arg(format!("{{ {command}\n}} 2>&1"))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| e.to_string())?;
    // Read on another thread, so a command that outlives its timeout, or
    // leaves a grandchild holding the pipe, cannot block the tool.
    let output = Arc::new(Mutex::new(Vec::new()));
    let reader = child.stdout.take().map(|mut stdout| {
        let output = output.clone();
        std::thread::spawn(move || {
            let mut buf = [0u8; 8192];
            while let Ok(n @ 1..) = stdout.read(&mut buf) {
                if let Ok(mut output) = output.lock() {
                    output.extend_from_slice(&buf[..n]);
                }
            }
        })
    });
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            break Some(status);
        }
        if timeout.is_some_and(|timeout| started.elapsed() > timeout) {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let drained = Instant::now();
    while reader.as_ref().is_some_and(|r| !r.is_finished()) && drained.elapsed() < Duration::from_secs(1) {
        std::thread::sleep(Duration::from_millis(5));
    }
    let text = output.lock().map(|o| String::from_utf8_lossy(&o).into_owned()).unwrap_or_default();
    let text = tail(&text);
    match status {
        Some(status) if status.success() => Ok(if text.is_empty() { "(no output)".into() } else { text }),
        Some(status) => Err(format!("{text}\n\nCommand exited with code {}", status.code().unwrap_or(-1))),
        None => Err(format!("{text}\n\nCommand timed out after {:.0} seconds", timeout.unwrap_or_default().as_secs_f64())),
    }
}

/// The last `MAX_LINES` lines within `MAX_BYTES`, noting what was dropped.
fn tail(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let (mut kept, mut bytes) = (Vec::new(), 0);
    for line in lines.iter().rev().take(MAX_LINES) {
        if bytes + line.len() + 1 > MAX_BYTES && !kept.is_empty() {
            break;
        }
        bytes += line.len() + 1;
        kept.push(*line);
    }
    kept.reverse();
    let mut out = kept.join("\n");
    if kept.len() < lines.len() {
        out = format!("[Showing the last {} of {} lines]\n{out}", kept.len(), lines.len());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::apply_edits;

    fn edits(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }

    #[test]
    fn edits_match_the_original_and_keep_line_endings() {
        let out = apply_edits("a\r\nb\r\nc\r\n", &edits(&[("a", "A"), ("c", "C")]), "f").unwrap();
        assert_eq!(out, "A\r\nb\r\nC\r\n");
    }

    #[test]
    fn edits_must_be_unique_and_disjoint() {
        assert!(apply_edits("x x", &edits(&[("x", "y")]), "f").unwrap_err().contains("2 occurrences"));
        assert!(apply_edits("abc", &edits(&[("ab", "1"), ("bc", "2")]), "f").unwrap_err().contains("overlap"));
        assert!(apply_edits("abc", &edits(&[("zz", "1")]), "f").unwrap_err().contains("Could not find"));
    }
}
