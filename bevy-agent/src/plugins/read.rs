use serde_json::{Value, json};

use super::{clip, str_arg};
use crate::tools::Native;

pub fn tool() -> Native {
    Native {
        name: "read",
        description: "Read a text file. Returns numbered lines; use offset/limit for long files."
            .into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "offset": { "type": "integer", "description": "First line, 1-based" },
                "limit": { "type": "integer", "description": "Maximum lines (default 2000)" }
            },
            "required": ["path"]
        }),
        run,
    }
}

fn run(args: Value) -> Result<String, String> {
    let path = str_arg(&args, "path")?;
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let offset = args
        .get("offset")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .max(1) as usize;
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(2000) as usize;
    let numbered: Vec<String> = text
        .lines()
        .enumerate()
        .skip(offset - 1)
        .take(limit)
        .map(|(i, line)| format!("{:>5}  {line}", i + 1))
        .collect();
    Ok(clip(numbered.join("\n"), 60_000))
}
