use serde_json::{Value, json};

use super::str_arg;
use crate::tools::Native;

pub fn tool() -> Native {
    Native {
        name: "edit",
        description: "Replace one exact, unique occurrence of old_text with new_text in a file."
            .into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "old_text": { "type": "string" },
                "new_text": { "type": "string" }
            },
            "required": ["path", "old_text", "new_text"]
        }),
        run,
    }
}

fn run(args: Value) -> Result<String, String> {
    let path = str_arg(&args, "path")?;
    let old = str_arg(&args, "old_text")?;
    let new = str_arg(&args, "new_text")?;
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    match text.matches(old).count() {
        0 => Err(format!("old_text not found in {path}")),
        1 => {
            std::fs::write(path, text.replacen(old, new, 1)).map_err(|error| error.to_string())?;
            Ok(format!("edited {path}"))
        }
        n => Err(format!(
            "old_text occurs {n} times in {path}; include more context"
        )),
    }
}
