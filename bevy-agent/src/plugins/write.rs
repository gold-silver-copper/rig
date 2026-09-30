use serde_json::{Value, json};

use super::str_arg;
use crate::tools::Native;

pub fn tool() -> Native {
    Native {
        name: "write",
        description: "Create or overwrite a file with the given content.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "content": { "type": "string" }
            },
            "required": ["path", "content"]
        }),
        run,
    }
}

fn run(args: Value) -> Result<String, String> {
    let path = str_arg(&args, "path")?;
    let content = str_arg(&args, "content")?;
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    std::fs::write(path, content).map_err(|error| format!("{path}: {error}"))?;
    Ok(format!("wrote {} bytes to {path}", content.len()))
}
