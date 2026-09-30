//! A minimal MCP server over stdio for trying the MCP plugin. It serves one
//! tool, `planet_facts`, which answers from a fixed table.
//!
//! ```sh
//! bevy-agent --mcp "facts=target/debug/examples/mcp_server"
//! ```

use std::io::{BufRead, Write};

use serde_json::{Value, json};

fn main() -> std::io::Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(request) = serde_json::from_str::<Value>(&line?) else {
            continue;
        };
        // Notifications carry no id and get no reply.
        let Some(id) = request.get("id").cloned() else {
            continue;
        };
        let result = match request["method"].as_str().unwrap_or_default() {
            "initialize" => json!({
                "protocolVersion": request["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "planet-facts", "version": "1.0.0"}
            }),
            "tools/list" => json!({"tools": [{
                "name": "planet_facts",
                "description": "Facts about a planet of the solar system: its moon count and a catalogue code.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"planet": {"type": "string", "description": "The planet's English name"}},
                    "required": ["planet"]
                }
            }]}),
            "tools/call" => {
                let planet = request["params"]["arguments"]["planet"]
                    .as_str()
                    .unwrap_or_default()
                    .to_lowercase();
                let text = match planet.as_str() {
                    "mars" => "Mars has 2 moons. Catalogue code: MCP-ARES-7731.",
                    "jupiter" => "Jupiter has 97 moons. Catalogue code: MCP-ZEUS-4410.",
                    "earth" => "Earth has 1 moon. Catalogue code: MCP-GAIA-0001.",
                    _ => "Unknown planet.",
                };
                json!({"content": [{"type": "text", "text": text}], "isError": false})
            }
            "ping" => json!({}),
            _ => {
                let reply = json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "method not found"}});
                writeln!(stdout, "{reply}")?;
                stdout.flush()?;
                continue;
            }
        };
        writeln!(
            stdout,
            "{}",
            json!({"jsonrpc": "2.0", "id": id, "result": result})
        )?;
        stdout.flush()?;
    }
    Ok(())
}
