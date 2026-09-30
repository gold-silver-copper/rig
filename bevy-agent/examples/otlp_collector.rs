//! A minimal OTLP/HTTP (JSON) trace collector for trying the telemetry
//! plugin: it prints each span it receives.
//!
//! ```sh
//! cargo run --example otlp_collector -- <port>
//! ```

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;

use serde_json::Value;

fn main() -> std::io::Result<()> {
    let port = std::env::args().nth(1).unwrap_or_else(|| "4318".into());
    let listener = TcpListener::bind(("127.0.0.1", port.parse::<u16>().unwrap_or(4318)))?;
    println!("listening on {}", listener.local_addr()?);
    for stream in listener.incoming() {
        let mut stream = stream?;
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut length = 0;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 || line == "\r\n" {
                break;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                length = value.trim().parse().unwrap_or(0);
            }
        }
        let mut body = vec![0; length];
        reader.read_exact(&mut body)?;
        if let Ok(export) = serde_json::from_slice::<Value>(&body) {
            print_spans(&export);
        }
        stream.write_all(
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 2\r\n\r\n{}",
        )?;
    }
    Ok(())
}

fn print_spans(export: &Value) {
    let spans = export["resourceSpans"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|resource| resource["scopeSpans"].as_array().into_iter().flatten())
        .flat_map(|scope| scope["spans"].as_array().into_iter().flatten());
    for span in spans {
        let attributes: Vec<String> = span["attributes"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|attribute| {
                let key = attribute["key"].as_str()?;
                let keep = [
                    "gen_ai.operation.name",
                    "gen_ai.provider.name",
                    "gen_ai.request.model",
                    "gen_ai.tool.name",
                    "gen_ai.usage.output_tokens",
                ];
                if !keep.contains(&key) {
                    return None;
                }
                let value = attribute["value"]
                    .as_object()?
                    .values()
                    .next()
                    .map(|value| value.to_string())?;
                Some(format!("{key}={value}"))
            })
            .collect();
        println!(
            "span {} trace={} parent={} {}",
            span["name"].as_str().unwrap_or("?"),
            span["traceId"].as_str().unwrap_or("").get(..8).unwrap_or(""),
            span["parentSpanId"].as_str().unwrap_or("-"),
            attributes.join(" ")
        );
    }
}
