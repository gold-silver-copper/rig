use anyhow::{Context, Result, bail};
use rig_core::completion::ToolDefinition;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{Arc, RwLock, mpsc},
};
use tokio::{io::AsyncReadExt, sync::oneshot};

#[derive(Clone, Deserialize, Serialize)]
pub struct Extension {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub url: String,
}
pub type Extensions = Arc<RwLock<BTreeMap<String, Extension>>>;

pub enum HostCommand {
    Native(String, oneshot::Sender<Result<String>>),
    Patch(oneshot::Sender<Result<String>>),
}

#[derive(Clone)]
pub struct Tools {
    pub root: PathBuf,
    pub host: mpsc::Sender<HostCommand>,
    pub extensions: Extensions,
}

fn definition(name: &str, description: &str, properties: Value, required: Value) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: description.into(),
        parameters: json!({"type":"object", "properties":properties, "required":required, "additionalProperties":false}),
    }
}

impl Tools {
    pub fn definitions(&self) -> Result<Vec<ToolDefinition>> {
        let mut tools = vec![
            definition(
                "read_file",
                "Read a UTF-8 file relative to the working directory",
                json!({"path":{"type":"string"}}),
                json!(["path"]),
            ),
            definition(
                "write_file",
                "Write a UTF-8 file, including the editable native plugin source",
                json!({"path":{"type":"string"},"content":{"type":"string"}}),
                json!(["path", "content"]),
            ),
            definition(
                "shell",
                "Run a shell command in the working directory (30s timeout). Never print credentials.",
                json!({"command":{"type":"string"}}),
                json!(["command"]),
            ),
            definition(
                "native",
                "Call the current hot-patchable native plugin; input status reports its version",
                json!({"input":{"type":"string"}}),
                json!(["input"]),
            ),
            definition(
                "patch_native",
                "Compile and apply the native plugin source without restarting. Waits for completion. If a build is already running, retry.",
                json!({}),
                json!([]),
            ),
        ];
        for extension in self
            .extensions
            .read()
            .map_err(|_| anyhow::anyhow!("extension registry poisoned"))?
            .values()
        {
            tools.push(ToolDefinition {
                name: extension.name.clone(),
                description: extension.description.clone(),
                parameters: extension.parameters.clone(),
            });
        }
        Ok(tools)
    }

    pub async fn execute(&self, name: &str, args: Value) -> Result<String> {
        let field = |key: &str| -> Result<&str> {
            args.get(key)
                .and_then(Value::as_str)
                .with_context(|| format!("missing string argument {key}"))
        };
        match name {
            "read_file" => {
                let path = self.root.join(field("path")?);
                let file = tokio::fs::File::open(path).await?;
                let mut bytes = Vec::new();
                file.take(65537).read_to_end(&mut bytes).await?;
                if bytes.len() > 65536 {
                    bail!("file exceeds 64 KiB; use shell to inspect a slice");
                }
                Ok(String::from_utf8(bytes)?)
            }
            "write_file" => {
                let path = self.root.join(field("path")?);
                if let Some(parent) = path.parent() {
                    tokio::fs::create_dir_all(parent).await?;
                }
                tokio::fs::write(path, field("content")?).await?;
                Ok("written".into())
            }
            "shell" => {
                let mut child = tokio::process::Command::new("sh")
                    .arg("-c")
                    .arg(field("command")?)
                    .current_dir(&self.root)
                    .kill_on_drop(true)
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()?;
                let stdout = child.stdout.take().context("shell stdout")?;
                let stderr = child.stderr.take().context("shell stderr")?;
                let collect = async {
                    let (out, err) = tokio::try_join!(drain(stdout), drain(stderr))?;
                    let status = child.wait().await?;
                    Ok::<_, anyhow::Error>(format!(
                        "exit={status}\n{}{}",
                        String::from_utf8_lossy(&out),
                        String::from_utf8_lossy(&err)
                    ))
                };
                tokio::time::timeout(std::time::Duration::from_secs(30), collect)
                    .await
                    .context("shell timed out")?
            }
            "native" | "patch_native" => {
                let (tx, rx) = oneshot::channel();
                let command = if name == "native" {
                    HostCommand::Native(field("input")?.into(), tx)
                } else {
                    HostCommand::Patch(tx)
                };
                self.host.send(command)?;
                tokio::time::timeout(std::time::Duration::from_secs(60), rx).await??
            }
            _ => {
                let extension = self
                    .extensions
                    .read()
                    .map_err(|_| anyhow::anyhow!("extension registry poisoned"))?
                    .get(name)
                    .cloned()
                    .context("unknown tool")?;
                let response: Value = reqwest::Client::new()
                    .post(extension.url)
                    .timeout(std::time::Duration::from_secs(30))
                    .json(&json!({"name":name,"arguments":args}))
                    .send()
                    .await?
                    .error_for_status()?
                    .json()
                    .await?;
                Ok(serde_json::to_string(&response)?)
            }
        }
    }
}

async fn drain(mut reader: impl tokio::io::AsyncRead + Unpin) -> std::io::Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let count = reader.read(&mut buffer).await?;
        if count == 0 {
            return Ok(output);
        }
        let keep = count.min(16384_usize.saturating_sub(output.len()));
        output.extend(buffer.iter().take(keep));
    }
}

pub fn validate_extension(extension: &Extension) -> Result<()> {
    if extension.name.is_empty()
        || extension.name.len() > 64
        || !extension
            .name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        bail!("tool name must be 1..64 ASCII letters, digits or underscores");
    }
    if ["read_file", "write_file", "shell", "native", "patch_native"]
        .contains(&extension.name.as_str())
    {
        bail!("cannot replace a built-in tool");
    }
    if extension.parameters.get("type").and_then(Value::as_str) != Some("object") {
        bail!("parameters must be an object JSON schema");
    }
    let url = reqwest::Url::parse(&extension.url)?;
    if url.scheme() != "http"
        || !matches!(url.host_str(), Some("127.0.0.1" | "[::1]"))
        || url.port().is_none()
    {
        bail!("callback must use an explicit loopback HTTP port");
    }
    Ok(())
}

#[cfg(test)]
mod tests;
