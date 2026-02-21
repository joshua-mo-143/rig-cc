use anyhow::Result;
use rig::{completion::ToolDefinition, tool::Tool};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{self, Write};
use tokio::time::{Duration, timeout};

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ToolError(String);

#[derive(Deserialize)]
pub struct ReadFileArgs {
    path: String,
}

#[derive(Deserialize, Serialize)]
pub struct ReadFile;

impl Tool for ReadFile {
    const NAME: &'static str = "read_file";
    type Error = ToolError;
    type Args = ReadFileArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".to_string(),
            description: "Read the contents of a file at the specified path. Returns the file contents as a string.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read"
                    }
                },
                "required": ["path"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        std::fs::read_to_string(&args.path).map_err(|e| {
            ToolError(format!(
                "Failed to read file '{path}': {e}",
                path = args.path
            ))
        })
    }
}

#[derive(Deserialize)]
pub struct WriteFileArgs {
    path: String,
    content: String,
}

#[derive(Deserialize, Serialize)]
pub struct WriteFile;

impl Tool for WriteFile {
    const NAME: &'static str = "write_file";
    type Error = ToolError;
    type Args = WriteFileArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "write_file".to_string(),
            description: "Write content to a file at the specified path. Creates parent directories if they don't exist. Overwrites the file if it already exists.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Some(parent) = std::path::Path::new(&args.path).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| ToolError(format!("Failed to create directories: {}", e)))?;
        }

        std::fs::write(&args.path, &args.content)
            .map_err(|e| ToolError(format!("Failed to write file '{}': {}", args.path, e)))?;

        Ok(format!(
            "Successfully wrote {} bytes to '{}'",
            args.content.len(),
            args.path
        ))
    }
}

#[derive(Deserialize)]
pub struct BashArgs {
    command: String,
}

#[derive(Deserialize, Serialize)]
pub struct Bash;

impl Bash {
    async fn run(&self, args: BashArgs) -> Result<String, ToolError> {
        use tokio::process::Command;

        let mut child = Command::new("bash")
            .arg("-c")
            .arg(&args.command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| ToolError(format!("Failed to spawn command: {}", e)))?;

        let warning_duration = Duration::from_secs(WARNING_TIMEOUT_SECS);
        let status = match timeout(warning_duration, child.wait()).await {
            Ok(result) => result.map_err(|e| ToolError(format!("Command failed: {}", e)))?,
            Err(_) => {
                eprintln!(
                    "\n[Command running for >{}s. Press Ctrl+C to interrupt]",
                    WARNING_TIMEOUT_SECS
                );
                io::stderr().flush().ok();
                child
                    .wait()
                    .await
                    .map_err(|e| ToolError(format!("Command failed: {}", e)))?
            }
        };

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let mut stdout_content = String::new();
        let mut stderr_content = String::new();

        if let Some(mut stdout) = stdout {
            use tokio::io::AsyncReadExt;
            let mut buf = Vec::new();
            stdout.read_to_end(&mut buf).await.ok();
            stdout_content = String::from_utf8_lossy(&buf).to_string();
        }

        if let Some(mut stderr) = stderr {
            use tokio::io::AsyncReadExt;
            let mut buf = Vec::new();
            stderr.read_to_end(&mut buf).await.ok();
            stderr_content = String::from_utf8_lossy(&buf).to_string();
        }

        let mut output = if status.success() {
            let mut out = stdout_content;
            if !stderr_content.is_empty() {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str("stderr:\n");
                out.push_str(&stderr_content);
            }
            out
        } else {
            let mut out = format!("Exit code: {}\n", status.code().unwrap_or(-1));
            if !stdout_content.is_empty() {
                out.push_str("stdout:\n");
                out.push_str(&stdout_content);
                out.push('\n');
            }
            if !stderr_content.is_empty() {
                out.push_str("stderr:\n");
                out.push_str(&stderr_content);
            }
            out
        };

        let total_bytes = output.len();
        if total_bytes > MAX_OUTPUT_BYTES {
            output.truncate(MAX_OUTPUT_BYTES);
            while !output.is_char_boundary(output.len()) {
                output.pop();
            }
            output.push_str(&format!(
                "\n... [output truncated, {} bytes total]",
                total_bytes
            ));
        }

        Ok(output)
    }
}

const MAX_OUTPUT_BYTES: usize = 50 * 1024;
const WARNING_TIMEOUT_SECS: u64 = 60;

impl Tool for Bash {
    const NAME: &'static str = "bash";
    type Error = ToolError;
    type Args = BashArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "bash".to_string(),
            description: "Execute a bash command and return its output. Use this for running shell commands, git operations, running tests, installing packages, etc. The command runs in the current working directory.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    }
                },
                "required": ["command"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.run(args).await
    }
}
