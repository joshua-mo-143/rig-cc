use anyhow::Result;
use futures::StreamExt;
use rig::{
    agent::MultiTurnStreamItem,
    completion::ToolDefinition,
    message::Message,
    prelude::*,
    providers::anthropic::{self, Client},
    streaming::{StreamedAssistantContent, StreamingPrompt},
    tool::Tool,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{self, Write};
use tokio::time::{Duration, timeout};

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct ToolError(String);

#[derive(Deserialize)]
struct ReadFileArgs {
    path: String,
}

#[derive(Deserialize, Serialize)]
struct ReadFile;

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
        std::fs::read_to_string(&args.path)
            .map_err(|e| ToolError(format!("Failed to read file '{}': {}", args.path, e)))
    }
}

#[derive(Deserialize)]
struct WriteFileArgs {
    path: String,
    content: String,
}

#[derive(Deserialize, Serialize)]
struct WriteFile;

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
        if let Some(parent) = std::path::Path::new(&args.path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| ToolError(format!("Failed to create directories: {}", e)))?;
            }
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
struct BashArgs {
    command: String,
}

#[derive(Deserialize, Serialize)]
struct Bash;

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

const SYSTEM_PROMPT: &str = r#"You are Claude Code, an interactive AI coding assistant running in the terminal.

You have access to these tools:
- bash: Execute shell commands (runs in current working directory)
- read_file: Read file contents
- write_file: Create or modify files

Guidelines:
- Use bash to explore projects, run tests, git operations, etc.
- Read files before modifying them to understand context
- Be concise and focused on solving the user's problem
- When making changes, explain what you're doing briefly
"#;

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env();

    let agent = client
        .agent(anthropic::completion::CLAUDE_4_SONNET)
        .preamble(SYSTEM_PROMPT)
        .tool(ReadFile)
        .tool(WriteFile)
        .tool(Bash)
        .max_tokens(8192)
        .build();

    println!("Rig Code v0.1.0");
    println!("Type 'exit' or 'quit' to exit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut history: Vec<Message> = Vec::new();

    loop {
        print!("> ");
        stdout.flush()?;

        let mut input = String::new();
        match stdin.read_line(&mut input) {
            Ok(0) => {
                println!("\nGoodbye!");
                break;
            }
            Ok(_) => {
                let input = input.trim();

                if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
                    println!("Goodbye!");
                    break;
                }

                if input.is_empty() {
                    continue;
                }

                println!();

                let mut stream = agent
                    .stream_prompt(input)
                    .with_history(history.clone())
                    .multi_turn(100)
                    .await;

                let mut response_text = String::new();
                let mut input_tokens = 0u64;
                let mut output_tokens = 0u64;

                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(MultiTurnStreamItem::StreamAssistantItem(
                            StreamedAssistantContent::Text(text),
                        )) => {
                            print!("{}", text.text);
                            stdout.flush()?;
                            response_text.push_str(&text.text);
                        }
                        Ok(MultiTurnStreamItem::StreamAssistantItem(
                            StreamedAssistantContent::ToolCall { tool_call, .. },
                        )) => {
                            println!("\n[Calling tool: {}]", tool_call.function.name);
                            stdout.flush()?;
                        }
                        Ok(MultiTurnStreamItem::StreamUserItem(
                            rig::streaming::StreamedUserContent::ToolResult { tool_result, .. },
                        )) => {
                            println!("[Tool result received for: {}]", tool_result.id);
                            stdout.flush()?;
                        }
                        Ok(MultiTurnStreamItem::FinalResponse(final_response)) => {
                            let usage = final_response.usage();
                            input_tokens = usage.input_tokens;
                            output_tokens = usage.output_tokens;
                        }
                        Err(e) => {
                            eprintln!("\nError: {}", e);
                            break;
                        }
                        _ => {}
                    }
                }

                println!("\n");
                println!(
                    "[Tokens: {} in / {} out]",
                    format_number(input_tokens),
                    format_number(output_tokens)
                );
                println!();

                history.push(Message::user(input));
                if !response_text.is_empty() {
                    history.push(Message::assistant(response_text));
                }
            }
            Err(e) => {
                eprintln!("Error reading input: {}", e);
            }
        }
    }

    Ok(())
}

fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}