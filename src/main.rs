use crate::tools::{Bash, ReadFile, WriteFile};
use anyhow::Result;
use futures::StreamExt;
use rig::{
    agent::MultiTurnStreamItem,
    message::Message,
    prelude::*,
    providers::anthropic::Client,
    streaming::{StreamedAssistantContent, StreamingPrompt},
};
use std::io::{self, Write};

pub mod tools;

const SYSTEM_PROMPT: &str = r#"You are Rig Code, an interactive AI coding assistant running in the terminal.

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
        .agent("claude-sonnet-4-5")
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
