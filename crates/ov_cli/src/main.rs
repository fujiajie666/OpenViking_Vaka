#![cfg_attr(feature = "compile-cli", allow(dead_code, unused_imports))]

//! OpenViking CLI binary entry point.

mod app;
mod base_client;
mod cli;
mod cli_arg_scan;
mod cli_context;
mod client;
mod commands;
#[cfg(feature = "compile-cli")]
mod compile_cli;
mod config;
mod config_agent;
mod config_command_ui;
mod config_wizard;
mod error;
mod error_ui;
mod handlers;
mod health_ui;
mod help_ui;
mod i18n;
mod openviking_assets;
mod output;
mod status_ui;
mod terminal_ui;
mod theme;
mod tui;
mod utils;

pub use cli_context::CliContext;

#[cfg(feature = "compile-cli")]
#[tokio::main]
async fn main() {
    compile_cli::run().await;
}

#[cfg(not(feature = "compile-cli"))]
#[tokio::main]
async fn main() {
    app::run().await;
}
