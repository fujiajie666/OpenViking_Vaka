use std::ffi::OsString;

use clap::{ArgAction, Parser, Subcommand, ValueEnum};

use crate::{
    CliContext,
    config::Config,
    error::{Error, Result},
    error_ui, handlers,
    output::OutputFormat,
};

const OPENVIKING_COMPILE_URL: &str = "https://api.vikingdb.cn-beijing.volces.com/openviking";
const OPENVIKING_API_KEY_ENV: &str = "OPENVIKING_API_KEY";

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputArg {
    Table,
    Json,
}

impl From<OutputArg> for OutputFormat {
    fn from(value: OutputArg) -> Self {
        match value {
            OutputArg::Table => Self::Table,
            OutputArg::Json => Self::Json,
        }
    }
}

/// Read-only OpenViking CLI for cloud agents.
#[derive(Parser)]
#[command(name = "ov")]
#[command(about = "Read-only OpenViking CLI for cloud agents")]
#[command(version = env!("OPENVIKING_CLI_VERSION"))]
#[command(arg_required_else_help = true)]
struct CompileCli {
    /// Choose human-readable table output or machine-readable JSON.
    #[arg(short, long, value_enum, global = true, default_value = "table")]
    output: OutputArg,

    /// Use compact output rendering.
    #[arg(
        long,
        global = true,
        default_value = "true",
        default_missing_value = "true",
        hide = true,
        num_args = 0..=1,
        require_equals = true,
        action = ArgAction::Set,
        value_name = "bool"
    )]
    compact: bool,

    #[command(subcommand)]
    command: ReadOnlyCommand,
}

#[derive(Subcommand)]
enum ReadOnlyCommand {
    /// Read full file content (Level 2).
    Read {
        /// Viking URI.
        #[arg(value_name = "uri")]
        uri: String,
    },

    /// Search file content with a regular expression.
    Grep {
        /// Target URI. Root-level grep is rejected to protect the service.
        #[arg(short, long, default_value = "viking://", value_name = "uri")]
        uri: String,
        /// Excluded URI prefix.
        #[arg(short = 'x', long = "exclude-uri", value_name = "uri")]
        exclude_uri: Option<String>,
        /// Search pattern.
        #[arg(value_name = "pattern")]
        pattern: String,
        /// Match without case sensitivity.
        #[arg(short, long)]
        ignore_case: bool,
        /// Maximum number of results.
        #[arg(
            short = 'n',
            long = "node-limit",
            alias = "limit",
            default_value = "256",
            value_parser = clap::value_parser!(i32).range(0..),
            value_name = "n"
        )]
        node_limit: i32,
        /// Maximum traversal depth.
        #[arg(
            short = 'L',
            long = "level-limit",
            default_value = "10",
            value_name = "n"
        )]
        level_limit: i32,
    },

    /// Search file names with a glob pattern.
    Glob {
        /// Glob pattern.
        #[arg(value_name = "pattern")]
        pattern: String,
        /// Search root URI.
        #[arg(short, long, default_value = "viking://", value_name = "uri")]
        uri: String,
        /// Maximum number of results.
        #[arg(
            short = 'n',
            long = "node-limit",
            alias = "limit",
            default_value = "256",
            value_parser = clap::value_parser!(i32).range(0..),
            value_name = "n"
        )]
        node_limit: i32,
    },

    /// List directory contents.
    Ls {
        /// Viking URI to list.
        #[arg(default_value = "viking://", value_name = "uri")]
        uri: String,
        /// Print paths only.
        #[arg(short, long)]
        simple: bool,
        /// List subdirectories recursively.
        #[arg(short, long)]
        recursive: bool,
        /// Abstract content limit.
        #[arg(
            long = "abs-limit",
            short = 'l',
            default_value = "256",
            value_name = "n"
        )]
        abs_limit: i32,
        /// Include hidden files.
        #[arg(short, long)]
        all: bool,
        /// Maximum number of nodes.
        #[arg(
            short = 'n',
            long = "node-limit",
            alias = "limit",
            default_value = "256",
            value_parser = clap::value_parser!(i32).range(0..),
            value_name = "n"
        )]
        node_limit: i32,
    },

    /// Print a directory tree.
    Tree {
        /// Viking URI to traverse.
        #[arg(value_name = "uri")]
        uri: String,
        /// Abstract content limit.
        #[arg(
            long = "abs-limit",
            short = 'l',
            default_value = "128",
            value_name = "n"
        )]
        abs_limit: i32,
        /// Include hidden files.
        #[arg(short, long)]
        all: bool,
        /// Maximum number of nodes.
        #[arg(
            short = 'n',
            long = "node-limit",
            alias = "limit",
            default_value = "256",
            value_parser = clap::value_parser!(i32).range(0..),
            value_name = "n"
        )]
        node_limit: i32,
        /// Maximum traversal depth.
        #[arg(
            short = 'L',
            long = "level-limit",
            default_value = "3",
            value_name = "n"
        )]
        level_limit: i32,
    },

    /// Run semantic retrieval.
    Find {
        /// Search query.
        #[arg(value_name = "query")]
        query: Option<String>,
        /// Image query: local path, data URI, HTTP URL, or viking:// URI.
        #[arg(long = "image", value_name = "path|uri")]
        image: Option<String>,
        /// Target URI.
        #[arg(short, long, default_value = "", value_name = "uri")]
        uri: String,
        /// Maximum final results returned.
        #[arg(
            short = 'n',
            long = "node-limit",
            alias = "limit",
            default_value = "10",
            value_parser = clap::value_parser!(i32).range(0..),
            value_name = "n"
        )]
        node_limit: i32,
        /// Score threshold.
        #[arg(short, long, value_name = "score")]
        threshold: Option<f64>,
        /// Only include results on or after this time.
        #[arg(long = "after", value_name = "time")]
        after: Option<String>,
        /// Only include results on or before this time.
        #[arg(long = "before", value_name = "time")]
        before: Option<String>,
        /// Only include specific levels (0=abstract, 1=overview, 2=file).
        #[arg(
            short = 'L',
            long = "level",
            value_delimiter = ',',
            value_name = "0,1,2"
        )]
        level: Option<Vec<i32>>,
        /// Only include specific context types.
        #[arg(long = "context-type", value_delimiter = ',', value_name = "type")]
        context_type: Option<Vec<String>>,
        /// Only include results matching all tags.
        #[arg(long = "tags", value_delimiter = ',')]
        tags: Option<Vec<String>>,
    },

    /// Run context-aware retrieval.
    Search {
        /// Search query.
        #[arg(value_name = "query")]
        query: Option<String>,
        /// Image query: local path, data URI, HTTP URL, or viking:// URI.
        #[arg(long = "image", value_name = "path|uri")]
        image: Option<String>,
        /// Target URI.
        #[arg(short, long, default_value = "", value_name = "uri")]
        uri: String,
        /// Session ID for context-aware search.
        #[arg(long, value_name = "id")]
        session_id: Option<String>,
        /// Maximum results per search pass.
        #[arg(
            short = 'n',
            long = "node-limit",
            alias = "limit",
            default_value = "10",
            value_parser = clap::value_parser!(i32).range(0..),
            value_name = "n"
        )]
        node_limit: i32,
        /// Score threshold.
        #[arg(short, long, value_name = "score")]
        threshold: Option<f64>,
        /// Only include results on or after this time.
        #[arg(long = "after", value_name = "time")]
        after: Option<String>,
        /// Only include results on or before this time.
        #[arg(long = "before", value_name = "time")]
        before: Option<String>,
        /// Only include specific levels (0=abstract, 1=overview, 2=file).
        #[arg(
            short = 'L',
            long = "level",
            value_delimiter = ',',
            value_name = "0,1,2"
        )]
        level: Option<Vec<i32>>,
        /// Only include specific context types.
        #[arg(long = "context-type", value_delimiter = ',', value_name = "type")]
        context_type: Option<Vec<String>>,
        /// Only include results matching all tags.
        #[arg(long = "tags", value_delimiter = ',')]
        tags: Option<Vec<String>>,
    },
}

pub(super) async fn run() {
    let args: Vec<OsString> = std::env::args_os().collect();
    let command_display = error_ui::display_command(&args);
    let cli = match CompileCli::try_parse_from(&args) {
        Ok(cli) => cli,
        Err(error) => error.exit(),
    };

    let output_format = OutputFormat::from(cli.output);
    let config = match runtime_config(std::env::var(OPENVIKING_API_KEY_ENV).ok()) {
        Ok(config) => config,
        Err(error) => {
            print_compile_error(&command_display, &error, output_format, cli.compact);
            std::process::exit(2);
        }
    };
    let ctx = CliContext::from_config(
        config,
        output_format,
        cli.compact,
        None,
        None,
        None,
        false,
        None,
        None,
        None,
    );

    let result = match cli.command {
        ReadOnlyCommand::Read { uri } => handlers::handle_read(uri, ctx).await,
        ReadOnlyCommand::Grep {
            uri,
            exclude_uri,
            pattern,
            ignore_case,
            node_limit,
            level_limit,
        } => {
            handlers::handle_grep(
                uri,
                exclude_uri,
                pattern,
                ignore_case,
                node_limit,
                level_limit,
                ctx,
            )
            .await
        }
        ReadOnlyCommand::Glob {
            pattern,
            uri,
            node_limit,
        } => handlers::handle_glob(pattern, uri, node_limit, ctx).await,
        ReadOnlyCommand::Ls {
            uri,
            simple,
            recursive,
            abs_limit,
            all,
            node_limit,
        } => handlers::handle_ls(uri, simple, recursive, abs_limit, all, node_limit, ctx).await,
        ReadOnlyCommand::Tree {
            uri,
            abs_limit,
            all,
            node_limit,
            level_limit,
        } => handlers::handle_tree(uri, abs_limit, all, node_limit, level_limit, ctx).await,
        ReadOnlyCommand::Find {
            query,
            image,
            uri,
            node_limit,
            threshold,
            after,
            before,
            level,
            context_type,
            tags,
        } => {
            handlers::handle_find(
                query,
                uri,
                image,
                node_limit,
                threshold,
                after,
                before,
                level,
                context_type,
                tags,
                ctx,
            )
            .await
        }
        ReadOnlyCommand::Search {
            query,
            image,
            uri,
            session_id,
            node_limit,
            threshold,
            after,
            before,
            level,
            context_type,
            tags,
        } => {
            handlers::handle_search(
                query,
                uri,
                image,
                session_id,
                node_limit,
                threshold,
                after,
                before,
                level,
                context_type,
                tags,
                ctx,
            )
            .await
        }
    };

    if let Err(error) = result {
        print_compile_error(&command_display, &error, output_format, cli.compact);
        std::process::exit(1);
    }
}

fn print_compile_error(command: &str, error: &Error, output_format: OutputFormat, compact: bool) {
    let message = match error {
        Error::Api { message, .. } => message.clone(),
        _ => error.to_string(),
    };

    if matches!(output_format, OutputFormat::Json) {
        let mut error_body = serde_json::json!({
            "code": error.code(),
            "message": message,
        });
        if let Error::Api {
            details: Some(details),
            ..
        } = error
        {
            error_body["details"] = details.clone();
        }
        let body = serde_json::json!({"ok": false, "error": error_body});
        if compact {
            eprintln!("{body}");
        } else {
            eprintln!(
                "{}",
                serde_json::to_string_pretty(&body).unwrap_or_else(|_| body.to_string())
            );
        }
        return;
    }

    eprintln!("{command}\n\nError [{}]: {message}", error.code());
    if matches!(error, Error::Config(_)) {
        eprintln!("\nSet the key with: export {OPENVIKING_API_KEY_ENV}=\"<your-api-key>\"");
    }
}

fn runtime_config(api_key: Option<String>) -> Result<Config> {
    let api_key = api_key
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            Error::Config(format!(
                "{OPENVIKING_API_KEY_ENV} is required. Export it before running ov."
            ))
        })?;
    api_key
        .parse::<reqwest::header::HeaderValue>()
        .map_err(|_| Error::Config(format!("{OPENVIKING_API_KEY_ENV} is not a valid API key")))?;

    Ok(Config {
        url: OPENVIKING_COMPILE_URL.to_string(),
        api_key: Some(api_key),
        echo_command: false,
        ..Config::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn exposes_only_the_seven_read_only_commands() {
        let names = CompileCli::command()
            .get_subcommands()
            .map(|command| command.get_name().to_string())
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            ["read", "grep", "glob", "ls", "tree", "find", "search"]
        );
    }

    #[test]
    fn rejects_mutating_commands() {
        assert!(CompileCli::try_parse_from(["ov", "rm", "viking://resources/a"]).is_err());
        assert!(CompileCli::try_parse_from(["ov", "write", "viking://resources/a"]).is_err());
        assert!(CompileCli::try_parse_from(["ov", "config"]).is_err());
        assert!(CompileCli::try_parse_from(["ov", "list"]).is_err());
    }

    #[test]
    fn runtime_config_uses_only_the_fixed_service_and_supplied_key() {
        let config = runtime_config(Some("  secret  ".to_string())).expect("valid config");

        assert_eq!(config.url, OPENVIKING_COMPILE_URL);
        assert_eq!(config.api_key.as_deref(), Some("secret"));
        assert!(config.root_api_key.is_none());
        assert!(config.account.is_none());
        assert!(config.user.is_none());
        assert!(!config.echo_command);
    }

    #[test]
    fn runtime_config_requires_a_non_empty_api_key() {
        for value in [None, Some(String::new()), Some("  ".to_string())] {
            let error = runtime_config(value).expect_err("missing key must fail");
            assert!(error.to_string().contains(OPENVIKING_API_KEY_ENV));
        }
    }

    #[test]
    fn runtime_config_rejects_a_key_that_cannot_be_sent_as_an_http_header() {
        let error =
            runtime_config(Some("bad\nkey".to_string())).expect_err("invalid key must fail");
        assert!(error.to_string().contains(OPENVIKING_API_KEY_ENV));
    }
}
