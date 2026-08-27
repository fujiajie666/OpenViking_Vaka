use std::collections::HashMap;
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

/// OpenViking retrieval and content-write CLI for cloud agents.
#[derive(Parser)]
#[command(name = "ov")]
#[command(about = "OpenViking retrieval and content-write CLI for cloud agents")]
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
    command: CompileCommand,
}

/// Commands exposed by the compile CLI; mutations are limited to writing text content.
#[derive(Subcommand)]
enum CompileCommand {
    /// Read full file content (Level 2).
    Read {
        /// Viking URI.
        #[arg(value_name = "uri")]
        uri: String,
    },

    /// Write text content.
    Write {
        /// Viking URI.
        #[arg(value_name = "uri")]
        uri: String,
        /// Content to write.
        #[arg(long, conflicts_with = "from_file", value_name = "text")]
        content: Option<String>,
        /// Read content from a local file.
        #[arg(long = "from-file", conflicts_with = "content", value_name = "path")]
        from_file: Option<String>,
        /// Append instead of replacing the file.
        #[arg(long)]
        append: bool,
        /// Write mode: replace, append, or create (default: replace).
        #[arg(long, value_name = "replace|append|create", conflicts_with = "append")]
        mode: Option<String>,
        /// Wait for async processing to finish.
        #[arg(long, default_value = "false")]
        wait: bool,
        /// Content post-write processing mode.
        #[arg(
            long = "processing-mode",
            default_value = "semantic_and_vectors",
            value_parser = ["semantic_and_vectors", "vectors_only"]
        )]
        processing_mode: String,
        /// Optional wait timeout in seconds.
        #[arg(
            long,
            value_parser = crate::config::parse_positive_timeout,
            value_name = "seconds"
        )]
        timeout: Option<f64>,
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
        CompileCommand::Read { uri } => handlers::handle_read(uri, ctx).await,
        CompileCommand::Write {
            uri,
            content,
            from_file,
            append,
            mode,
            wait,
            processing_mode,
            timeout,
        } => {
            let effective_mode = if let Some(mode) = mode {
                mode
            } else if append {
                "append".to_string()
            } else {
                "replace".to_string()
            };
            handlers::handle_write(
                uri,
                content,
                from_file,
                effective_mode,
                wait,
                timeout,
                processing_mode,
                ctx,
            )
            .await
        }
        CompileCommand::Grep {
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
        CompileCommand::Glob {
            pattern,
            uri,
            node_limit,
        } => handlers::handle_glob(pattern, uri, node_limit, ctx).await,
        CompileCommand::Ls {
            uri,
            simple,
            recursive,
            abs_limit,
            all,
            node_limit,
        } => handlers::handle_ls(uri, simple, recursive, abs_limit, all, node_limit, ctx).await,
        CompileCommand::Tree {
            uri,
            abs_limit,
            all,
            node_limit,
            level_limit,
        } => handlers::handle_tree(uri, abs_limit, all, node_limit, level_limit, ctx).await,
        CompileCommand::Find {
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
                false, // The restricted CLI does not inline matched content in retrieval results.
                ctx,
            )
            .await
        }
        CompileCommand::Search {
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
                false, // The restricted CLI does not inline matched content in retrieval results.
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
    let authorization = format!("Bearer {api_key}");
    authorization
        .parse::<reqwest::header::HeaderValue>()
        .map_err(|_| Error::Config(format!("{OPENVIKING_API_KEY_ENV} is not a valid API key")))?;

    Ok(Config {
        url: OPENVIKING_COMPILE_URL.to_string(),
        api_key: None,
        extra_headers: Some(HashMap::from([(
            reqwest::header::AUTHORIZATION.as_str().to_string(),
            authorization,
        )])),
        echo_command: false,
        ..Config::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn exposes_only_retrieval_and_content_write_commands() {
        let command = CompileCli::command();
        assert_eq!(command.get_name(), "ov");

        let names = command
            .get_subcommands()
            .map(|command| command.get_name().to_string())
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            [
                "read", "write", "grep", "glob", "ls", "tree", "find", "search"
            ]
        );
    }

    #[test]
    fn parses_the_full_write_command_contract() {
        let cli = CompileCli::try_parse_from([
            "ov",
            "write",
            "viking://resources/a.md",
            "--content",
            "hello",
            "--mode",
            "create",
            "--wait",
            "--processing-mode",
            "vectors_only",
            "--timeout",
            "12.5",
        ])
        .expect("write command should parse");

        match cli.command {
            CompileCommand::Write {
                uri,
                content,
                from_file,
                append,
                mode,
                wait,
                processing_mode,
                timeout,
            } => {
                assert_eq!(uri, "viking://resources/a.md");
                assert_eq!(content.as_deref(), Some("hello"));
                assert!(from_file.is_none());
                assert!(!append);
                assert_eq!(mode.as_deref(), Some("create"));
                assert!(wait);
                assert_eq!(processing_mode, "vectors_only");
                assert_eq!(timeout, Some(12.5));
            }
            _ => panic!("expected write command"),
        }
    }

    #[test]
    fn rejects_other_mutating_and_administration_commands() {
        assert!(CompileCli::try_parse_from(["ov", "rm", "viking://resources/a"]).is_err());
        assert!(CompileCli::try_parse_from(["ov", "config"]).is_err());
        assert!(CompileCli::try_parse_from(["ov", "list"]).is_err());
    }

    #[test]
    fn runtime_config_uses_only_the_fixed_service_and_supplied_key() {
        let config = runtime_config(Some("  secret  ".to_string())).expect("valid config");

        assert_eq!(config.url, OPENVIKING_COMPILE_URL);
        assert!(config.api_key.is_none());
        assert_eq!(
            config
                .extra_headers
                .as_ref()
                .and_then(|headers| headers.get(reqwest::header::AUTHORIZATION.as_str()))
                .map(String::as_str),
            Some("Bearer secret")
        );
        assert!(config.root_api_key.is_none());
        assert!(config.account.is_none());
        assert!(config.user.is_none());
        assert!(!config.echo_command);
    }

    #[test]
    fn runtime_config_preserves_vault_placeholder_in_bearer_header() {
        let config = runtime_config(Some("ARK_SECRET_PLACEHOLDER".to_string()))
            .expect("Vault placeholder should be accepted");
        let client = crate::base_client::BaseClient::new(
            &config.url,
            config.api_key.clone(),
            config.account.clone(),
            config.user.clone(),
            config.actor_peer_id.clone(),
            config.timeout,
            config.profile,
            config.extra_headers.clone(),
        );
        let headers = client.build_headers();

        assert_eq!(
            headers
                .get(reqwest::header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer ARK_SECRET_PLACEHOLDER")
        );
        assert!(!headers.contains_key("X-API-Key"));
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
