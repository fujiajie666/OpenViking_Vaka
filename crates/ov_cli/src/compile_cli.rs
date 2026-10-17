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
    /// Read file content (Level 2)
    Read {
        /// Viking URI
        #[arg(value_name = "uri")]
        uri: String,
        /// Starting line offset (0-indexed)
        #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(i64).range(0..), help_heading = "Common options")]
        offset: i64,
        /// Number of lines to read; -1 reads to the end
        #[arg(long, default_value_t = -1, allow_negative_numbers = true, value_parser = clap::value_parser!(i64).range(-1..), help_heading = "Common options")]
        limit: i64,
    },
    /// Write text content to an existing file
    Write {
        /// Viking URI
        #[arg(value_name = "uri")]
        uri: String,
        /// Content to write
        #[arg(
            long,
            conflicts_with = "from_file",
            value_name = "text",
            help_heading = "Common options"
        )]
        content: Option<String>,
        /// Read content from a local file
        #[arg(
            long = "from-file",
            conflicts_with = "content",
            value_name = "path",
            help_heading = "Common options"
        )]
        from_file: Option<String>,
        /// Append instead of replacing the file
        #[arg(long, help_heading = "Common options")]
        append: bool,
        /// Write mode: replace, append, or create (default: replace)
        #[arg(
            long,
            value_name = "replace|append|create",
            conflicts_with = "append",
            help_heading = "Advanced options"
        )]
        mode: Option<String>,
        /// Wait for async processing to finish
        #[arg(long, default_value = "false", help_heading = "Common options")]
        wait: bool,
        /// Content post-write processing mode
        #[arg(
            long = "processing-mode",
            default_value = "semantic_and_vectors",
            value_parser = ["semantic_and_vectors", "vectors_only"],
            help_heading = "Advanced options"
        )]
        processing_mode: String,
        /// Comma-separated k=v retrieval tags to write with the content
        #[arg(long = "tags", value_delimiter = ',')]
        tags: Vec<String>,
        /// Tag update mode; clear does not require --tags
        #[arg(long = "tag-mode", default_value = "replace", value_parser = ["replace", "append", "clear"])]
        tag_mode: String,
        /// ACL JSON, for example {"acl_mode":"restricted","entries":[]}
        #[arg(long, value_parser = |s: &str| serde_json::from_str::<serde_json::Value>(s))]
        acl: Option<serde_json::Value>,
    },
    /// Run content pattern search
    Grep {
        /// Target URI
        #[arg(
            short,
            long,
            default_value = "viking://",
            value_name = "uri",
            help_heading = "Common options"
        )]
        uri: String,
        /// Excluded URI range. Any entry whose URI falls under this URI prefix is skipped
        #[arg(
            short = 'x',
            long = "exclude-uri",
            value_name = "uri",
            help_heading = "Advanced options"
        )]
        exclude_uri: Option<String>,
        /// Search pattern
        #[arg(value_name = "pattern")]
        pattern: String,
        /// Case insensitive
        #[arg(short, long, help_heading = "Common options")]
        ignore_case: bool,
        /// Number of lines to show after each match
        #[arg(
            short = 'a',
            long = "after-context",
            default_value = "0",
            value_parser = clap::value_parser!(i32).range(0..),
            value_name = "n",
            help_heading = "Common options"
        )]
        after_context: i32,
        /// Number of lines to show before each match
        #[arg(
            short = 'b',
            long = "before-context",
            default_value = "0",
            value_parser = clap::value_parser!(i32).range(0..),
            value_name = "n",
            help_heading = "Common options"
        )]
        before_context: i32,
        /// Maximum number of results
        #[arg(
            short = 'n',
            long = "node-limit",
            alias = "limit",
            default_value = "256",
            value_parser = clap::value_parser!(i32).range(0..),
            value_name = "n",
            help_heading = "Common options"
        )]
        node_limit: i32,
        /// Maximum depth level to traverse (default: 10)
        #[arg(
            short = 'L',
            long = "level-limit",
            default_value = "10",
            value_name = "n",
            help_heading = "Advanced options"
        )]
        level_limit: i32,
        /// Comma-separated k=v retrieval tags; all tags must match
        #[arg(
            long = "tags",
            value_delimiter = ',',
            value_name = "k=v",
            help_heading = "Common options"
        )]
        tags: Vec<String>,
        /// Fields to include in output (currently: tags)
        #[arg(
            short = 'f',
            long = "fields",
            value_delimiter = ',',
            value_name = "FIELDS",
            help_heading = "Output options"
        )]
        fields: Option<Vec<String>>,
    },
    /// Run file glob pattern search
    Glob {
        /// Glob pattern
        #[arg(value_name = "pattern")]
        pattern: String,
        /// Search root URI
        #[arg(
            short,
            long,
            default_value = "viking://",
            value_name = "uri",
            help_heading = "Common options"
        )]
        uri: String,
        /// Maximum number of results
        #[arg(
            short = 'n',
            long = "node-limit",
            alias = "limit",
            default_value = "256",
            value_parser = clap::value_parser!(i32).range(0..),
            value_name = "n",
            help_heading = "Common options"
        )]
        node_limit: i32,
        /// Simple output (one entry per line)
        #[arg(short, long, help_heading = "Common options")]
        simple: bool,
        /// Comma-separated k=v retrieval tags; all tags must match
        #[arg(
            long = "tags",
            value_delimiter = ',',
            value_name = "k=v",
            help_heading = "Common options"
        )]
        tags: Vec<String>,
        /// Comma-separated fields to display (name,uri,path,type,size,mode,mtime,locked,id,tags)
        #[arg(
            short = 'f',
            long = "fields",
            value_delimiter = ',',
            value_name = "FIELDS",
            help_heading = "Output options"
        )]
        fields: Option<Vec<String>>,
    },
    /// List directory contents
    #[command(alias = "list")]
    Ls {
        /// Viking URI to list (default: viking://)
        #[arg(default_value = "viking://", value_name = "uri")]
        uri: String,
        /// Simple path output (just paths, no table)
        #[arg(short, long, help_heading = "Common options")]
        simple: bool,
        /// List all subdirectories recursively
        #[arg(short, long, help_heading = "Common options")]
        recursive: bool,
        /// Abstract content limit (only for agent output)
        #[arg(
            long = "abs-limit",
            short = 'l',
            default_value = "256",
            value_name = "n",
            help_heading = "Advanced options"
        )]
        abs_limit: i32,
        /// Include directory L0 abstracts (defaults to the selected output mode)
        #[arg(
            long = "include-abstract",
            default_missing_value = "true",
            num_args = 0..=1,
            require_equals = true,
            action = ArgAction::Set,
            value_name = "bool",
            help_heading = "Output options"
        )]
        include_abstract: Option<bool>,
        /// Include directory L1 overviews
        #[arg(
            long = "include-overview",
            default_missing_value = "true",
            num_args = 0..=1,
            require_equals = true,
            action = ArgAction::Set,
            value_name = "bool",
            help_heading = "Output options"
        )]
        include_overview: Option<bool>,
        /// Maximum overview content length
        #[arg(
            long = "overview-limit",
            default_value = "4000",
            value_parser = clap::value_parser!(i32).range(1..),
            value_name = "n",
            help_heading = "Advanced options"
        )]
        overview_limit: i32,
        /// Show all hidden files
        #[arg(short, long, help_heading = "Common options")]
        all: bool,
        /// Maximum number of nodes to list
        #[arg(
            long = "node-limit",
            short = 'n',
            default_value = "256",
            value_parser = clap::value_parser!(i32).range(0..),
            value_name = "n",
            help_heading = "Common options"
        )]
        node_limit: i32,
        /// Number of visible entries to skip
        #[arg(
            long,
            default_value = "0",
            value_parser = clap::value_parser!(i32).range(0..),
            value_name = "n",
            help_heading = "Common options"
        )]
        offset: i32,
        /// Maximum number of visible entries to return
        #[arg(
            long,
            value_parser = clap::value_parser!(i32).range(1..),
            value_name = "n",
            help_heading = "Common options"
        )]
        limit: Option<i32>,
        /// Sort entries by name or modification time
        #[arg(long, value_parser = ["name", "mtime"], value_name = "field", help_heading = "Common options")]
        sort_by: Option<String>,
        /// Sort direction
        #[arg(
            long,
            requires = "sort_by",
            value_parser = ["asc", "desc"],
            value_name = "order",
            help_heading = "Common options"
        )]
        sort_order: Option<String>,
        /// Comma-separated fields to display (name,uri,path,type,size,mode,mtime,locked,id,count,tags,abstract,overview)
        #[arg(
            short = 'f',
            long = "fields",
            value_delimiter = ',',
            value_name = "FIELDS",
            help_heading = "Output options"
        )]
        fields: Option<Vec<String>>,
        /// Comma-separated k=v retrieval tags; all tags must match
        #[arg(
            long = "tags",
            value_delimiter = ',',
            value_name = "k=v",
            help_heading = "Common options"
        )]
        tags: Vec<String>,
    },
    /// Get directory tree
    Tree {
        /// Viking URI to get tree for
        #[arg(value_name = "uri")]
        uri: String,
        /// Maximum returned abstract length
        #[arg(
            long = "abs-limit",
            short = 'l',
            default_value = "128",
            value_name = "n",
            help_heading = "Advanced options"
        )]
        abs_limit: i32,
        /// Include directory L0 abstracts (defaults to the selected output mode)
        #[arg(
            long = "include-abstract",
            default_missing_value = "true",
            num_args = 0..=1,
            require_equals = true,
            action = ArgAction::Set,
            value_name = "bool",
            help_heading = "Output options"
        )]
        include_abstract: Option<bool>,
        /// Include directory L1 overviews
        #[arg(
            long = "include-overview",
            default_missing_value = "true",
            num_args = 0..=1,
            require_equals = true,
            action = ArgAction::Set,
            value_name = "bool",
            help_heading = "Output options"
        )]
        include_overview: Option<bool>,
        /// Maximum overview content length
        #[arg(
            long = "overview-limit",
            default_value = "4000",
            value_parser = clap::value_parser!(i32).range(1..),
            value_name = "n",
            help_heading = "Advanced options"
        )]
        overview_limit: i32,
        /// Show all hidden files
        #[arg(short, long, help_heading = "Common options")]
        all: bool,
        /// Only include directories
        #[arg(long = "directories-only", help_heading = "Common options")]
        directories_only: bool,
        /// Maximum number of nodes to list
        #[arg(
            long = "node-limit",
            short = 'n',
            default_value = "256",
            value_parser = clap::value_parser!(i32).range(0..),
            value_name = "n",
            help_heading = "Common options"
        )]
        node_limit: i32,
        /// Number of visible entries to skip
        #[arg(
            long,
            default_value = "0",
            value_parser = clap::value_parser!(i32).range(0..),
            value_name = "n",
            help_heading = "Common options"
        )]
        offset: i32,
        /// Maximum number of visible entries to return
        #[arg(
            long,
            value_parser = clap::value_parser!(i32).range(1..),
            value_name = "n",
            help_heading = "Common options"
        )]
        limit: Option<i32>,
        /// Maximum depth level to traverse (default: 3)
        #[arg(
            short = 'L',
            long = "level-limit",
            default_value = "3",
            value_name = "n",
            help_heading = "Common options"
        )]
        level_limit: i32,
        /// Simple path output (just paths, no tree formatting)
        #[arg(short, long, help_heading = "Common options")]
        simple: bool,
        /// Comma-separated fields to display (name,uri,path,type,size,mode,mtime,locked,id,count,tags,abstract,overview)
        #[arg(
            short = 'f',
            long = "fields",
            value_delimiter = ',',
            value_name = "FIELDS",
            help_heading = "Output options"
        )]
        fields: Option<Vec<String>>,
        /// Comma-separated k=v retrieval tags; all tags must match
        #[arg(
            long = "tags",
            value_delimiter = ',',
            value_name = "k=v",
            help_heading = "Common options"
        )]
        tags: Vec<String>,
    },
    /// Run semantic retrieval
    Find {
        /// Search query
        #[arg(value_name = "query")]
        query: Option<String>,
        /// Image query: local path, data URI, HTTP URL, or viking:// URI
        #[arg(
            long = "image",
            value_name = "path|uri",
            help_heading = "Common options"
        )]
        image: Option<String>,
        /// Target URI
        #[arg(
            short,
            long,
            default_value = "",
            value_name = "uri",
            help_heading = "Common options"
        )]
        uri: String,
        /// Maximum final results returned
        #[arg(
            short = 'n',
            long = "node-limit",
            alias = "limit",
            default_value = "10",
            value_parser = clap::value_parser!(i32).range(0..),
            value_name = "n",
            help_heading = "Common options"
        )]
        node_limit: i32,
        /// Score threshold
        #[arg(short, long, value_name = "score", help_heading = "Common options")]
        threshold: Option<f64>,
        /// Only include results on or after this time (e.g. 48h, 7d, 2026-03-10, ISO-8601)
        #[arg(long = "after", value_name = "time", help_heading = "Advanced options")]
        after: Option<String>,
        /// Only include results on or before this time (e.g. 24h, 2026-03-15, ISO-8601)
        #[arg(
            long = "before",
            value_name = "time",
            help_heading = "Advanced options"
        )]
        before: Option<String>,
        /// Only include results with specific level(s) (0=abstract, 1=overview, 2=file)
        #[arg(
            short = 'L',
            long = "level",
            value_delimiter = ',',
            value_name = "0,1,2",
            help_heading = "Common options"
        )]
        level: Option<Vec<i32>>,
        /// Only include results with specific context type(s) (memory, resource, skill)
        #[arg(
            long = "context-type",
            value_delimiter = ',',
            value_name = "type",
            help_heading = "Common options"
        )]
        context_type: Option<Vec<String>>,
        /// Only include results matching all of these explicit tags
        #[arg(long = "tags", value_delimiter = ',')]
        tags: Option<Vec<String>>,
        /// Include the full visible content for every matched URI
        #[arg(long, help_heading = "Advanced options")]
        read_content: bool,
        /// Enable event decay with protection duration: 0 (immediate) or Xm/Xh/Xd
        #[arg(
            long,
            alias = "events_time_decay_protection",
            value_parser = crate::parse_event_time_decay_protection,
            help_heading = "Advanced options"
        )]
        events_time_decay_protection: Option<String>,
    },
    /// Run context-aware retrieval
    Search {
        /// Search query
        #[arg(value_name = "query")]
        query: Option<String>,
        /// Image query: local path, data URI, HTTP URL, or viking:// URI
        #[arg(
            long = "image",
            value_name = "path|uri",
            help_heading = "Common options"
        )]
        image: Option<String>,
        /// Retrieval type
        #[arg(
            long,
            default_value = "semantic",
            value_parser = ["semantic", "keywords"],
            help_heading = "Common options"
        )]
        search_type: String,
        /// Return ranked hits (list) or assembled context (context)
        #[arg(
            long,
            default_value = "list",
            value_parser = ["list", "context"],
            value_name = "list|context",
            help_heading = "Common options"
        )]
        mode: String,
        /// Target URI (list mode only)
        #[arg(
            short,
            long,
            default_value = "",
            value_name = "uri",
            help_heading = "Common options"
        )]
        uri: String,
        /// Session ID for context-aware search
        #[arg(long, value_name = "id", help_heading = "Common options")]
        session_id: Option<String>,
        /// Maximum results per search pass. Search may merge multiple passes.
        #[arg(
            short = 'n',
            long = "node-limit",
            alias = "limit",
            default_value = "10",
            value_parser = clap::value_parser!(i32).range(0..),
            value_name = "n",
            help_heading = "Common options"
        )]
        node_limit: i32,
        /// Score threshold
        #[arg(short, long, value_name = "score", help_heading = "Advanced options")]
        threshold: Option<f64>,
        /// Only include results on or after this time (e.g. 48h, 7d, 2026-03-10, ISO-8601)
        #[arg(long = "after", value_name = "time", help_heading = "Advanced options")]
        after: Option<String>,
        /// Only include results on or before this time (e.g. 24h, 2026-03-15, ISO-8601)
        #[arg(
            long = "before",
            value_name = "time",
            help_heading = "Advanced options"
        )]
        before: Option<String>,
        /// Filter levels (0=abstract, 1=overview, 2=file); ignored in context mode
        #[arg(
            short = 'L',
            long = "level",
            value_delimiter = ',',
            value_name = "0,1,2",
            help_heading = "Advanced options"
        )]
        level: Option<Vec<i32>>,
        /// Only include results with specific context type(s) (memory, resource, skill)
        #[arg(
            long = "context-type",
            value_delimiter = ',',
            value_name = "type",
            help_heading = "Advanced options"
        )]
        context_type: Option<Vec<String>>,
        /// Only include results matching all of these explicit tags
        #[arg(long = "tags", value_delimiter = ',')]
        tags: Option<Vec<String>>,
        /// Include the full visible content for every matched URI (list mode only)
        #[arg(long, help_heading = "Advanced options")]
        read_content: bool,
        /// Enable event decay with protection duration: 0 (immediate) or Xm/Xh/Xd
        #[arg(
            long,
            alias = "events_time_decay_protection",
            value_parser = crate::parse_event_time_decay_protection,
            help_heading = "Advanced options"
        )]
        events_time_decay_protection: Option<String>,
    },
}

/// Execute the restricted command set with a Bearer key; report failures and exit nonzero.
pub(super) async fn run() {
    let args: Vec<OsString> = std::env::args_os().collect();
    let command_display = error_ui::display_command(&args);
    let cli = match CompileCli::try_parse_from(&args) {
        Ok(cli) => cli,
        Err(error) => error.exit(),
    };

    let output_format = OutputFormat::from(cli.output);
    let configured_url = match Config::load_required() {
        Ok(config) => Some(config.url),
        Err(Error::MissingConfig) => None,
        Err(error) => {
            print_compile_error(&command_display, &error, output_format, cli.compact);
            std::process::exit(2);
        }
    };
    let config = match runtime_config(std::env::var(OPENVIKING_API_KEY_ENV).ok(), configured_url) {
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
        CompileCommand::Read { uri, offset, limit } => {
            handlers::handle_read(uri, offset, limit, ctx).await
        }
        CompileCommand::Write {
            uri,
            content,
            from_file,
            append,
            mode,
            wait,
            processing_mode,
            tags,
            tag_mode,
            acl,
        } => {
            let effective_mode = if let Some(m) = mode {
                m
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
                processing_mode,
                tags,
                tag_mode,
                acl,
                ctx,
            )
            .await
        }
        CompileCommand::Grep {
            uri,
            exclude_uri,
            pattern,
            ignore_case,
            after_context,
            before_context,
            node_limit,
            level_limit,
            tags,
            fields,
        } => {
            handlers::handle_grep(
                uri,
                exclude_uri,
                pattern,
                ignore_case,
                after_context,
                before_context,
                node_limit,
                level_limit,
                tags,
                fields,
                ctx,
            )
            .await
        }
        CompileCommand::Glob {
            pattern,
            uri,
            node_limit,
            simple,
            fields,
            tags,
        } => handlers::handle_glob(pattern, uri, node_limit, simple, fields, tags, ctx).await,
        CompileCommand::Ls {
            uri,
            simple,
            recursive,
            abs_limit,
            include_abstract,
            include_overview,
            overview_limit,
            all,
            node_limit,
            offset,
            limit,
            sort_by,
            sort_order,
            fields,
            tags,
        } => {
            handlers::handle_ls(
                uri,
                simple,
                recursive,
                abs_limit,
                include_abstract,
                include_overview,
                overview_limit,
                all,
                node_limit,
                offset,
                limit,
                sort_by,
                sort_order,
                fields,
                tags,
                ctx,
            )
            .await
        }
        CompileCommand::Tree {
            uri,
            abs_limit,
            include_abstract,
            include_overview,
            overview_limit,
            all,
            directories_only,
            node_limit,
            offset,
            limit,
            level_limit,
            simple,
            fields,
            tags,
        } => {
            handlers::handle_tree(
                uri,
                abs_limit,
                include_abstract,
                include_overview,
                overview_limit,
                all,
                directories_only,
                node_limit,
                offset,
                limit,
                level_limit,
                simple,
                fields,
                tags,
                ctx,
            )
            .await
        }
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
            read_content,
            events_time_decay_protection,
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
                read_content,
                events_time_decay_protection,
                ctx,
            )
            .await
        }
        CompileCommand::Search {
            query,
            image,
            search_type,
            mode,
            uri,
            session_id,
            node_limit,
            threshold,
            after,
            before,
            level,
            context_type,
            tags,
            read_content,
            events_time_decay_protection,
        } => {
            handlers::handle_search(
                query,
                uri,
                image,
                search_type,
                mode,
                session_id,
                node_limit,
                threshold,
                after,
                before,
                level,
                context_type,
                tags,
                read_content,
                events_time_decay_protection,
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
    if matches!(error, Error::Config(message) if message.contains(OPENVIKING_API_KEY_ENV)) {
        eprintln!("\nSet the key with: export {OPENVIKING_API_KEY_ENV}=\"<your-api-key>\"");
    }
}

/// Validate the required Bearer key and optional HTTP(S) base URL, ignoring other profile credentials.
fn runtime_config(api_key: Option<String>, service_url: Option<String>) -> Result<Config> {
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

    let service_url = service_url
        .map(|value| value.trim().trim_end_matches('/').to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| OPENVIKING_COMPILE_URL.to_string());
    let parsed_url = reqwest::Url::parse(&service_url)
        .map_err(|_| Error::Config("ovcli.conf contains an invalid URL".to_string()))?;
    if !matches!(parsed_url.scheme(), "http" | "https")
        || parsed_url.host_str().is_none()
        || parsed_url.query().is_some()
        || parsed_url.fragment().is_some()
    {
        return Err(Error::Config(
            "ovcli.conf URL must be an HTTP(S) base URL without a query or fragment".to_string(),
        ));
    }

    Ok(Config {
        url: service_url,
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
    use clap::{Command, CommandFactory};

    /// Captures the parsing behavior that must remain identical for shared CLI commands.
    #[derive(Debug, PartialEq)]
    struct ArgumentContract {
        id: String,
        short: Option<char>,
        long: Option<String>,
        aliases: Vec<String>,
        value_names: Vec<String>,
        num_args: String,
        value_delimiter: Option<char>,
        default_values: Vec<String>,
        possible_values: Vec<String>,
        action: String,
        required: bool,
    }

    /// Returns the non-global argument contract for one subcommand in declaration order.
    fn argument_contract(command: &Command) -> Vec<ArgumentContract> {
        command
            .get_arguments()
            .filter(|argument| !argument.is_global_set())
            .map(|argument| ArgumentContract {
                id: argument.get_id().to_string(),
                short: argument.get_short(),
                long: argument.get_long().map(str::to_string),
                aliases: argument
                    .get_all_aliases()
                    .unwrap_or_default()
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
                value_names: argument
                    .get_value_names()
                    .unwrap_or_default()
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
                num_args: format!("{:?}", argument.get_num_args()),
                value_delimiter: argument.get_value_delimiter(),
                default_values: argument
                    .get_default_values()
                    .iter()
                    .map(|value| value.to_string_lossy().into_owned())
                    .collect(),
                possible_values: argument
                    .get_possible_values()
                    .iter()
                    .map(|value| value.get_name().to_string())
                    .collect(),
                action: format!("{:?}", argument.get_action()),
                required: argument.is_required_set(),
            })
            .collect()
    }

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
    fn selected_commands_match_the_full_cli_argument_contract() {
        std::thread::Builder::new()
            .name("compile-cli-contract".to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(assert_selected_commands_match_the_full_cli)
            .expect("contract test thread must start")
            .join()
            .expect("contract test thread must complete");
    }

    /// Compares selected subcommands on a larger stack because the full Clap tree is deeply nested.
    fn assert_selected_commands_match_the_full_cli() {
        let full_cli = crate::Cli::command();
        let compile_cli = CompileCli::command();

        for name in [
            "read", "write", "grep", "glob", "ls", "tree", "find", "search",
        ] {
            let full_command = full_cli
                .get_subcommands()
                .find(|command| command.get_name() == name)
                .expect("selected command must exist in the full CLI");
            let compile_command = compile_cli
                .get_subcommands()
                .find(|command| command.get_name() == name)
                .expect("selected command must exist in the compile CLI");

            assert_eq!(
                argument_contract(compile_command),
                argument_contract(full_command),
                "argument contract differs for {name}"
            );
            assert_eq!(
                compile_command.get_all_aliases().collect::<Vec<_>>(),
                full_command.get_all_aliases().collect::<Vec<_>>(),
                "aliases differ for {name}"
            );
        }
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
            "--tags",
            "env=prod,team=search",
            "--tag-mode",
            "append",
            "--acl",
            r#"{"acl_mode":"restricted","entries":[]}"#,
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
                tags,
                tag_mode,
                acl,
            } => {
                assert_eq!(uri, "viking://resources/a.md");
                assert_eq!(content.as_deref(), Some("hello"));
                assert!(from_file.is_none());
                assert!(!append);
                assert_eq!(mode.as_deref(), Some("create"));
                assert!(wait);
                assert_eq!(processing_mode, "vectors_only");
                assert_eq!(tags, ["env=prod", "team=search"]);
                assert_eq!(tag_mode, "append");
                assert_eq!(
                    acl,
                    Some(serde_json::json!({"acl_mode": "restricted", "entries": []}))
                );
            }
            _ => panic!("expected write command"),
        }
    }

    #[test]
    fn rejects_other_mutating_and_administration_commands() {
        assert!(CompileCli::try_parse_from(["ov", "rm", "viking://resources/a"]).is_err());
        assert!(CompileCli::try_parse_from(["ov", "config"]).is_err());
        assert!(CompileCli::try_parse_from(["ov", "mv", "a", "b"]).is_err());
    }

    #[test]
    fn runtime_config_uses_configured_service_or_default_and_vault_key() {
        let config = runtime_config(
            Some("  secret  ".to_string()),
            Some(" https://stg.example.com/openviking/ ".to_string()),
        )
        .expect("valid config");

        assert_eq!(config.url, "https://stg.example.com/openviking");
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

        let default_config =
            runtime_config(Some("secret".to_string()), None).expect("valid config");
        assert_eq!(default_config.url, OPENVIKING_COMPILE_URL);
    }

    #[test]
    fn runtime_config_preserves_vault_placeholder_in_bearer_header() {
        let config = runtime_config(Some("ARK_SECRET_PLACEHOLDER".to_string()), None)
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
            let error = runtime_config(value, None).expect_err("missing key must fail");
            assert!(error.to_string().contains(OPENVIKING_API_KEY_ENV));
        }
    }

    #[test]
    fn runtime_config_rejects_a_key_that_cannot_be_sent_as_an_http_header() {
        let error =
            runtime_config(Some("bad\nkey".to_string()), None).expect_err("invalid key must fail");
        assert!(error.to_string().contains(OPENVIKING_API_KEY_ENV));
    }
}
