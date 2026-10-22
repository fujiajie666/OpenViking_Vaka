# @openviking/compile

OpenViking retrieval and content-write CLI for cloud agents. The package installs an `ov` command with seven retrieval commands, `snapshot diff`, `snapshot diff-paths`, and `write`; other mutation and administration commands are not included. Commands follow the same option contract and behavior as their counterparts in `@openviking/cli`, except that `write` always records its edit source as `compile`, exposes no source option, and accepts only `vectors_only` processing.

## Install

```bash
npm i -g @openviking/compile
```

Do not install this package globally alongside `@openviking/cli`; both packages provide an executable named `ov`. Install one package globally, or run this restricted distribution in an isolated project with `npm exec`.

```bash
npm exec --yes --package=@openviking/compile -- ov --help
```

## Authentication

The CLI reads its API key only from `OPENVIKING_API_KEY`:

```bash
export OPENVIKING_API_KEY="<your-api-key>"
```

The value is sent unchanged as a Bearer credential in the `Authorization`
header. This is compatible with sandbox Vaults that expose a placeholder in
the environment and replace it at the network egress. The native CLI trusts
both Mozilla public roots and certificates installed in the operating
system's trust store, including sandbox egress gateway CAs.

The API endpoint defaults to:

```text
https://api.vikingdb.cn-beijing.volces.com/openviking
```

To use another OpenViking data-plane endpoint, write its URL to
`~/.openviking/ovcli.conf`:

```json
{
  "url": "https://openviking.example.com/openviking"
}
```

Only the URL is read from this file. Authentication still comes exclusively
from `OPENVIKING_API_KEY`; credentials in `ovcli.conf` are ignored. Set
`OPENVIKING_CLI_CONFIG_FILE` to select a different config file.

## Commands

| Command | Purpose |
| --- | --- |
| `ov read <uri>` | Read full content or a range of lines |
| `ov write <uri> --content <text>` | Replace, append, or create text content |
| `ov grep --uri <uri> <pattern>` | Search file content with a regular expression |
| `ov glob <pattern>` | Search file names with a glob pattern |
| `ov ls [uri]` | List directory contents |
| `ov tree <uri>` | Print a directory tree |
| `ov find <query>` | Run semantic retrieval |
| `ov search <query>` | Retrieve ranked hits or assembled context using semantic or keyword search |
| `ov snapshot diff <uri> --to <ref>` | Compare one file between snapshots |
| `ov snapshot diff-paths --paths <uris> --to <ref>` | List changed source files under directories between snapshots |

Run `ov <command> --help` for command-specific options. Add `--output json` for machine-readable output.

```bash
ov read viking://resources/notes.md --offset 10 --limit 20
ov ls viking://resources --offset 0 --limit 50 --sort-by mtime --sort-order desc
ov tree viking://resources --directories-only --include-overview
ov grep --uri viking://resources "pattern" --before-context 2 --after-context 2
ov search "deployment" --search-type keywords
ov search "project context" --mode context
ov find "recent events" --events-time-decay-protection 7d
```

Listings preserve `has_more` in JSON output and display a pagination hint in table output when more entries are available.

`snapshot diff` compares one file between snapshots. Use `--from` to select the older snapshot; omitting it compares against an empty file. The default output is diff text; add `--output json` for the structured result.

```bash
ov snapshot diff viking://resources/docs/notes.md --from <base-ref> --to <head-ref>
```

`snapshot diff-paths` accepts comma-separated directory URIs. Use `--from` to select the older snapshot; omitting it compares against an empty tree. References accept commit IDs, branches, or tags. Pagination uses `--offset` (default: `0`) and `--limit` (default: `100`, range: `1`–`1000`).

```bash
ov snapshot diff-paths --paths viking://resources/docs,viking://resources/notes --from <base-ref> --to <head-ref> --offset 0 --limit 100 --output json
```

`write` accepts inline text or a local UTF-8 text file:

```bash
ov write viking://resources/notes.md --content "updated text"
ov write viking://resources/notes.md --from-file ./notes.md --append
ov write viking://resources/new.md --content "new text" --mode create --wait
```

The write modes are `replace` (default), `append`, and `create`. Writes use `vectors_only` processing by default, updating vectors without semantic processing. The optional `--processing-mode` argument accepts only `vectors_only`. Writes support `--tags`, `--tag-mode replace|append|clear`, and ACL JSON via `--acl`. The `clear` tag mode does not require `--tags`.

Every write sends `edit_source: "compile"` to record Compile provenance in output versions. No source argument is needed. The server must support Compile edit sources.

`write --wait` waits within the HTTP request timeout. A separate `write --timeout` option is not supported by the current API.

## Supported platforms

| Platform | Architecture |
| --- | --- |
| Linux | x64, ARM64 |
| macOS | Apple Silicon, Intel |
| Windows | x64 |

## License

Apache-2.0
