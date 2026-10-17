# @openviking/compile

OpenViking retrieval and content-write CLI for cloud agents. The package installs an `ov` command with seven retrieval commands plus `write`; other mutation and administration commands are not included. Each included command follows the same option contract and behavior as its counterpart in `@openviking/cli`.

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

`write` accepts inline text or a local UTF-8 text file:

```bash
ov write viking://resources/notes.md --content "updated text"
ov write viking://resources/notes.md --from-file ./notes.md --append
ov write viking://resources/new.md --content "new text" --mode create --wait
```

The write modes are `replace` (default), `append`, and `create`. Use `--processing-mode vectors_only` to skip semantic processing while updating vectors. Writes support `--tags`, `--tag-mode replace|append|clear`, and ACL JSON via `--acl`. The `clear` tag mode does not require `--tags`.

`write --wait` waits within the HTTP request timeout. A separate `write --timeout` option is not supported by the current API.

## Supported platforms

| Platform | Architecture |
| --- | --- |
| Linux | x64, ARM64 |
| macOS | Apple Silicon, Intel |
| Windows | x64 |

## License

Apache-2.0
