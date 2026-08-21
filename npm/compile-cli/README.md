# @openviking-compile/cli

Read-only OpenViking CLI for cloud agents. The package installs an `ov` command with seven retrieval commands and no mutation or administration commands.

## Install

```bash
npm i -g @openviking-compile/cli
```

Do not install this package globally alongside `@openviking/cli`; both packages provide an executable named `ov`.

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

The API endpoint is compiled into the binary and cannot be overridden:

```text
https://api.vikingdb.cn-beijing.volces.com/openviking
```

The CLI does not read `~/.openviking/ovcli.conf` or `OPENVIKING_CLI_CONFIG_FILE`.

## Commands

| Command | Purpose |
| --- | --- |
| `ov read <uri>` | Read full file content |
| `ov grep --uri <uri> <pattern>` | Search file content with a regular expression |
| `ov glob <pattern>` | Search file names with a glob pattern |
| `ov ls [uri]` | List directory contents |
| `ov tree <uri>` | Print a directory tree |
| `ov find <query>` | Run semantic retrieval |
| `ov search <query>` | Run context-aware retrieval |

Run `ov <command> --help` for command-specific options. Add `--output json` for machine-readable output.

## Supported platforms

| Platform | Architecture |
| --- | --- |
| Linux | x64, ARM64 |
| macOS | Apple Silicon, Intel |
| Windows | x64 |

## License

Apache-2.0
