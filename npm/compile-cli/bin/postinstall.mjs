#!/usr/bin/env node

process.stderr.write(`
@openviking-compile/cli installed.

Set the API key before use:
  export OPENVIKING_API_KEY="<your-api-key>"

Available commands:
  ov read <uri>
  ov grep --uri <uri> <pattern>
  ov glob <pattern> [--uri <uri>]
  ov ls [uri]
  ov tree <uri>
  ov find <query> [--uri <uri>]
  ov search <query> [--uri <uri>]

The service endpoint is fixed in the binary.
`);
