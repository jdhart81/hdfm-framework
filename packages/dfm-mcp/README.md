# @viridis/dfm-mcp

A local [Model Context Protocol](https://modelcontextprotocol.io) server for Dendritic Forest Management (DFM). It lets Claude, Cursor or any MCP client check a harvest or treatment plan against a landscape's retained corridors and plan an old-growth spine, using the [dfm-core](../dfm-core/README.md) engine unchanged. It runs on your machine, reads only the files you point it at, and sends nothing anywhere.

**Status: 0.1.0, experimental.** Structural connectivity only: a pass means corridors stay connected at the recorded minimum width. It does not establish species movement, genetic exchange, regulatory compliance, carbon outcomes or old-growth condition.

## Run it

Not on npm yet; run it from this repository (Node 22+):

```sh
cd packages/dfm-core && npm ci && cd ../dfm-mcp && npm ci
node src/cli.mjs          # MCP over stdio
```

Client config (Claude Desktop, Cursor and others), with your own path:

```json
{"mcpServers": {"dfm": {"command": "node", "args": ["/path/to/hdfm-framework/packages/dfm-mcp/src/cli.mjs"]}}}
```

## Tools

All tools are read-only.

| Tool | What it answers |
|---|---|
| `describe_method` | What a check and the spine establish, the input format, versions, how to cite |
| `list_examples`, `get_example` | Bundled fictional landscapes to try, and their packages as templates |
| `check_plan` | Does the plan break or pinch a link between core areas? `pass`, `fail` or `incomplete`, naming each unit that breaks a link |
| `draft_spine` | Corridors along streams and connector lines, each at least the minimum width, for a steward to review |
| `spine_network` | Links, loops and single points of failure; `rho2` is the share of core pairs no single disturbance separates |
| `project_spine` | Core pairs linked through committed habitat, and through habitat at old-growth age, at each milestone year |
| `climate_routes` | For each core, the coolest core or exit it can reach through links that hold after the plan |
| `next_holdings` | Parcels without consent that would extend the committed spine, by the links each would complete |

Each analysis tool takes exactly one of `package` (a [Landscape Package v1](https://dendriticforest.com/data-format/) object), `path` (a local `.json` file) or `example`. Geometry is left out of answers unless you pass `include_geometry: true`. Analyses run in a worker thread and stop after 120 s (set `DFM_MCP_TIMEOUT_MS` to change it).

Try it: ask your agent to run `check_plan` on the `woodlot-clearcut` example, then on `woodlot-thinning`.

## Guarantees

Tested in [`test/server.test.mjs`](test/server.test.mjs):

- **H1** Every answer carries its source (engine versions, licence, how to cite) and the limitations above.
- **H2** The engine's status is relayed as it is. `fail` and `incomplete` are never summarized as a pass, and `incomplete` reads as "no result".
- **H3** Answers equal a direct dfm-core call on the same package, and repeat identically.
- **H4** A check VergeCommon stored for a woodland co-op plan reproduces exactly: input checksum, status and lost links.
- **H5** Nothing is written and nothing leaves your machine: the server imports no network module.
- **H6** A call over the time limit returns an error; the server keeps answering.
- **H7** No tool takes coordinates, a URL or a destination.

```sh
npm test
```

`npm run examples` rebuilds `examples/` from the dfm-core fixtures; the output is byte-identical on every run.

MIT. From [Viridis Conservation](https://viridisconservation.com/). Neighbors can run the same check on every harvest plan through a woodland co-op on [VergeCommon](https://vergecommon.com/woodland/).
