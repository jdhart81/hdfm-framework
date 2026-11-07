#!/usr/bin/env node
// Start the DFM MCP server on stdio:  npx @viridis/dfm-mcp   (or: node src/cli.mjs)
import {buildServer} from './server.mjs';

await buildServer().serve();
