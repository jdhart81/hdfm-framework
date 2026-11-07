// Minimal, dependency-free MCP server over stdio (JSON-RPC 2.0, one message per line).
//
// Implements the subset of the Model Context Protocol a tool-only server needs:
// initialize, notifications/initialized, ping, tools/list and tools/call.
//
// Invariants (tested in test/server.test.mjs):
// - stdout carries protocol messages only; logs go to stderr.
// - every request with an id gets exactly one response; notifications get none.
// - a tool that throws returns a result with isError: true, never a crashed server.
// - tool calls are async, so a slow call never stops the server answering others.
import {createInterface} from 'node:readline';

export const SUPPORTED_PROTOCOL_VERSIONS = ['2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05'];
export const ERR = {PARSE: -32700, INVALID_REQUEST: -32600, METHOD_NOT_FOUND: -32601, INTERNAL: -32603};

/** Thrown by a tool for an expected failure; reported to the client as isError. */
export class ToolError extends Error {}

const result = (id, r) => ({jsonrpc: '2.0', id, result: r});
const error = (id, code, message) => ({jsonrpc: '2.0', id, error: {code, message}});
const toolError = text => ({content: [{type: 'text', text}], isError: true});

export class StdioServer {
  constructor({name, version, instructions = ''}) {
    Object.assign(this, {name, version, instructions});
    this.tools = new Map();
  }

  tool(name, {title, description, inputSchema, readOnly = true}, func) {
    this.tools.set(name, {name, title: title ?? name, description, inputSchema, readOnly, func});
    return this;
  }

  spec(t) {
    return {name: t.name, title: t.title, description: t.description, inputSchema: t.inputSchema,
      annotations: {title: t.title, readOnlyHint: t.readOnly, destructiveHint: false, idempotentHint: true, openWorldHint: false}};
  }

  /** Response for one incoming message, or null for a notification. */
  async handle(msg) {
    if (!msg || typeof msg !== 'object' || Array.isArray(msg) || msg.jsonrpc !== '2.0' || typeof msg.method !== 'string') {
      return error(msg && typeof msg === 'object' ? msg.id ?? null : null, ERR.INVALID_REQUEST, 'invalid request');
    }
    if (!('id' in msg)) return null;
    const {id, method} = msg, params = msg.params ?? {};
    try {
      if (method === 'initialize') {
        const asked = params.protocolVersion;
        return result(id, {
          protocolVersion: SUPPORTED_PROTOCOL_VERSIONS.includes(asked) ? asked : SUPPORTED_PROTOCOL_VERSIONS[0],
          capabilities: {tools: {listChanged: false}},
          serverInfo: {name: this.name, version: this.version},
          instructions: this.instructions,
        });
      }
      if (method === 'ping') return result(id, {});
      if (method === 'tools/list') return result(id, {tools: [...this.tools.values()].map(t => this.spec(t))});
      if (method === 'tools/call') return result(id, await this.call(params));
      return error(id, ERR.METHOD_NOT_FOUND, `method not found: ${method}`);
    } catch (e) {
      process.stderr.write(`${e.stack ?? e}\n`);
      return error(id, ERR.INTERNAL, String(e.message ?? e));
    }
  }

  async call({name, arguments: args = {}} = {}) {
    const t = this.tools.get(name);
    if (!t) return toolError(`unknown tool: ${name}`);
    if (!args || typeof args !== 'object' || Array.isArray(args)) return toolError('arguments must be an object');
    const missing = (t.inputSchema.required ?? []).filter(k => !(k in args));
    if (missing.length) return toolError(`missing required argument(s): ${missing.join(', ')}`);
    const unknown = Object.keys(args).filter(k => !(k in (t.inputSchema.properties ?? {}))).sort();
    if (unknown.length) return toolError(`unknown argument(s): ${unknown.join(', ')}`);
    let out;
    try {
      out = await t.func(args);
    } catch (e) {
      if (!(e instanceof ToolError)) process.stderr.write(`${e.stack ?? e}\n`);
      return toolError(e instanceof ToolError ? e.message : `${e.name}: ${e.message}`);
    }
    return {content: [{type: 'text', text: JSON.stringify(out, null, 2)}], structuredContent: out, isError: false};
  }

  /** Read newline-delimited JSON-RPC from `input`, write one line per response to `output`. */
  serve(input = process.stdin, output = process.stdout) {
    const rl = createInterface({input, crlfDelay: Infinity});
    const pending = new Set();
    const write = resp => { if (resp) output.write(`${JSON.stringify(resp)}\n`); };
    rl.on('line', line => {
      line = line.trim();
      if (!line) return;
      let msg;
      try { msg = JSON.parse(line); } catch { write(error(null, ERR.PARSE, 'parse error')); return; }
      const p = this.handle(msg).then(write);
      pending.add(p);
      p.finally(() => pending.delete(p));
    });
    return new Promise(resolve => rl.on('close', () => Promise.allSettled([...pending]).then(resolve)));
  }
}
