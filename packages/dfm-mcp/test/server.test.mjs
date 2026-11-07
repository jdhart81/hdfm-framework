// Protocol behaviour and invariants H1-H7 of the DFM MCP server (see src/server.mjs).
import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFile, readdir, writeFile, mkdtemp, rm} from 'node:fs/promises';
import {PassThrough} from 'node:stream';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import * as core from '@viridis/dfm-core';
import {buildServer, EXAMPLES_DIR, loadExample} from '../src/server.mjs';

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const server = buildServer();
let nextId = 1;
const rpc = (method, params) => server.handle({jsonrpc: '2.0', id: nextId++, method, ...(params ? {params} : {})});
const call = async (name, args = {}) => (await rpc('tools/call', {name, arguments: args})).result;
const body = res => { assert.equal(res.isError, false, JSON.stringify(res.content)); return res.structuredContent; };
const hashDir = async dir => {
  const h = createHash('sha256');
  for (const f of (await readdir(dir)).sort()) h.update(f).update(await readFile(path.join(dir, f)));
  return h.digest('hex');
};

// -- protocol --------------------------------------------------------------------
test('initialize echoes a supported version and falls back to the latest', async () => {
  const r = (await rpc('initialize', {protocolVersion: '2025-06-18', capabilities: {}, clientInfo: {name: 't', version: '0'}})).result;
  assert.equal(r.protocolVersion, '2025-06-18');
  assert.ok(r.capabilities.tools);
  assert.equal(r.serverInfo.name, 'dfm');
  assert.equal((await rpc('initialize', {protocolVersion: '1999-01-01'})).result.protocolVersion, '2025-11-25');
});

test('notifications get no response; unknown methods are -32601', async () => {
  assert.equal(await server.handle({jsonrpc: '2.0', method: 'notifications/initialized'}), null);
  assert.equal((await rpc('nope')).error.code, -32601);
  assert.equal((await server.handle({id: 1, method: 'x'})).error.code, -32600);
});

test('tools/list names every tool, all read-only', async () => {
  const tools = (await rpc('tools/list')).result.tools;
  assert.deepEqual(tools.map(t => t.name).sort(), ['check_plan', 'climate_routes', 'describe_method', 'draft_spine', 'get_example',
    'list_examples', 'next_holdings', 'project_spine', 'spine_network']);
  for (const t of tools) {
    assert.equal(t.annotations.readOnlyHint, true, t.name);
    assert.equal(t.annotations.openWorldHint, false, t.name);
    assert.equal(t.inputSchema.type, 'object', t.name);
  }
});

test('bad arguments are tool errors, not crashes', async () => {
  assert.equal((await call('no_such_tool')).isError, true);
  assert.equal((await call('get_example', {})).isError, true);
  assert.equal((await call('get_example', {example: '../../etc/passwd'})).isError, true);
  assert.equal((await call('describe_method', {extra: 1})).isError, true);
  assert.equal((await call('check_plan', {})).isError, true, 'none of package, path, example');
  assert.equal((await call('check_plan', {example: 'woodlot-clearcut', path: 'x.json'})).isError, true, 'two inputs');
  assert.equal((await call('check_plan', {package: {dfm_package: '9.9'}})).isError, true, 'wrong version');
  assert.equal((await call('check_plan', {path: '/etc/passwd'})).isError, true, 'not a .json file');
});

test('serve writes one JSON line per response and answers parse errors', async () => {
  const input = new PassThrough(), output = new PassThrough();
  let text = '';
  output.on('data', d => { text += d; });
  const done = buildServer().serve(input, output);
  input.end([JSON.stringify({jsonrpc: '2.0', id: 1, method: 'ping'}), JSON.stringify({jsonrpc: '2.0', method: 'notifications/initialized'}),
    'not json', JSON.stringify({jsonrpc: '2.0', id: 2, method: 'tools/list'})].join('\n') + '\n');
  await done;
  const resps = text.trim().split('\n').map(l => JSON.parse(l));
  assert.equal(resps.length, 3);
  assert.deepEqual(resps.map(r => r.id).sort((a, b) => (a ?? 0) - (b ?? 0)), [null, 1, 2]);
  assert.equal(resps.find(r => r.id === null).error.code, -32700);
});

// -- invariants ------------------------------------------------------------------
test('H1 every successful answer carries source and limitations', async () => {
  const cases = [['describe_method', {}], ['list_examples', {}], ['get_example', {example: 'woodlot-clearcut'}],
    ['check_plan', {example: 'woodlot-clearcut'}], ['draft_spine', {example: 'watershed-lines'}], ['next_holdings', {example: 'watershed-spine'}]];
  for (const [name, args] of cases) {
    const res = await call(name, args);
    const b = body(res);
    assert.equal(b.source.repository, 'https://github.com/jdhart81/hdfm-framework', name);
    assert.equal(b.source.license, 'MIT');
    assert.equal(b.source.engine.connectivity, core.ENGINE_VERSION);
    assert.ok(b.limitations.some(l => l.startsWith('Structural connectivity only')), name);
    assert.deepEqual(JSON.parse(res.content[0].text), b, `${name}: text mirrors structuredContent`);
  }
});

test('H2 fail and incomplete are never summarized as a pass', async () => {
  const fail = body(await call('check_plan', {example: 'woodlot-clearcut'}));
  assert.equal(fail.status, 'fail');
  assert.match(fail.summary, /^Fails: .*harvest-3/);
  assert.doesNotMatch(fail.summary, /pass/i);

  const pass = body(await call('check_plan', {example: 'woodlot-thinning'}));
  assert.equal(pass.status, 'pass');
  assert.match(pass.summary, /^Passes the structural check/);

  const doc = await loadExample('woodlot-clearcut');
  const broken = structuredClone(doc.package);
  delete broken.params.minWidthM;
  const inc = body(await call('check_plan', {package: broken}));
  assert.equal(inc.status, 'incomplete');
  assert.match(inc.summary, /^No result/);
  assert.doesNotMatch(inc.summary, /pass/i);
  assert.ok(inc.result.reasons.length > 0);
});

test('H3 same answer as the engine, and identical on repeat', async () => {
  for (const ex of ['woodlot-clearcut', 'woodlot-thinning', 'prairie-stepping-stones']) {
    const input = core.fromLandscapePackage((await loadExample(ex)).package);
    const direct = core.checkConnectivitySync(input);
    const a = body(await call('check_plan', {example: ex, include_geometry: true}));
    const b = body(await call('check_plan', {example: ex, include_geometry: true}));
    assert.deepEqual(a, b, `${ex}: repeat`);
    assert.deepEqual(a.result, JSON.parse(JSON.stringify(direct)), `${ex}: engine result passed through unchanged`);
  }
  const lines = core.fromLandscapePackage((await loadExample('watershed-lines')).package);
  const spine = body(await call('draft_spine', {example: 'watershed-lines', include_geometry: true}));
  assert.deepEqual(spine.result, JSON.parse(JSON.stringify(core.deriveSpine(lines))));
});

test('H4 reproduces the check VergeCommon stored for a plan', async () => {
  const doc = await loadExample('vergecommon-plan');
  const r = body(await call('check_plan', {example: 'vergecommon-plan'})).result;
  assert.equal(r.inputChecksum, doc.stored.inputChecksum);
  assert.equal(r.status, doc.stored.status);
  assert.deepEqual(r.lostLinks, doc.stored.lostLinks);
  assert.deepEqual(r.pinchedLinks, doc.stored.pinchedLinks);
});

test('H5 read-only: path input is read, never written; examples unchanged', async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'dfm-mcp-'));
  try {
    const file = path.join(dir, 'plan.json');
    const text = JSON.stringify((await loadExample('vergecommon-plan')).package);
    await writeFile(file, text);
    const before = await hashDir(EXAMPLES_DIR);
    const viaPath = body(await call('check_plan', {path: file}));
    const viaExample = body(await call('check_plan', {example: 'vergecommon-plan'}));
    assert.equal(viaPath.result.inputChecksum, viaExample.result.inputChecksum);
    await call('draft_spine', {example: 'watershed-lines'});
    assert.equal(await readFile(file, 'utf8'), text);
    assert.deepEqual(await readdir(dir), ['plan.json']);
    assert.equal(await hashDir(EXAMPLES_DIR), before);
  } finally {
    await rm(dir, {recursive: true, force: true});
  }
});

test('H5 offline: the server source imports no network module and calls no fetch', async () => {
  for (const f of await readdir(path.join(ROOT, 'src'))) {
    const src = await readFile(path.join(ROOT, 'src', f), 'utf8');
    assert.doesNotMatch(src, /from ['"](node:)?(http|https|http2|net|dgram|tls|dns)['"]/, f);
    assert.doesNotMatch(src, /\bfetch\s*\(/, f);
    assert.doesNotMatch(src, /\b(writeFile|appendFile|createWriteStream|mkdir|unlink|rm)\s*\(/, `${f} writes nothing`);
  }
});

test('H6 a call over the time limit returns an error and the server keeps answering', async () => {
  process.env.DFM_MCP_TIMEOUT_MS = '1';
  try {
    const res = await call('spine_network', {example: 'prairie-stepping-stones'});
    assert.equal(res.isError, true);
    assert.match(res.content[0].text, /stopped after/);
  } finally {
    delete process.env.DFM_MCP_TIMEOUT_MS;
  }
  assert.deepEqual((await rpc('ping')).result, {});
  assert.equal(body(await call('check_plan', {example: 'woodlot-thinning'})).status, 'pass');
});

test('H7 no tool takes coordinates, a URL or a destination', async () => {
  for (const t of (await rpc('tools/list')).result.tools) {
    const props = Object.keys(t.inputSchema.properties ?? {});
    for (const p of props) assert.doesNotMatch(p, /^(lat|lon|lng|url|uri|endpoint|email|webhook|upload)/i, `${t.name}.${p}`);
  }
});

// -- spine tools on the examples -------------------------------------------------
test('spine tools run on the examples and relay engine status', async () => {
  const net = body(await call('spine_network', {example: 'watershed-lines', cuts: false}));
  assert.equal(net.status, 'ok');
  assert.equal(net.result.summary.rho2, null, 'robustness is null when the search is skipped (SP7)');
  assert.match(net.summary, /not computed/);
  const next = body(await call('next_holdings', {example: 'watershed-spine'}));
  assert.equal(next.status, 'ok');
  assert.ok(next.result.frontier.length > 0);
  const routes = body(await call('climate_routes', {example: 'watershed-spine'}));
  assert.equal(routes.status, 'ok');
  const proj = body(await call('project_spine', {example: 'watershed-spine', years: [2026, 2051, 2126]}));
  assert.equal(proj.status, 'ok');
  const draft = body(await call('draft_spine', {example: 'watershed-lines'}));
  assert.ok(draft.result.geometryOmitted && !draft.result.features[0].geometry, 'geometry omitted by default');
});

test('examples are what the build script makes', async () => {
  const names = (await readdir(EXAMPLES_DIR)).map(f => f.replace(/\.json$/, '')).sort();
  const listed = body(await call('list_examples')).examples.map(e => e.name);
  assert.deepEqual(listed, names);
  for (const n of names) assert.equal((await loadExample(n)).package.dfm_package, core.SCHEMA_VERSION, n);
});

// -- the installed command -------------------------------------------------------
test('the dfm-mcp command speaks MCP over stdio', async () => {
  const child = spawn(process.execPath, [path.join(ROOT, 'src', 'cli.mjs')], {stdio: ['pipe', 'pipe', 'pipe']});
  const lines = [];
  let buf = '';
  child.stdout.on('data', d => { buf += d; let i; while ((i = buf.indexOf('\n')) >= 0) { lines.push(JSON.parse(buf.slice(0, i))); buf = buf.slice(i + 1); } });
  const send = m => child.stdin.write(`${JSON.stringify(m)}\n`);
  send({jsonrpc: '2.0', id: 1, method: 'initialize', params: {protocolVersion: '2025-06-18'}});
  send({jsonrpc: '2.0', method: 'notifications/initialized'});
  send({jsonrpc: '2.0', id: 2, method: 'tools/call', params: {name: 'check_plan', arguments: {example: 'vergecommon-plan'}}});
  const deadline = Date.now() + 30000;
  while (lines.length < 2 && Date.now() < deadline) await new Promise(r => setTimeout(r, 50));
  child.stdin.end();
  await new Promise(r => child.on('close', r));
  assert.equal(lines.length, 2);
  assert.equal(lines.find(l => l.id === 1).result.serverInfo.name, 'dfm');
  assert.equal(lines.find(l => l.id === 2).result.structuredContent.status, 'fail');
});

test('registry entry matches the package', async () => {
  const pkg = JSON.parse(await readFile(path.join(ROOT, 'package.json'), 'utf8'));
  const entry = JSON.parse(await readFile(path.join(ROOT, 'server.json'), 'utf8'));
  assert.equal(entry.name, pkg.mcpName, 'npm ownership check for the MCP registry reads mcpName');
  assert.equal(entry.version, pkg.version);
  assert.equal(entry.packages[0].version, pkg.version);
  assert.equal(entry.packages[0].identifier, pkg.name);
  assert.ok(entry.description.length <= 100);
  assert.equal(pkg.bin['dfm-mcp'], 'src/cli.mjs');
});
