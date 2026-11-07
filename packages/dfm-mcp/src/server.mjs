// Dendritic Forest Management as a local MCP server (stdio).
//
// Agents can check a harvest or treatment plan against a landscape's retained corridors,
// draft the old-growth spine, test it as a network, project it forward, find climate
// routes and list the next holdings that would extend it. Every analysis is the
// @viridis/dfm-core engine, unchanged; this package only adds the protocol.
//
// Invariants (each tested in test/server.test.mjs):
// - H1 Every successful answer carries `source` (repository, versions, licence, how to cite)
//      and `limitations` (what a structural check does and does not establish).
// - H2 The engine's status is relayed verbatim. `fail` and `incomplete` are never summarized
//      as a pass, and an `incomplete` result says "no result".
// - H3 Same answer as the engine: status, input checksum and links equal a direct dfm-core
//      call on the same package, and repeated calls give identical results.
// - H4 A plan VergeCommon stored reproduces exactly (input checksum, status, lost links).
// - H5 Read-only and offline: no tool writes a file, and the server imports no network module.
// - H6 Bounded: analyses run in a worker thread with a time limit; a call over the limit
//      returns an error and the server keeps answering.
// - H7 No tool takes or returns anything but the package it was given: data stays on the
//      user's machine.
import {readFile, stat} from 'node:fs/promises';
import {Worker} from 'node:worker_threads';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import * as core from '@viridis/dfm-core';
import {StdioServer, ToolError} from './stdio.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PKG = JSON.parse(await readFile(path.join(HERE, '..', 'package.json'), 'utf8'));
export const EXAMPLES_DIR = path.join(HERE, '..', 'examples');
export const REPO = 'https://github.com/jdhart81/hdfm-framework';
export const MAX_FILE_BYTES = 25 * 1024 * 1024;

export const LIMITATIONS = [
  'Structural connectivity only: a pass means corridors stay connected at the recorded minimum width. It does not establish species movement, genetic exchange, regulatory compliance, carbon outcomes or old-growth condition.',
  'A spine is a draft of old-growth candidates for a steward to review, not a finding that any stand is old growth.',
  'A result is only as good as its inputs; relay the warnings about missing sources, dates and reuse terms.',
  'Treat an "incomplete" result as no result, and report its reasons.',
];

const SOURCE = {
  repository: REPO,
  package: PKG.name,
  version: PKG.version,
  engine: {connectivity: core.ENGINE_VERSION, spine: core.SPINE_VERSION, outlook: core.OUTLOOK_VERSION, landscapePackage: core.SCHEMA_VERSION},
  license: 'MIT',
  cite: `${REPO}/blob/main/CITATION.cff`,
  method: 'https://dendriticforest.com/',
};

const answer = body => ({...body, source: SOURCE, limitations: LIMITATIONS});
const SLUG = /^[a-z0-9][a-z0-9-]{0,63}$/;

// -- inputs ---------------------------------------------------------------------
const INPUT_PROPS = {
  package: {type: 'object', description: 'A DFM Landscape Package v1 (dfm_package "1.0", WGS84 GeoJSON layers). See https://dendriticforest.com/data-format/'},
  path: {type: 'string', description: 'Path to a Landscape Package .json file on this machine (read, never written).'},
  example: {type: 'string', description: 'Name of a bundled example from list_examples.'},
};
const ONE_OF = 'Give exactly one of package, path or example.';

async function readJsonFile(p) {
  if (typeof p !== 'string' || !p.toLowerCase().endsWith('.json')) throw new ToolError('path must name a .json file');
  const full = path.resolve(p);
  let info;
  try { info = await stat(full); } catch { throw new ToolError(`no such file: ${p}`); }
  if (!info.isFile()) throw new ToolError(`not a file: ${p}`);
  if (info.size > MAX_FILE_BYTES) throw new ToolError(`file is over ${MAX_FILE_BYTES / 1024 / 1024} MB; analyze one watershed at a time`);
  try { return JSON.parse(await readFile(full, 'utf8')); } catch (e) { throw new ToolError(`not valid JSON: ${e.message}`); }
}

export async function loadExample(name) {
  if (typeof name !== 'string' || !SLUG.test(name)) throw new ToolError('example must be a name from list_examples');
  try { return JSON.parse(await readFile(path.join(EXAMPLES_DIR, `${name}.json`), 'utf8')); }
  catch { throw new ToolError(`unknown example: ${name}`); }
}

/** Resolve the tool arguments to dfm-core input. Accepts a bare package or {package: ...}. */
export async function loadInput(args) {
  const given = ['package', 'path', 'example'].filter(k => args[k] !== undefined);
  if (given.length !== 1) throw new ToolError(ONE_OF);
  let doc = args.package ?? (args.path !== undefined ? await readJsonFile(args.path) : (await loadExample(args.example)).package);
  if (doc && !doc.dfm_package && doc.package?.dfm_package) doc = doc.package;
  if (!doc || typeof doc !== 'object' || Array.isArray(doc)) throw new ToolError('package must be a JSON object');
  try {
    return {input: core.fromLandscapePackage(doc), name: doc.name ?? null};
  } catch (e) {
    throw new ToolError(`not a DFM Landscape Package v1: ${e.message}`);
  }
}

// -- worker (H6) ----------------------------------------------------------------
export const timeoutMs = () => Number(process.env.DFM_MCP_TIMEOUT_MS) || 120000;

export function runInWorker(fn, input, opts) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./worker.mjs', import.meta.url), {workerData: {fn, input, opts}});
    const limit = timeoutMs();
    const timer = setTimeout(() => {
      worker.terminate();
      reject(new ToolError(`analysis stopped after ${limit / 1000} s; analyze a smaller area, or for spine_network pass cuts: false`));
    }, limit);
    worker.once('message', msg => { clearTimeout(timer); worker.terminate(); msg.ok ? resolve(msg.value) : reject(new Error(msg.error)); });
    worker.once('error', e => { clearTimeout(timer); reject(e); });
  });
}

/** Drop bulky geometry unless asked for it; everything else is the engine's result, unchanged. */
function trim(result, includeGeometry) {
  if (includeGeometry) return result;
  const {geometry, features, ...rest} = result;
  if (features) rest.features = features.map(f => ({properties: f.properties}));
  if (geometry || features) rest.geometryOmitted = 'Pass include_geometry: true for GeoJSON.';
  return rest;
}

const ids = links => links.map(l => `${l.a}–${l.b}`).join(', ');

export function summarizeCheck(r) {
  if (r.status === 'incomplete') return `No result: the input is incomplete (${r.reasons.length} reason${r.reasons.length === 1 ? '' : 's'}). ${r.reasons[0] ?? ''}`.trim();
  const width = r.parameters?.minWidthM;
  if (r.status === 'fail') {
    if (r.lostLinks?.length) {
      const causes = [...new Set(r.lostLinks.flatMap(l => l.causes ?? []))];
      return `Fails: the plan breaks ${r.lostLinks.length} link${r.lostLinks.length === 1 ? '' : 's'} between core areas (${ids(r.lostLinks)})${causes.length ? `; caused by ${causes.join(', ')}` : ''}.`;
    }
    return `Fails: ${r.reasons[0] ?? 'the plan overlaps retained habitat without permission'}.`;
  }
  if (r.status === 'pass') {
    const pinched = r.pinchedLinks?.length ? ` ${r.pinchedLinks.length} link${r.pinchedLinks.length === 1 ? '' : 's'} hold only within 10% of the minimum (${ids(r.pinchedLinks)}).` : '';
    return `Passes the structural check: every link between core areas holds at the minimum width${width ? ` of ${width} m` : ''}.${pinched}`;
  }
  return `Engine status: ${r.status}.`;
}

const statusSummary = (r, okText) => (r.status === 'incomplete'
  ? `No result: the input is incomplete. ${r.reasons?.[0] ?? ''}`.trim()
  : okText(r));

// -- server ---------------------------------------------------------------------
export function buildServer() {
  const server = new StdioServer({
    name: 'dfm',
    version: PKG.version,
    instructions: 'Dendritic Forest Management: structural corridor checks and old-growth spine planning on a DFM Landscape Package. '
      + 'Start with list_examples and check_plan on an example. Relay every answer\'s limitations and warnings; '
      + 'a pass is structural connectivity only, and "incomplete" is no result.',
  });
  const withGeometry = {include_geometry: {type: 'boolean', description: 'Include GeoJSON geometry in the answer (default false).'}};
  const inputTool = (extra = {}) => ({type: 'object', properties: {...INPUT_PROPS, ...withGeometry, ...extra}, additionalProperties: false});

  server.tool('describe_method', {
    title: 'Describe the DFM method',
    description: 'What a DFM corridor check and the old-growth spine establish and do not establish, the input format, versions and how to cite.',
    inputSchema: {type: 'object', properties: {}, additionalProperties: false},
  }, async () => answer({
    what: 'Checks a treatment or harvest plan against a landscape\'s retained corridors and reports which links between core areas it would break or pinch; drafts an old-growth spine along streams, valleys and ridges, tests it as a network and finds the next holdings that would extend it.',
    input: 'DFM Landscape Package v1: one JSON object of WGS84 GeoJSON layers (parcels, coreAreas, retained, roads, water, crossings, treatments, optional streams, connectors, exits) with params.minWidthM and its source.',
    statuses: {pass: 'every link holds at the minimum width', fail: 'a link is lost, or a unit overlaps retained habitat without permission', incomplete: 'no result; see reasons', ok: 'spine analysis completed'},
    tools: [...server.tools.keys()],
    docs: {engine: `${REPO}/blob/main/packages/dfm-core/README.md`, dataFormat: 'https://dendriticforest.com/data-format/', researchStatus: `${REPO}/blob/main/docs/RESEARCH_STATUS.md`, agents: `${REPO}/blob/main/AGENTS.md`},
    coops: 'Neighbors can run the same check on every harvest plan through a woodland co-op on VergeCommon: https://vergecommon.com/woodland/',
  }));

  server.tool('list_examples', {
    title: 'List bundled examples',
    description: 'Fictional and synthetic landscapes bundled with the server, to try every tool without your own data.',
    inputSchema: {type: 'object', properties: {}, additionalProperties: false},
  }, async () => {
    const {readdir} = await import('node:fs/promises');
    const names = (await readdir(EXAMPLES_DIR)).filter(f => f.endsWith('.json')).map(f => f.slice(0, -5)).sort();
    const examples = [];
    for (const n of names) {
      const doc = await loadExample(n);
      examples.push({name: n, title: doc.package.name, about: doc.about, tryWith: doc.tryWith});
    }
    return answer({examples});
  });

  server.tool('get_example', {
    title: 'Get a bundled example package',
    description: 'The full Landscape Package for a bundled example, as a template for your own data.',
    inputSchema: {type: 'object', properties: {example: INPUT_PROPS.example}, required: ['example'], additionalProperties: false},
  }, async ({example}) => {
    const doc = await loadExample(example);
    return answer({name: example, about: doc.about, package: doc.package});
  });

  server.tool('check_plan', {
    title: 'Check a plan against retained corridors',
    description: `Does the proposed treatment or harvest plan break or pinch a link between core areas? Returns pass, fail or incomplete, naming each unit that breaks a link. ${ONE_OF}`,
    inputSchema: inputTool(),
  }, async (args) => {
    const {input, name} = await loadInput(args);
    const r = await runInWorker('check', input);
    return answer({landscape: name, status: r.status, summary: summarizeCheck(r), result: trim(r, args.include_geometry)});
  });

  server.tool('draft_spine', {
    title: 'Draft the old-growth spine',
    description: `Draft retained-habitat corridors along the package's streams and connector lines, each at least the minimum width, for a steward to review. ${ONE_OF}`,
    inputSchema: inputTool(),
  }, async (args) => {
    const {input, name} = await loadInput(args);
    const r = await runInWorker('deriveSpine', input);
    return answer({landscape: name, status: r.status,
      summary: statusSummary(r, x => `Drafted ${x.features.length} corridor section${x.features.length === 1 ? '' : 's'} for review.`),
      result: trim(r, args.include_geometry)});
  });

  server.tool('spine_network', {
    title: 'Test the spine as a network',
    description: `Links, loops and single points of failure of the spine; rho2 is the share of core pairs no single disturbance can separate. A full search can take tens of seconds; cuts: false skips it. ${ONE_OF}`,
    inputSchema: inputTool({
      cuts: {type: 'boolean', description: 'Search for single points of failure (default true). Robustness is null when skipped.'},
      cut_budget: {type: 'integer', minimum: 100, maximum: 200000, description: 'Most disturbances to test before reporting robustness as null (default 20000).'},
    }),
  }, async (args) => {
    const {input, name} = await loadInput(args);
    const r = await runInWorker('spineNetwork', input, {cuts: args.cuts ?? true, cutBudget: args.cut_budget ?? 20000});
    return answer({landscape: name, status: r.status,
      summary: statusSummary(r, x => `${x.summary.linkedPairs} of ${x.summary.totalPairs} core pairs linked; robustness ${x.summary.rho2 === null ? 'not computed' : `rho2 = ${x.summary.rho2}`}.`),
      result: trim(r, args.include_geometry)});
  });

  server.tool('project_spine', {
    title: 'Project the spine forward in time',
    description: `At each milestone year, the core pairs linked through committed habitat and through habitat at old-growth age. Needs stand ages or remnant records and params.ageAsOfYear. ${ONE_OF}`,
    inputSchema: inputTool({years: {type: 'array', items: {type: 'integer'}, maxItems: 12, description: 'Milestone years (default from params.milestoneYears).'}}),
  }, async (args) => {
    const {input, name} = await loadInput(args);
    const r = await runInWorker('projectSpine', input, args.years ? {years: args.years} : {});
    return answer({landscape: name, status: r.status, summary: statusSummary(r, () => 'Projection complete; ages are reported, never condition.'), result: trim(r, args.include_geometry)});
  });

  server.tool('climate_routes', {
    title: 'Find climate routes',
    description: `For each core area, the coolest core or exit it can reach through links that hold after the plan. Needs temp_c on cores. ${ONE_OF}`,
    inputSchema: inputTool(),
  }, async (args) => {
    const {input, name} = await loadInput(args);
    const r = await runInWorker('climateRoutes', input);
    return answer({landscape: name, status: r.status, summary: statusSummary(r, () => 'Climate routes computed through links that hold after the plan.'), result: trim(r, args.include_geometry)});
  });

  server.tool('next_holdings', {
    title: 'Find the next holdings to join',
    description: `Parcels without consent whose spine touches the committed spine, ordered by the core links each would complete if it joined. ${ONE_OF}`,
    inputSchema: inputTool(),
  }, async (args) => {
    const {input, name} = await loadInput(args);
    const r = await runInWorker('buildOutFrontier', input);
    return answer({landscape: name, status: r.status,
      summary: statusSummary(r, x => (x.frontier.length ? `${x.frontier.length} holding${x.frontier.length === 1 ? '' : 's'} would extend the committed spine; first: ${x.frontier[0].parcel}.` : 'No holding without consent touches the committed spine.')),
      result: trim(r, args.include_geometry)});
  });

  return server;
}
