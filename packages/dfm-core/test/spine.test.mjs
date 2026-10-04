// Old-growth spine: derivation and network. Each test names the invariant in src/spine.mjs it checks.
// Tests marked "review" reproduce findings from the independent review of 2026-10-04.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import Ajv2020 from 'ajv/dist/2020.js';
import {deriveSpine, spineNetwork, checkConnectivitySync, canonicalJSON, toLandscapePackage, fromLandscapePackage} from '../src/index.mjs';
import {effectiveHabitat, linkedPairs} from '../src/connectivity.mjs';
import {diskPolygon} from '../src/robust.mjs';
import {union, difference} from '../src/geo.mjs';
import {pt, rect} from '../fixtures/woodlot.mjs';
import {watershed, withAges} from '../fixtures/watershed.mjs';

const schema = JSON.parse(await readFile(new URL('../../dfm-schema/spine-results.schema.json', import.meta.url), 'utf8'));
const ajv = new Ajv2020({allErrors: true, strict: false});
ajv.addSchema(schema);
const valid = (def, r) => { const v = ajv.getSchema(`${schema.$id}#/$defs/${def}`); assert.ok(v(r), JSON.stringify(v.errors)); };
const line = (coords, properties) => ({type: 'Feature', geometry: {type: 'LineString', coordinates: coords.map(([x, y]) => pt(x, y))}, properties});
const core = (dfmId, x0, y0, x1, y1) => ({type: 'Feature', geometry: rect(x0, y0, x1, y1), properties: {dfm_id: dfmId, core_class: 'old-growth-candidate'}});
const keys = list => (list ?? []).map(p => `${p.a}|${p.b}`).sort();
const params = () => ({minWidthM: 100, minWidthSource: 'Test value', spineWidthByOrderM: {1: 115, 2: 140}, spineWidthSource: 'Test widths'});

/** A Y of three streams (local meters) with a core at each tip: a pure tree. */
function yShape({link = null, road = null, crossing = null} = {}) {
  const links = {
    // From the middle of a to the middle of b: a loop in the line graph that no core sits on.
    mid: [[250, 750], [500, 850], [750, 750]],
    // From the middle of a to 14 m short of b, away from every core: joins nothing.
    loose: [[250, 750], [500, 850], [750, 770]],
  };
  return {
    streams: [
      line([[0, 1000], [500, 500]], {dfm_id: 'a', stream_order: 1}),
      line([[1000, 1000], [500, 500]], {dfm_id: 'b', stream_order: 1}),
      line([[500, 500], [500, 0]], {dfm_id: 'c', stream_order: 2}),
    ],
    connectors: link ? [line(links[link], {dfm_id: 'ridge-ab', kind: 'ridge'})] : [],
    coreAreas: [core('core-a', -80, 980, 80, 1100), core('core-b', 920, 980, 1080, 1100), core('core-c', 420, -120, 580, 20)],
    roads: road ? [line([[0, 250], [1000, 250]], {dfm_id: 'road-1', width_m: 6})] : [],
    crossings: crossing ? [{type: 'Feature', geometry: {type: 'Point', coordinates: pt(500, 250)}, properties: {dfm_id: 'x-1', passage: crossing}}] : [],
    params: params(),
  };
}

/** Cores a and b sit on a loop: the stream route and a ridge route enter each core 200 m apart. */
function looped() {
  return {
    streams: [
      line([[0, 1000], [500, 500]], {dfm_id: 'a', stream_order: 1}),
      line([[1000, 1000], [500, 500]], {dfm_id: 'b', stream_order: 1}),
      line([[500, 500], [500, 0]], {dfm_id: 'c', stream_order: 2}),
    ],
    connectors: [line([[0, 1200], [500, 1350], [1000, 1200]], {dfm_id: 'ridge-ab', kind: 'ridge'})],
    coreAreas: [core('core-a', -200, 960, 60, 1260), core('core-b', 940, 960, 1200, 1260), core('core-c', 420, -120, 580, 20)],
    params: params(),
  };
}

const ws = watershed();
const derived = deriveSpine(ws);
const net = spineNetwork(ws);

test('SP1 SP2 SP4 derived sections: widths from the table by order, never below the minimum, provenance recorded, no core class', () => {
  assert.equal(derived.status, 'ok', derived.reasons.join(' '));
  valid('derivation', derived);
  assert.equal(derived.features.length, 14);
  const width = Object.fromEntries(derived.features.map(f => [f.properties.source_id, f.properties.width_m]));
  assert.equal(width.h1, 115); assert.equal(width['main-upper'], 140); assert.equal(width.main, 180); assert.equal(width['saddle-nw'], 115);
  for (const f of derived.features) {
    const p = f.properties;
    assert.ok(p.width_m >= ws.params.minWidthM, `${p.dfm_id} is at least the minimum width`);
    assert.equal(p.spine, true);
    assert.ok(['stream', 'saddle'].includes(p.origin));
    assert.ok(p.width_source.length > 0);
    assert.equal(p.core_class, undefined, 'derivation never assigns a core class (I6)');
  }
});

test('SP1 a derived section carries a minimum-width link: cores at both ends of one stream are linked by the check', () => {
  const input = {streams: [line([[0, 500], [1000, 500]], {dfm_id: 's', stream_order: 1})], params: {minWidthM: 100, minWidthSource: 'Test value', spineWidthByOrderM: {1: 115}, spineWidthSource: 'Test'}};
  const d = deriveSpine(input);
  const check = checkConnectivitySync({coreAreas: [core('w', -150, 400, 0, 600), core('e', 1000, 400, 1150, 600)], retained: d.features, params: {minWidthM: 100, minWidthSource: 'Test value'}});
  assert.equal(check.status, 'pass', check.reasons.join(' '));
  assert.deepEqual(keys(check.linkedAfter), ['e|w']);
  assert.deepEqual(check.pinchedLinks, []);
});

test('SP1 with no width table every section uses the minimum plus the pinch margin, rounded up past the next 5 m, and says so', () => {
  const input = yShape();
  delete input.params.spineWidthByOrderM; delete input.params.spineWidthSource;
  const d = deriveSpine(input);
  assert.equal(d.status, 'ok');
  assert.deepEqual([...new Set(d.features.map(f => f.properties.width_m))], [115]);
  assert.match(d.features[0].properties.width_source, /^Default: the 100 m minimum width plus the 10% pinch margin/);
  assert.match(d.warnings.join(' '), /No spine width table is recorded/);
  const check = checkConnectivitySync({coreAreas: input.coreAreas, retained: d.features, params: input.params});
  assert.equal(check.status, 'pass');
  assert.deepEqual(check.pinchedLinks, [], 'the default clears the pinch margin');
});

test('SP1 review: a width at the minimum warns that no link holds; one within the pinch margin warns of pinch points', () => {
  const exact = yShape(); exact.params.spineWidthByOrderM = {1: 100, 2: 140};
  const r = deriveSpine(exact);
  assert.equal(r.status, 'ok');
  assert.match(r.warnings.join(' '), /\["1"\] is exactly the 100 m minimum: the check finds no link through a corridor exactly that wide/);
  const near = yShape(); near.params.spineWidthByOrderM = {1: 105, 2: 140};
  assert.match(deriveSpine(near).warnings.join(' '), /\["1"\] is 105 m, within the 10% pinch margin/);
});

test('SP2 SP3 the width table: narrowing with order, below the minimum, a missing order, no source, or a non-whole key is incomplete', () => {
  const cases = [
    [{1: 150, 2: 120}, /must not decrease with stream order: order 2 \(120 m\) is narrower than order 1 \(150 m\)/],
    [{1: 90, 2: 120}, /"1"\] is 90 m, narrower than the 100 m minimum/],
    [{2: 140}, /Stream a is order 1, but params.spineWidthByOrderM has no width for order 1 or below/],
    [{'1.0': 115, 2: 140}, /key "1.0" must be a whole stream order/],
  ];
  for (const [table, reason] of cases) {
    const input = yShape(); input.params.spineWidthByOrderM = table;
    const r = deriveSpine(input);
    assert.equal(r.status, 'incomplete'); assert.deepEqual(r.features, []);
    assert.match(r.reasons.join(' '), reason);
  }
  const noSource = yShape(); delete noSource.params.spineWidthSource;
  assert.match(deriveSpine(noSource).reasons.join(' '), /spineWidthSource must record/);
});

test('SP3 streams need an integer order, links a known kind, and lines at least 1 m long; otherwise incomplete, naming the line', () => {
  const input = yShape({link: 'mid'});
  input.streams[0].properties.stream_order = 1.5;
  delete input.streams[1].properties.stream_order;
  input.connectors[0].properties.kind = 'road';
  const r = deriveSpine(input);
  assert.equal(r.status, 'incomplete');
  assert.match(r.reasons.join(' '), /Stream a needs an integer stream_order/);
  assert.match(r.reasons.join(' '), /Stream b needs an integer stream_order/);
  assert.match(r.reasons.join(' '), /Connector ridge-ab needs a kind: ridge, valley, saddle/);
  const dupe = yShape(); dupe.streams[1].properties.dfm_id = 'a';
  assert.match(deriveSpine(dupe).reasons.join(' '), /IDs must be unique/);
  const dot = yShape(); dot.streams.push(line([[200, 200], [200, 200]], {dfm_id: 'dot', stream_order: 1}));
  assert.match(deriveSpine(dot).reasons.join(' '), /Line dot has a part shorter than 1 m/);
});

test('I7 review: malformed inputs return incomplete with plain reasons, never a thrown error', () => {
  for (const bad of [{streams: {}}, {streams: [{type: 'Feature', geometry: {type: 'LineString'}, properties: {}}]}, {connectors: 'x'}, {params: 3}, null]) {
    const d = deriveSpine(bad);
    assert.equal(d.status, 'incomplete');
    const n = spineNetwork(bad);
    assert.equal(n.status, 'incomplete');
  }
  assert.match(deriveSpine({streams: {}}).reasons.join(' '), /streams must be an array of GeoJSON features/);
  assert.match(spineNetwork({...yShape(), coreAreas: {}}).reasons.join(' '), /coreAreas must be an array/);
});

test('I8 derivation is deterministic and independent of input order', () => {
  const a = deriveSpine(watershed()), b = deriveSpine(watershed());
  assert.equal(canonicalJSON(a), canonicalJSON(b));
  const reversed = watershed(); reversed.streams.reverse(); reversed.connectors.reverse();
  assert.deepEqual(deriveSpine(reversed).features.map(f => f.properties.dfm_id), a.features.map(f => f.properties.dfm_id));
});

test('SP4 review: any derived corridor that overlaps mapped open water warns, even when its centerline misses the water', () => {
  const input = yShape({link: 'mid'});
  input.water = [{type: 'Feature', geometry: rect(530, 100, 560, 300), properties: {dfm_id: 'pond'}}]; // beside c, inside its corridor
  input.water.push({type: 'Feature', geometry: rect(480, 820, 520, 860), properties: {dfm_id: 'tarn'}}); // on the ridge link
  const w = deriveSpine(input).warnings.join(' ');
  assert.match(w, /corridor of stream c overlaps mapped open water pond/);
  assert.match(w, /corridor of link ridge-ab overlaps mapped open water tarn/);
});

test('SP5 SP7 a pure tree: no loops, every section between cores fails alone, rho2 = 0', () => {
  const r = spineNetwork(yShape());
  assert.equal(r.status, 'ok', r.reasons.join(' '));
  valid('network', r);
  assert.equal(r.summary.dendritic, true);
  assert.equal(r.summary.loops, 0);
  assert.equal(r.summary.loopsThroughCores, 0);
  assert.equal(r.summary.rho2, 0);
  assert.equal(r.summary.pFail1, 1);
  assert.ok(r.summary.exposedShare > 0.95);
  assert.deepEqual(keys(r.coreLinks), ['core-a|core-b', 'core-a|core-c', 'core-b|core-c']);
  const zone = lines => r.singlePointsOfFailure.find(f => f.nearLines.join('+') === lines);
  assert.deepEqual(keys(zone('a').separates), ['core-a|core-b', 'core-a|core-c'], 'a branch zone cuts its core off');
  assert.deepEqual(keys(zone('c').separates), ['core-a|core-c', 'core-b|core-c']);
  assert.equal(keys(zone('a+b+c').separates).length, 3, 'the confluence separates all three');
  assert.ok(r.singlePointsOfFailure.every(f => f.verified), 'each zone is confirmed exactly at nominal width');
});

test('SP7 SP8 a loop in the line graph does not make pairs robust when their cores hang off it', () => {
  const r = spineNetwork(yShape({link: 'mid'}));
  assert.equal(r.summary.loops, 1);
  assert.equal(r.summary.dendritic, false);
  assert.equal(r.summary.robustPairs, 0, 'the stretches from each core to the loop can still be cut alone');
  assert.ok(r.singlePointsOfFailure.some(f => f.nearLines.includes('a')));
});

test('SP7 cores on a loop, with routes entering each core far apart, are robust; the stem to the third core is not', () => {
  const r = spineNetwork(looped());
  assert.equal(r.status, 'ok', r.reasons.join(' '));
  assert.equal(r.summary.loops, 0, 'the ridge route meets the streams only through the cores');
  assert.equal(r.summary.loopsThroughCores, 1);
  assert.deepEqual(r.coreLinks.filter(x => x.robust).map(x => `${x.a}|${x.b}`), ['core-a|core-b']);
  assert.ok(Math.abs(r.summary.rho2 - 1 / 3) < 1e-12);
  assert.ok(r.singlePointsOfFailure.every(f => !keys(f.separates).includes('core-a|core-b')));
  assert.ok(r.singlePointsOfFailure.some(f => f.nearLines.includes('c')));
});

test('SP6 review: two routes are robust only when one disturbance cannot cut both, wherever it lands', () => {
  const pair = d => ({
    streams: [line([[0, 500], [1000, 500]], {dfm_id: 's1', stream_order: 1}), line([[0, 500 + d], [1000, 500 + d]], {dfm_id: 's2', stream_order: 1})],
    coreAreas: [core('core-w', -400, 300, 0, 1100), core('core-e', 1000, 300, 1400, 1100)], params: params(),
  });
  for (const d of [120, 190]) {
    const r = spineNetwork(pair(d));
    assert.equal(r.coreLinks[0].robust, false, `${d} m apart: a disturbance between them nicks both corridors below the minimum`);
    assert.ok(r.singlePointsOfFailure.some(f => f.verified && f.nearLines.join() === 's1,s2'));
    assert.equal(r.summary.exposedShare, 1);
  }
  assert.equal(spineNetwork(pair(400)).coreLinks[0].robust, true, '400 m apart: no single disturbance reaches both');
  const small = pair(120); small.params.disturbanceWidthM = 20;
  assert.equal(spineNetwork(small).coreLinks[0].robust, true, 'a 20 m disturbance between them leaves both corridors above the minimum');
});

test('SP6 review: sections the line graph marks as severed are still tested, because the habitat can carry them', () => {
  // One stream runs from core a through core k to core b; a road crosses k with no crossing recorded.
  const r = spineNetwork({
    streams: [line([[0, 500], [2400, 500]], {dfm_id: 's', stream_order: 1})],
    coreAreas: [core('core-a', -300, 300, 0, 700), core('core-k', 1000, 300, 1400, 700), core('core-b', 2400, 300, 2700, 700)],
    roads: [line([[1200, 0], [1200, 1000]], {dfm_id: 'town-road', width_m: 6})], params: params(),
  });
  assert.ok(r.severed.length > 0, 'the line graph marks the stream as cut by the road');
  assert.deepEqual(keys(r.coreLinks), ['core-a|core-k', 'core-b|core-k'], 'the check links each side to the core');
  assert.ok(r.coreLinks.every(x => x.robust === false), 'a disturbance on either stretch separates its pair');
  assert.ok(r.singlePointsOfFailure.every(f => f.nearLines.includes('s')));
});

test('SP6 review: a disturbance may overlap a core but does not remove core habitat', () => {
  // Two routes 400 m apart join two narrow cores; disks over the cores must not count as failures.
  const r = spineNetwork({
    streams: [line([[0, 500], [1000, 500]], {dfm_id: 's1', stream_order: 1}), line([[0, 900], [1000, 900]], {dfm_id: 's2', stream_order: 1})],
    coreAreas: [core('core-w', -130, 400, 0, 1000), core('core-e', 1000, 400, 1130, 1000)], params: params(),
  });
  assert.equal(r.coreLinks[0].robust, true);
  assert.deepEqual(r.singlePointsOfFailure, []);
});

test('SP7 review: a second line drawn 40 m from the stream is not a second route', () => {
  const r = spineNetwork({
    streams: [line([[0, 500], [1000, 500]], {dfm_id: 's', stream_order: 1})],
    connectors: [line([[0, 500], [100, 540], [900, 540], [1000, 500]], {dfm_id: 'v', kind: 'valley'})],
    coreAreas: [core('core-w', -150, 400, 0, 600), core('core-e', 1000, 400, 1150, 600)], params: params(),
  });
  assert.equal(r.summary.loops, 1, 'the line graph has a loop');
  assert.equal(r.summary.robustPairs, 0, 'but one cut severs both corridors');
  assert.deepEqual(r.coreLinks.map(x => x.robust), [false]);
});

test('SP5 review: a road that crosses the stream twice needs a crossing at both places', () => {
  const base = {
    streams: [line([[0, 500], [1000, 500]], {dfm_id: 's', stream_order: 1})],
    coreAreas: [core('core-w', -150, 400, 0, 600), core('core-e', 1000, 400, 1150, 600)],
    roads: [line([[400, 0], [400, 900], [600, 900], [600, 0]], {dfm_id: 'hairpin', width_m: 6})], params: params(),
  };
  const x = (at, passage) => ({type: 'Feature', geometry: {type: 'Point', coordinates: pt(at, 500)}, properties: {dfm_id: `x${at}`, passage}});
  const one = spineNetwork({...base, crossings: [x(400, 'verified')]}, {cuts: false});
  assert.deepEqual(one.coreLinks, [], 'no link, as the check finds');
  assert.deepEqual(one.severed.map(s => s.roads), [['hairpin']]);
  const both = spineNetwork({...base, crossings: [x(400, 'verified'), x(600, 'verified')]}, {cuts: false});
  assert.deepEqual(keys(both.coreLinks), ['core-e|core-w']);
  assert.deepEqual(both.severed, []);
  const check = checkConnectivitySync({...base, retained: deriveSpine(base).features, crossings: [x(400, 'verified')]});
  assert.deepEqual(check.linkedAfter, [], 'the network agrees with the check');
});

test('SP5 review: road widths of zero or below are invalid, not ignored', () => {
  const zero = yShape({road: true}); zero.roads[0].properties.width_m = 0;
  assert.match(spineNetwork(zero).reasons.join(' '), /roads\[0\] properties.width_m must be between 1 and 100 m/);
  const param = yShape({road: true}); delete param.roads[0].properties.width_m; param.params.roadWidthM = 0;
  assert.match(spineNetwork(param).reasons.join(' '), /params.roadWidthM must be between 1 and 100 m/);
});

test('SP5 a road severs a section unless a crossing on that road carries it; assumed crossings warn; the result says why', () => {
  const cut = spineNetwork(yShape({road: true}), {cuts: false});
  assert.deepEqual(cut.severed.map(s => [s.sourceId, s.roads]), [['c', ['road-1']]]);
  assert.deepEqual(keys(cut.coreLinks), ['core-a|core-b'], 'core c is cut off by the road');
  const verified = spineNetwork(yShape({road: true, crossing: 'verified'}), {cuts: false});
  assert.deepEqual(verified.severed, []);
  assert.equal(verified.coreLinks.length, 3);
  const assumed = spineNetwork(yShape({road: true, crossing: 'assumed'}), {cuts: false});
  assert.deepEqual(assumed.severed, []);
  assert.match(assumed.warnings.join(' '), /Crossing x-1 on road road-1 carries spine line c; its passage is assumed/);
  assert.equal(spineNetwork(yShape({road: true, crossing: 'none'}), {cuts: false}).severed.length, 1, "a crossing recorded as 'none' carries nothing");
});

test('SP5 a line that stops short of another does not join it; lines that cross mid-way are reported', () => {
  const loose = spineNetwork(yShape({link: 'loose'}), {cuts: false});
  assert.equal(loose.summary.loops, 0);
  assert.match(loose.warnings.join(' '), /Line ridge-ab ends 14\.\d m from line b, beyond the 5 m junction distance/);
  const crossing = yShape(); crossing.connectors = [line([[100, 300], [900, 300]], {dfm_id: 'across', kind: 'valley'})];
  assert.match(spineNetwork(crossing, {cuts: false}).warnings.join(' '), /Lines c and across cross without a junction|Lines across and c cross without a junction/);
});

test('SP6 skipping the cut tests reports robustness as unknown, never as true', () => {
  const r = spineNetwork(yShape(), {cuts: false});
  valid('network', r);
  assert.equal(r.summary.rho2, null);
  assert.equal(r.singlePointsOfFailure, null);
  assert.ok(r.coreLinks.every(x => x.robust === null));
  assert.match(r.warnings.join(' '), /Cut tests were skipped/);
});

test('SP6 SP7 the watershed: the main stem between the tributaries and the outlet stretch fail alone; only north-west is robust', () => {
  assert.equal(net.status, 'ok', net.reasons.join(' '));
  valid('network', net);
  assert.equal(net.summary.linkedPairs, 6);
  assert.equal(net.summary.loops, 2, 'the two saddle links close two loops in the line graph');
  assert.deepEqual(net.coreLinks.filter(p => p.robust).map(p => `${p.a}|${p.b}`), ['core-n|core-w']);
  assert.ok(Math.abs(net.summary.rho2 - 1 / 6) < 1e-12);
  const main = net.singlePointsOfFailure.filter(f => f.nearLines.includes('main'));
  assert.ok(main.some(f => f.verified && keys(f.separates).join() === 'core-e|core-n,core-e|core-w,core-n|core-s,core-s|core-w'), 'between the tributaries: north and west from east and south');
  assert.ok(main.some(f => f.verified && keys(f.separates).join() === 'core-e|core-s,core-n|core-s,core-s|core-w'), 'the outlet stretch isolates the riparian core');
  assert.ok(net.summary.exposedShare > 0 && net.summary.exposedShare < 0.5);
  assert.deepEqual(net.severed, []);
  assert.equal(net.summary.disturbanceWidthM, 182, 'default: the widest corridor (the 180 m main stem) plus 2 m');
  assert.ok(net.summary.cutsTested > 500);
});

test('SP5 without the east crossing the east saddle link is severed and the east loop is gone', () => {
  const r = spineNetwork(watershed({crossingE: 'none'}), {cuts: false});
  assert.deepEqual([...new Set(r.severed.map(s => s.sourceId))], ['saddle-e']);
  assert.equal(r.summary.loops, 1);
});

test('SP3 I7 network inputs are validated: cores, road widths and crossing records', () => {
  const input = yShape({road: true}); input.coreAreas = input.coreAreas.slice(0, 1);
  delete input.roads[0].properties.width_m;
  input.crossings = [{type: 'Feature', geometry: {type: 'Point', coordinates: pt(500, 250)}, properties: {passage: 'maybe'}}];
  const r = spineNetwork(input);
  assert.equal(r.status, 'incomplete');
  assert.match(r.reasons.join(' '), /At least two core areas/);
  assert.match(r.reasons.join(' '), /roads\[0\] is a centerline: set params.roadWidthM/);
  assert.match(r.reasons.join(' '), /crossings\[0\] must be a point with passage/);
  valid('network', r);
});

test('I8 network results are deterministic', () => {
  assert.equal(canonicalJSON(spineNetwork(yShape({link: 'mid'}))), canonicalJSON(spineNetwork(yShape({link: 'mid'}))));
});

// Round-3 review regressions and an independent soundness check of "robust".

/** Two parallel order-1 streams `d` m apart between two wide cores, drawn at latitude `lat0`. */
function parallel(d, lat0 = 0) {
  const M = (2 * Math.PI * 6371008.8) / 360, at = (x, y) => [x / (M * Math.cos(lat0 * Math.PI / 180)), lat0 + y / M];
  const ln = (coords, properties) => ({type: 'Feature', geometry: {type: 'LineString', coordinates: coords.map(([x, y]) => at(x, y))}, properties});
  const box = (dfmId, x0, y0, x1, y1) => ({type: 'Feature', geometry: {type: 'Polygon', coordinates: [[at(x0, y0), at(x1, y0), at(x1, y1), at(x0, y1), at(x0, y0)]]}, properties: {dfm_id: dfmId, core_class: 'old-growth-candidate'}});
  return {
    streams: [ln([[0, 500], [1000, 500]], {dfm_id: 's1', stream_order: 1}), ln([[0, 500 + d], [1000, 500 + d]], {dfm_id: 's2', stream_order: 1})],
    coreAreas: [box('core-w', -400, 300, 0, 1100), box('core-e', 1000, 300, 1400, 1100)],
    params: params(),
    at,
  };
}
const strip = ({at, ...input}) => input;

/** Independent oracle: the check's own link rule on the habitat less one disk (applied outside the cores). */
function oracle(input) {
  const cores = input.coreAreas.map(f => ({id: f.properties.dfm_id, feature: f}));
  const {habitat} = effectiveHabitat({...input, retained: deriveSpine(input).features, treatments: []}, [], []);
  const coreUnion = union(input.coreAreas);
  return (center, rhoM) => {
    const cut = difference(diskPolygon(center, rhoM, {circumscribed: true}), coreUnion);
    return new Set(linkedPairs(cut ? difference(habitat, cut) : habitat, cores, input.params.minWidthM).pairs);
  };
}

test('SP6 SP7 review: a line shorter than the junction distance has no section, but its link is still tested; exposure is unknown, not zero', () => {
  // A 4 m stream bridges a 3 m gap between two cores.
  const r = spineNetwork({streams: [line([[-0.5, 500], [3.5, 500]], {dfm_id: 's', stream_order: 1})], coreAreas: [core('core-w', -150, 400, 0, 600), core('core-e', 3, 400, 153, 600)], params: params()});
  assert.equal(r.status, 'ok', r.reasons.join(' '));
  valid('network', r);
  assert.equal(r.summary.sections, 0);
  assert.equal(r.summary.disturbanceWidthM, 117, 'the width comes from the derived corridor, not from sections');
  assert.deepEqual(r.coreLinks, [{a: 'core-e', b: 'core-w', robust: false}], 'one disturbance over the gap separates the cores');
  assert.ok(r.singlePointsOfFailure.some(z => z.verified && z.nearLines.join() === 's'), 'the zone names the short line');
  assert.equal(r.summary.pFail1, null);
  assert.equal(r.summary.exposedShare, null);
  assert.match(r.warnings.join(' '), /Every spine line is shorter than the 5 m junction distance/);
});

test('SP6 review: the network analysis needs a minimum width of at least 5 m', () => {
  const input = yShape(); input.params.minWidthM = 3; input.params.spineWidthByOrderM = {1: 10, 2: 12};
  const r = spineNetwork(input);
  assert.equal(r.status, 'incomplete');
  assert.match(r.reasons.join(' '), /needs params.minWidthM of at least 5 m/);
  assert.equal(deriveSpine(input).status, 'ok', 'derivation itself accepts narrow widths');
});

test('SP6 review: a search that runs past its cut budget reports robustness as unknown, never as true', () => {
  const r = spineNetwork(strip(parallel(400)), {cutBudget: 10});
  assert.equal(r.status, 'ok');
  valid('network', r);
  assert.equal(r.summary.cutsTested, 10);
  for (const k of ['robustPairs', 'rho2', 'pFail1', 'exposedShare']) assert.equal(r.summary[k], null, k);
  assert.deepEqual(r.coreLinks.map(x => x.robust), [null]);
  assert.equal(r.singlePointsOfFailure, null);
  assert.match(r.warnings.join(' '), /needs more than 10 cuts at this size, so robustness is not reported/);
});

test('SP6 review: at 60° N the same layout gives the same answers as at the equator', () => {
  const near = spineNetwork(strip(parallel(120, 60)));
  assert.deepEqual(near.coreLinks.map(x => x.robust), [false]);
  assert.ok(near.singlePointsOfFailure.some(z => z.verified && z.nearLines.join() === 's1,s2'));
  assert.deepEqual(spineNetwork(strip(parallel(400, 60))).coreLinks.map(x => x.robust), [true]);
});

test('SP6 soundness: no sampled disturbance at nominal width separates a pair reported robust', () => {
  // Each case samples the places a disturbance would do most harm. Layouts that are robust check
  // the claim; layouts that are not (190 m apart, cores hanging off a loop) catch a search that
  // misses a weakness and claims robustness it does not have.
  const along = (lines, step) => lines.flatMap(ln => {
    const cs = ln.geometry.coordinates, out = [];
    for (let i = 1; i < cs.length; i++) for (let t = 0; t <= 1 + 1e-9; t += step) out.push([cs[i - 1][0] + t * (cs[i][0] - cs[i - 1][0]), cs[i - 1][1] + t * (cs[i][1] - cs[i - 1][1])]);
    return out;
  });
  const cases = [];
  for (const d of [230, 190]) {
    const two = parallel(d), centers = [];
    for (let x = -50; x <= 1050; x += 50) for (const y of [500, 500 + d / 2 - 2, 500 + d / 2, 500 + d / 2 + 2, 500 + d]) centers.push(two.at(x, y));
    cases.push([strip(two), centers, d === 230 ? ['core-e|core-w'] : []]);
  }
  const loop = looped();
  cases.push([loop, along([...loop.streams, ...loop.connectors], 0.1), ['core-a|core-b']]);
  const tree = yShape({link: 'mid'});
  cases.push([tree, along([...tree.streams, ...tree.connectors], 0.1), []]);
  for (const [input, centers, expected] of cases) {
    const net = spineNetwork(input);
    const robust = net.coreLinks.filter(x => x.robust).map(x => `${x.a}|${x.b}`);
    const after = oracle(input);
    for (const c of centers) {
      const linked = after(c, net.summary.disturbanceWidthM / 2);
      for (const k of robust) assert.ok(linked.has(k), `a disturbance at ${c.map(v => v.toFixed(6))} separates ${k}, reported robust`);
    }
    assert.deepEqual(robust, expected);
  }
});

test('SP6 the oracle agrees with reported failures: a verified zone separates its pairs at nominal width', () => {
  for (const input of [strip(parallel(190)), yShape()]) {
    const net = spineNetwork(input), after = oracle(input);
    const verified = net.singlePointsOfFailure.filter(z => z.verified);
    assert.ok(verified.length > 0);
    for (const z of verified) {
      const linked = after(z.location, net.summary.disturbanceWidthM / 2);
      for (const p of z.separates) assert.ok(!linked.has(`${p.a}|${p.b}`), `zone at ${z.location} separates ${p.a}|${p.b}`);
    }
  }
});

test('SP6 review: a corridor pinched just under the minimum width is no route, wherever the raster cells fall', () => {
  // Route A narrows to 19.8 m at two notches (minimum 20 m); route B is a detour, so one cut on B separates the cores.
  for (const dx of [1.0, 1.25]) {
    const a = (x, y) => pt(x + dx, y);
    const routeA = {type: 'Feature', geometry: {type: 'Polygon', coordinates: [[a(-15, -200), a(15, -200), a(15, -8), a(9.9, 0), a(15, 8), a(15, 200), a(-15, 200), a(-15, 8), a(-9.9, 0), a(-15, -8), a(-15, -200)]]}, properties: {dfm_id: 'route-a'}};
    const input = {
      coreAreas: [core('core-s', -100, -300, 100, -200), core('core-n', -100, 200, 100, 300)],
      retained: [routeA, {type: 'Feature', geometry: rect(100, -300, 160, 300), properties: {dfm_id: 'route-b'}}],
      params: {minWidthM: 20, minWidthSource: 'Test value', disturbanceWidthM: 80},
    };
    const r = spineNetwork(input);
    assert.equal(r.status, 'ok', r.reasons.join(' '));
    assert.deepEqual(r.coreLinks.map(x => x.robust), [false], `route A offset ${dx} m`);
  }
});

test('SP5 review: spine features saved in the retained layer are drawn from the lines, not counted again', () => {
  const input = yShape({link: 'mid'});
  const saved = {...input, retained: deriveSpine(input).features};
  const strip = r => ({...r, inputChecksum: null});
  assert.equal(canonicalJSON(strip(spineNetwork(saved))), canonicalJSON(strip(spineNetwork(input))));
});

test('Landscape Package: spine layers round-trip, and a package without them reads exactly as before', () => {
  const input = {...watershed(), retained: withAges(derived.features)};
  const pkg = toLandscapePackage(input, {created: '2026-10-04T00:00:00Z'});
  assert.equal(pkg.layers.streams.length, 12);
  assert.equal(pkg.layers.connectors.length, 2);
  const back = fromLandscapePackage(pkg);
  assert.equal(canonicalJSON(back.streams), canonicalJSON(input.streams));
  const plain = toLandscapePackage({...input, streams: [], connectors: []}, {created: '2026-10-04T00:00:00Z'});
  assert.ok(!('streams' in plain.layers) && !('connectors' in plain.layers), 'empty spine layers are not written');
  assert.ok(!('streams' in fromLandscapePackage(plain)), 'and not read back as empty keys, so checksums are unchanged');
});
