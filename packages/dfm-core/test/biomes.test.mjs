// Any biome: the engine works the same in forest, prairie and flat country. Each test names the
// invariant it checks (B1-B7 in src/connectivity.mjs, src/spine.mjs and src/outlook.mjs).
// The prairie results are computed once and shared.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import Ajv2020 from 'ajv/dist/2020.js';
import {
  deriveSpine, spineNetwork, projectSpine, climateRoutes, buildOutFrontier, checkConnectivitySync,
  toLandscapePackage, fromLandscapePackage, canonicalJSON, LIGHT_INTENSITIES, SPINE_LINK_KINDS,
} from '../src/index.mjs';
import {effectiveHabitat, linkedPairs} from '../src/connectivity.mjs';
import {diskPolygon} from '../src/robust.mjs';
import {union, difference} from '../src/geo.mjs';
import {pt, rect, woodlot} from '../fixtures/woodlot.mjs';
import {prairie, withPrairieAges, REMNANT_SOURCE} from '../fixtures/prairie.mjs';

const read = async name => JSON.parse(await readFile(new URL(`../../dfm-schema/${name}.schema.json`, import.meta.url), 'utf8'));
const results = await read('spine-results'), pkgSchema = await read('landscape-package');
const ajv = new Ajv2020({allErrors: true, strict: false});
ajv.addSchema(results);
const valid = (def, r) => { const v = ajv.getSchema(`${results.$id}#/$defs/${def}`); assert.ok(v(r), JSON.stringify(v.errors)); };
const keys = list => (list ?? []).map(p => `${p.a}|${p.b}`).sort();
const strip = r => ({...r, inputChecksum: null, parameters: null});
const ALL = ['marsh-e|preserve-s', 'marsh-e|remnant-n', 'marsh-e|remnant-w', 'preserve-s|remnant-n', 'preserve-s|remnant-w', 'remnant-n|remnant-w'];
const SOUTH = ['marsh-e|preserve-s', 'marsh-e|remnant-w', 'preserve-s|remnant-w'];

/** The prairie with its drafted spine (ages recorded) and patches saved as retained habitat. */
const plan = (opts = {}) => {
  const b = prairie(opts);
  return {...b, retained: [...withPrairieAges(deriveSpine(b).features), ...b.retained]};
};
const net = spineNetwork(prairie());
const checkAt = gap => checkConnectivitySync(plan({gap}));
const checks = Object.fromEntries([0, 40, 80, 100, 200].map(g => [g, checkAt(g)]));
const climate = climateRoutes(plan()), climateNoExit = climateRoutes(plan({exits: false})), climateNoGap = climateRoutes(plan({gap: 0}));
const byId = r => Object.fromEntries(r.cores.map(c => [c.id, c]));

test('B1 every function runs on a flat prairie with no ridges or valleys', () => {
  const d = deriveSpine(prairie());
  assert.equal(d.status, 'ok', d.reasons.join(' '));
  valid('derivation', d);
  assert.equal(net.status, 'ok', net.reasons.join(' '));
  valid('network', net);
  const p = projectSpine(plan(), {years: [2026, 2110]});
  assert.equal(p.status, 'ok', p.reasons.join(' '));
  valid('projection', p);
  assert.equal(climate.status, 'ok', climate.reasons.join(' '));
  valid('climate', climate);
  const f = buildOutFrontier(plan());
  assert.equal(f.status, 'ok', f.reasons.join(' '));
  valid('frontier', f);
});

test('B1 no rule reads a biome label: naming the biome changes nothing but the checksum', () => {
  const named = plan();
  named.params.biome = 'grassland';
  assert.equal(canonicalJSON(strip(checkConnectivitySync(named))), canonicalJSON(strip(checks[100])));
  assert.equal(canonicalJSON({...climateRoutes(named), inputChecksum: null}), canonicalJSON({...climate, inputChecksum: null}));
});

test('B2 stepping stones link patches only within the crossing distance', () => {
  assert.deepEqual(keys(checks[0].linkedBefore), SOUTH, 'corridors only: the remnant to the north is cut off by cropland');
  assert.match(checks[0].warnings.join(' '), /Cores remnant-n and remnant-w are not linked at 60 m in the current state/);
  assert.deepEqual(keys(checks[80].linkedBefore), SOUTH, 'the last gap, 90 m, is wider than 80 m');
  assert.match(checks[80].warnings.join(' '), /not linked at 60 m with stepping stones up to 80 m apart/);
  assert.deepEqual(keys(checks[100].linkedBefore), ALL, 'every gap in the chain is at most 90 m');
  assert.match(checks[100].limitations.join(' '), /stepping stones of habitat at least that wide up to 100 m apart \(Fixture value for tests: gaps grassland birds cross\)/);
});

test('B2 a longer crossing distance never removes a link', () => {
  const gaps = [0, 40, 80, 100, 200];
  for (let i = 1; i < gaps.length; i++) {
    const before = new Set(keys(checks[gaps[i - 1]].linkedBefore)), after = new Set(keys(checks[gaps[i]].linkedBefore));
    for (const k of before) assert.ok(after.has(k), `${k} is linked at ${gaps[i - 1]} m but not at ${gaps[i]} m`);
  }
});

test('B2 without a crossing distance the check is unchanged, and 0 means the same as unset', () => {
  for (const opts of [{}, {treatments: ['harvest-3']}, {corridorWidth: 99, road: false}]) {
    const unset = woodlot(opts), zero = woodlot(opts);
    zero.params.gapCrossingM = 0;
    assert.equal(canonicalJSON(strip(checkConnectivitySync(zero))), canonicalJSON(strip(checkConnectivitySync(unset))));
  }
  assert.equal(canonicalJSON(strip(checks[0])), canonicalJSON(strip(checkConnectivitySync({...plan({gap: 0}), params: {...plan({gap: 0}).params, gapCrossingM: 0}}))));
});

test('B2 the crossing distance needs a source and a value from 0 to 1,000 m', () => {
  const cases = [[-1, 'x', /from 0 to 1,000/], [1001, 'x', /from 0 to 1,000/], ['50', 'x', /from 0 to 1,000/], [100, undefined, /gapCrossingSource must record/]];
  for (const [gap, source, reason] of cases) {
    const input = plan();
    input.params.gapCrossingM = gap;
    if (source === undefined) delete input.params.gapCrossingSource;
    for (const r of [checkConnectivitySync(input), spineNetwork(input), projectSpine(input, {years: [2026]}), climateRoutes(input)]) {
      assert.equal(r.status, 'incomplete');
      assert.match(r.reasons.join(' '), reason);
    }
  }
});

test('B2 B6 plowing a stepping stone breaks the chain and is named; a permitted burn over the same patches passes', () => {
  const plow = checkConnectivitySync(plan({treatments: ['plow-3']}));
  assert.equal(plow.status, 'fail');
  assert.deepEqual(keys(plow.lostLinks), ['marsh-e|remnant-n', 'preserve-s|remnant-n', 'remnant-n|remnant-w']);
  assert.ok(plow.lostLinks.every(l => l.causes.join() === 'plow-3'));
  const burn = checkConnectivitySync(plan({treatments: ['burn-2']}));
  assert.equal(burn.status, 'pass', burn.reasons.join(' '));
  assert.deepEqual(keys(burn.linkedAfter), ALL);
  assert.match(burn.warnings.join(' '), /Treatment burn-2 overlaps retained habitat or a core as a permitted light treatment/);
});

test('B6 fire, grazing, mowing and brush management are permitted upkeep only with a recorded permission and reason', () => {
  for (const t of ['prescribed-burn', 'prescribed-grazing', 'late-season-mowing', 'brush-management']) assert.ok(LIGHT_INTENSITIES.includes(t));
  const pkgIntensities = pkgSchema.properties.layers.properties.treatments.items.properties.properties.then.properties.intensity.enum;
  assert.deepEqual(pkgIntensities, LIGHT_INTENSITIES, 'the package schema lists the same treatments');
  const noReason = plan({treatments: ['burn-2']});
  delete noReason.treatments[0].properties.reason;
  const r = checkConnectivitySync(noReason);
  assert.equal(r.status, 'fail');
  assert.match(r.warnings.join(' '), /Treatment burn-2 is marked corridor_permitted but needs an intensity from .* and a recorded reason/);
  const unmarked = plan({treatments: ['burn-2']});
  delete unmarked.treatments[0].properties.corridor_permitted;
  assert.equal(checkConnectivitySync(unmarked).status, 'fail', 'a burn without a recorded permission removes habitat like any other unit');
});

test('B3 the network with stepping stones: every pair linked; the creek, the right-of-way and the patch chain are single points of failure', () => {
  assert.deepEqual(keys(net.coreLinks), ALL);
  assert.equal(net.summary.robustPairs, 0, 'a tree of corridors and one chain of patches: nothing has a second route');
  assert.equal(net.summary.disturbanceWidthM, 120);
  assert.ok(net.singlePointsOfFailure.every(z => z.verified), 'each zone is confirmed at nominal width');
  const chain = net.singlePointsOfFailure.find(z => z.nearPatches.length);
  assert.ok(chain, 'a zone on the patch chain names its patches');
  assert.deepEqual(keys(chain.separates), ['marsh-e|remnant-n', 'preserve-s|remnant-n', 'remnant-n|remnant-w']);
  assert.ok(net.singlePointsOfFailure.some(z => z.nearLines.includes('rail')));
});

test('B3 the default disturbance is wide enough to cut the widest corridor and to open a gap wider than the crossing distance', () => {
  const input = prairie();
  delete input.params.disturbanceWidthM;
  input.params.gapCrossingM = 150;
  // Radius sqrt((r + gap / 2)^2 + ((widest - minWidth) / 2)^2) - r + 1 m, or the widest corridor plus 2 m if larger.
  assert.equal(spineNetwork(input, {cuts: false}).summary.disturbanceWidthM, 156);
  input.params.gapCrossingM = 50;
  assert.equal(spineNetwork(input, {cuts: false}).summary.disturbanceWidthM, 102);
  const noLines = chains([200]);
  delete noLines.params.disturbanceWidthM;
  const r = spineNetwork(noLines);
  assert.equal(r.summary.robustPairs, null);
  assert.match(r.warnings.join(' '), /No spine line is drawn, so the disturbance width has no default/);
});

test('B3 review: the default disturbance opens a gap wider than the crossing distance across the widest corridor', () => {
  const input = {
    streams: [{type: 'Feature', geometry: {type: 'LineString', coordinates: [pt(0, 300), pt(1000, 300)]}, properties: {dfm_id: 's', stream_order: 1}}],
    coreAreas: [core('core-w', -300, 0, 0, 600), core('core-e', 1000, 0, 1300, 600)],
    params: {minWidthM: 60, minWidthSource: 'Test value', spineWidthByOrderM: {1: 100}, spineWidthSource: 'Test value', gapCrossingM: 100, gapCrossingSource: 'Test value'},
  };
  const r = spineNetwork(input);
  assert.equal(r.summary.disturbanceWidthM, 107);
  assert.deepEqual(r.coreLinks.map(x => x.robust), [false]);
  assert.ok(r.singlePointsOfFailure.every(z => z.verified));
  assert.deepEqual([...oracle(input)(pt(500, 300), 107 / 2)], [], 'a disturbance of the default width on the corridor separates the cores');
});

test('B3 review: at 45° N over a tall extent, the raster never joins farther than the exact rule', () => {
  // Cores 300.3 m apart with a 300 m crossing distance: no stepping-stone join; one 62 m link joins them.
  const M = (2 * Math.PI * 6371008.8) / 360, at = (x, y) => [x / (M * Math.cos(Math.PI / 4)), 45 + y / M];
  const box = (dfmId, x0, y0, x1, y1, extra = {}) => ({type: 'Feature', geometry: {type: 'Polygon', coordinates: [[at(x0, y0), at(x1, y0), at(x1, y1), at(x0, y1), at(x0, y0)]]}, properties: {dfm_id: dfmId, ...extra}});
  const W = 300.3;
  const input = {
    coreAreas: [box('core-w', -300, 0, 0, 600, {core_class: 'reserve'}), box('core-e', W, 0, W + 300, 600, {core_class: 'reserve'})],
    connectors: [{type: 'Feature', geometry: {type: 'LineString', coordinates: [at(-1, 535), at(W + 1, 535)]}, properties: {dfm_id: 'link', kind: 'planned', width_m: 62, width_source: 'Test value'}}],
    retained: [box('far-north', 0, 0.49 * M, 10, 0.49 * M + 10)], // stretches the extent 54 km north
    params: {minWidthM: 60, minWidthSource: 'Test value', gapCrossingM: 300, gapCrossingSource: 'Test value', disturbanceWidthM: W + 40},
  };
  const r = spineNetwork(input);
  assert.deepEqual(r.coreLinks.map(x => x.robust), [false]);
  assert.deepEqual([...oracle(input)(at(W / 2, 535), (W + 40) / 2)], [], 'one disturbance on the link separates the cores');
});

test('B2 review: stepping stones never cross a road; a recorded crossing carries the link', () => {
  for (const crossing of [null, 'verified']) {
    const input = woodlot({crossing});
    Object.assign(input.params, {gapCrossingM: 10, gapCrossingSource: 'Test value'}); // the road is 6 m wide
    const r = checkConnectivitySync(input);
    assert.deepEqual(keys(r.linkedBefore), crossing ? ['core-A|core-B'] : [], crossing ? 'linked over the recorded crossing' : 'a 6 m road is not a 10 m gap to cross');
  }
});

test('B2 review: a gap is never measured through a road, around a distant road end or crossing, or through a seam in the road', () => {
  // Two cores 80 m apart (gap 100 m allowed) with an 8 m road between them.
  const params = {minWidthM: 60, minWidthSource: 'Test value', gapCrossingM: 100, gapCrossingSource: 'Test value'};
  const road = (y0, y1) => ({type: 'Feature', geometry: {type: 'LineString', coordinates: [pt(0, y0), pt(0, y1)]}, properties: {dfm_id: 'road', width_m: 8}});
  const linked = input => keys(checkConnectivitySync({...input, params}).linkedBefore);
  const coreE = core('core-e', 40, -600, 300, -400);
  // The road ends 500 m north of core-e: the way around its end is about 600 m.
  assert.deepEqual(linked({coreAreas: [core('core-w', -300, -600, -40, 200), coreE], retained: [core('tiny', -290, -590, -280, -580)], roads: [road(-1000, 100)]}), []);
  // A crossing 400 m north carries a corridor across the road, not the gap down here.
  const strip = {type: 'Feature', geometry: rect(-300, -35, 300, 35), properties: {dfm_id: 'corridor'}};
  const crossing = {type: 'Feature', geometry: {type: 'Point', coordinates: pt(0, 0)}, properties: {dfm_id: 'x', passage: 'verified'}};
  assert.deepEqual(linked({coreAreas: [core('core-w', -300, -600, -40, 100), coreE], retained: [strip], roads: [road(-1000, 1000)], crossings: [crossing]}), []);
  // A road drawn as two surfaces that touch at a corner, or with a 1 cm or 0.5 m seam, has no opening.
  const cores = [core('core-w', -300, -300, -40, 300), core('core-e', 52, -300, 300, 300)], tiny = core('tiny', -290, -290, -280, -280);
  const surface = (id, x0, y0, x1, y1) => ({type: 'Feature', geometry: rect(x0, y0, x1, y1), properties: {dfm_id: id}});
  for (const roads of [
    [surface('r1', -4, -1000, 4, 0), surface('r2', 4, 0, 12, 1000)],
    [surface('r1', -4, -1000, 4, 0), surface('r2', -4, 0.01, 4, 1000)],
    [surface('r1', -4, -1000, 4, 0), surface('r2', -4, 0.5, 4, 1000)],
  ]) assert.deepEqual(linked({coreAreas: cores, retained: [tiny], roads}), []);
  // Control: with no road at all the same cores are linked by the gap.
  assert.deepEqual(linked({coreAreas: cores, retained: [tiny], roads: []}), ['core-e|core-w']);
});

test('B2 review: around a road end a longer crossing distance still never removes a link', () => {
  // Two cores whose eroded corners sit 35 m either side of a dead-end road, 89-93 m below its end;
  // the gap bends around the end. Crossing distances step across 180 m, where the reach (r + g / 2) passes 120 m.
  const r = 30, gaps = [170, 179.95, 180.05, 190, 200];
  for (const ye of [89, 91, 93]) {
    const cores = [core('core-p', -35 - r - 120, -ye - r - 250, -35 + r, -ye + r), core('core-q', 35 - r, -ye - r - 250, 35 + r + 120, -ye + r)];
    const road = {type: 'Feature', geometry: {type: 'LineString', coordinates: [pt(0, -2000), pt(0, 0)]}, properties: {dfm_id: 'road', width_m: 8}};
    const linkedAt = g => checkConnectivitySync({coreAreas: cores, retained: [core('tiny', -35 - r - 110, -ye - 240, -35 - r - 100, -ye - 230)], roads: [road],
      params: {minWidthM: 2 * r, minWidthSource: 'Test value', gapCrossingM: g, gapCrossingSource: 'Test value'}}).linkedBefore.length > 0;
    const seen = gaps.map(linkedAt);
    for (let i = 1; i < gaps.length; i++) assert.ok(!seen[i - 1] || seen[i], `${ye} m below the end: linked at ${gaps[i - 1]} m but not at ${gaps[i]} m`);
    assert.ok(seen.at(-1), `${ye} m below the end: linked once the gap allows the way around`);
  }
});

test('B7 review: exits lie within the landscape, and an exit that links to no core warns', () => {
  const far = plan();
  far.exits[0] = {...far.exits[0], geometry: {type: 'Polygon', coordinates: [[pt(950, 2330 + 1.1e6), pt(1050, 2330 + 1.1e6), pt(1050, 2400 + 1.1e6), pt(950, 2400 + 1.1e6), pt(950, 2330 + 1.1e6)]]}};
  assert.match(climateRoutes(far).reasons.join(' '), /With its exits the landscape spans more than 0.5°/);
  const off = plan();
  off.exits[0] = {...off.exits[0], geometry: rect(950, 2600, 1050, 2700)}; // 200 m past the end of the spine
  const r = climateRoutes(off);
  assert.equal(r.status, 'ok');
  assert.match(r.warnings.join(' '), /Exit exit-n links to no core after the plan/);
});

test('B5 review: a remnant that also records a stand age warns that remnant status applies', () => {
  const input = plan();
  input.coreAreas[0].properties.stand_age = 20;
  assert.match(projectSpine(input, {years: [2026]}).warnings.join(' '), /Feature remnant-w is marked remnant and also records stand_age 20; remnant status applies/);
});

// Patch-only layouts between two cores: rows of 110 m tall patches with 60-80 m gaps. Square
// patches can be removed by one disturbance; 300 m long ones cannot, so only a widened gap breaks
// their chain.
const SQUARE = [60, 250, 440, 630, 820, 1010].map(x => [x, x + 110]);
const LONG = [[60, 360], [440, 740], [820, 1120]];
const UNBROKEN = [[0, 1200]]; // one continuous strip: any cut through it opens a 120 m gap, wider than the 100 m crossing
const core = (dfmId, x0, y0, x1, y1) => ({type: 'Feature', geometry: rect(x0, y0, x1, y1), properties: {dfm_id: dfmId, core_class: 'reserve'}});
function chains(rows, xs = SQUARE) {
  return {
    coreAreas: [core('core-w', -300, 0, 0, 1400), core('core-e', 1200, 0, 1500, 1400)],
    retained: rows.flatMap((y, k) => xs.map(([x0, x1], i) => ({type: 'Feature', geometry: rect(x0, y, x1, y + 110), properties: {dfm_id: `r${k + 1}-${i + 1}`}}))),
    params: {minWidthM: 60, minWidthSource: 'Test value', gapCrossingM: 100, gapCrossingSource: 'Test value', disturbanceWidthM: 120},
  };
}
/** Independent oracle: the check's own rule, stepping stones included, on the habitat less one disk outside the cores. */
function oracle(input) {
  const cores = input.coreAreas.map(f => ({id: f.properties.dfm_id, feature: f}));
  const derived = (input.streams || input.connectors) ? deriveSpine(input).features : [];
  const {habitat, barrier} = effectiveHabitat({...input, retained: [...(input.retained ?? []), ...derived], treatments: []}, [], []);
  const coreUnion = union(input.coreAreas), gap = input.params.gapCrossingM ?? 0;
  return (center, rhoM) => {
    const cut = difference(diskPolygon(center, rhoM, {circumscribed: true}), coreUnion);
    const h = cut ? difference(habitat, cut) : habitat;
    return new Set(linkedPairs(h, cores, input.params.minWidthM, {gapM: gap, barrier}).pairs);
  };
}

test('B3 soundness with stepping stones: no sampled disturbance at nominal width separates a pair reported robust', () => {
  // Two chains 690 m apart are robust; one chain is not, so a search that missed a weakness would show.
  // A continuous strip breaks only where a cut opens a gap just wider than the crossing distance, so a
  // raster that joined farther than the exact rule would claim it robust.
  // A 104 m disturbance opens a gap only 4 m wider than the crossing distance: a tight test of the join rule.
  const cases = [[[200, 1000], SQUARE, ['core-e|core-w']], [[200], SQUARE, []], [[200], LONG, []], [[200], UNBROKEN, []], [[200], UNBROKEN, [], 104]];
  for (const [rows, xs, expected, width] of cases) {
    const input = chains(rows, xs);
    if (width) input.params.disturbanceWidthM = width;
    const r = spineNetwork(input);
    assert.equal(r.status, 'ok', r.reasons.join(' '));
    valid('network', r);
    const robust = r.coreLinks.filter(x => x.robust).map(x => `${x.a}|${x.b}`);
    assert.deepEqual(robust, expected);
    const after = oracle(input), rho = r.summary.disturbanceWidthM / 2;
    for (const y0 of rows) {
      const y = y0 + 55; // patch centers, the gaps between patches, and next to the cores
      const xsAt = xs === UNBROKEN ? [100, 300, 500, 700, 900, 1100] : [30, ...xs.flatMap(([x0, x1]) => [(x0 + x1) / 2, x1 + 40]), 1160];
      for (const x of xsAt) for (const dy of [-20, 0, 20]) {
        const linked = after(pt(x, y + dy), rho);
        for (const k of robust) assert.ok(linked.has(k), `a disturbance at (${x}, ${Math.round(y + dy)}) separates ${k}, reported robust`);
      }
    }
    if (!expected.length) {
      assert.ok(r.singlePointsOfFailure.length > 0);
      // At 120 m each zone is confirmed at nominal width; at 104 m only near-central cuts break the strip.
      if (!width) assert.ok(r.singlePointsOfFailure.every(z => z.verified && z.nearPatches.length > 0));
    }
  }
});

test('B4 flat-land links: swales, moraines, shorelines, rights-of-way, field margins, hedgerows and planned links', () => {
  const d = deriveSpine(prairie());
  const by = Object.fromEntries(d.features.map(f => [f.properties.source_id, f.properties]));
  assert.equal(by.creek.origin, 'stream'); assert.equal(by.creek.width_m, 100);
  assert.equal(by.rail.origin, 'right-of-way'); assert.equal(by.rail.width_m, 75);
  assert.equal(by.rail.width_source, 'Fixture: 75 m railroad right-of-way', 'a link carries its own width and source');
  assert.equal(by.moraine.origin, 'moraine'); assert.equal(by.moraine.width_source, 'Fixture link width');
  const kinds = pkgSchema.properties.layers.properties.connectors.items.properties.properties.properties.kind.enum;
  assert.deepEqual(kinds, SPINE_LINK_KINDS, 'the package schema lists the same kinds');
  assert.deepEqual(results.$defs.derivation.properties.features.items.properties.properties.properties.origin.enum, ['stream', ...SPINE_LINK_KINDS]);
  for (const kind of ['swale', 'shoreline', 'field-margin', 'hedgerow', 'planned']) {
    const input = prairie();
    input.connectors[1].properties.kind = kind;
    assert.equal(deriveSpine(input).features.find(f => f.properties.source_id === 'moraine').properties.origin, kind);
  }
});

test('B4 a link narrower than the minimum, or with a width but no source, is incomplete', () => {
  const narrow = prairie();
  narrow.connectors[0].properties.width_m = 40;
  assert.match(deriveSpine(narrow).reasons.join(' '), /Connector rail is 40 m wide, narrower than the 60 m minimum width/);
  const noSource = prairie();
  delete noSource.connectors[0].properties.width_source;
  assert.match(deriveSpine(noSource).reasons.join(' '), /Connector rail has width_m; record width_source too/);
  const pinch = prairie();
  pinch.connectors[0].properties.width_m = 64;
  assert.match(deriveSpine(pinch).warnings.join(' '), /Connector rail is 64 m wide, within the pinch margin/);
  const unknown = prairie();
  unknown.connectors[0].properties.kind = 'fence';
  assert.match(deriveSpine(unknown).reasons.join(' '), /Connector rail needs a kind: ridge, valley, saddle, swale/);
});

test('B5 remnants count at old-growth age from the start; restored habitat waits for its age', () => {
  const p = projectSpine(plan(), {years: [2026, 2095, 2109, 2110]});
  const at = Object.fromEntries(p.milestones.map(m => [m.year, keys(m.oldGrowthAgeLinks)]));
  assert.deepEqual(at[2026], [], 'the creek (31 years) and the patches (16) are young');
  assert.deepEqual(at[2109], [], 'the creek is old by 2095, the patches only from 2110');
  assert.deepEqual(at[2110], ['remnant-n|remnant-w']);
  // With every farm committed and the creek a remnant, the south links run through remnants alone today.
  const all = plan();
  for (const f of all.parcels) {
    Object.assign(f.properties, {consent: 'covered', consent_ref: `fixture-${f.properties.dfm_id}`, consent_year: 2024});
    delete f.properties.planned_year;
  }
  for (const f of all.retained) if (f.properties.source_id === 'creek') Object.assign(f.properties, {remnant: true, remnant_source: REMNANT_SOURCE});
  const now = projectSpine(all, {years: [2026]}).milestones[0];
  assert.deepEqual(keys(now.oldGrowthAgeLinks), SOUTH);
  assert.equal(now.unknownAgeM2, 0, 'remnant status is a recorded age, not an unknown one');
});

test('B5 remnant status needs a source and must be true or false', () => {
  const noSource = plan();
  delete noSource.coreAreas[0].properties.remnant_source;
  assert.match(projectSpine(noSource, {years: [2026]}).reasons.join(' '), /Feature remnant-w is marked remnant; record remnant_source/);
  const notBool = plan();
  notBool.retained[0].properties.remnant = 'yes';
  assert.match(projectSpine(notBool, {years: [2026]}).reasons.join(' '), /remnant must be true or false/);
});

test('B7 in flat country the cooler ground lies beyond the landscape: without an exit only the coolest core is unflagged', () => {
  const c = byId(climateNoExit);
  assert.equal(c['remnant-n'].status, 'coolest');
  for (const id of ['marsh-e', 'preserve-s', 'remnant-w']) {
    assert.equal(c[id].status, 'short');
    assert.ok(c[id].coolingC < 1, `${id} is less than 1 °C warmer than the coolest core`);
  }
  assert.deepEqual(climateNoExit.flagged, ['marsh-e', 'preserve-s', 'remnant-w']);
  assert.deepEqual(climateNoExit.exits, []);
});

test('B7 an exit carries routes into the next landscape', () => {
  const c = byId(climate);
  for (const id of ['marsh-e', 'preserve-s', 'remnant-n', 'remnant-w']) {
    assert.equal(c[id].status, 'route');
    assert.deepEqual(c[id].coolest, {id: 'exit-n', tempC: 7.2, via: [id, 'exit-n'], exit: true, toward: 'The next landscape north (fictional)'});
  }
  assert.equal(c['remnant-n'].coolingC, 2.2);
  assert.deepEqual(climate.flagged, []);
  assert.deepEqual(climate.exits, [{id: 'exit-n', tempC: 7.2, toward: 'The next landscape north (fictional)', linkedCores: ['marsh-e', 'preserve-s', 'remnant-n', 'remnant-w']}]);
  assert.match(climate.limitations.join(' '), /An exit stands for the spine continuing into the next landscape/);
});

test('B7 a core cut off from the exit is flagged: a cooler core and an exit exist but neither is reachable', () => {
  const c = byId(climateNoGap);
  assert.equal(c['remnant-n'].status, 'route', 'the moraine link reaches the exit');
  assert.equal(c['remnant-w'].status, 'none');
  assert.equal(c['marsh-e'].status, 'short');
  assert.deepEqual(climateNoGap.flagged, ['marsh-e', 'preserve-s', 'remnant-w']);
  const plowed = byId(climateRoutes(plan({treatments: ['plow-3']})));
  assert.equal(plowed['remnant-w'].status, 'none', 'plowing a stepping stone cuts the south off from the exit');
});

test('B7 exits are validated: a polygon, an ID unlike any core, and a temperature', () => {
  const cases = [
    [e => { delete e.properties.temp_c; }, /Exit exit-n needs temp_c/],
    [e => { e.properties.dfm_id = 'remnant-n'; }, /Exit remnant-n has the same ID as a core area/],
    [e => { e.geometry = {type: 'Point', coordinates: pt(1000, 2400)}; }, /exits\[0\] must be a polygon/],
  ];
  for (const [change, reason] of cases) {
    const input = plan();
    change(input.exits[0]);
    const r = climateRoutes(input);
    assert.equal(r.status, 'incomplete');
    assert.match(r.reasons.join(' '), reason);
  }
  assert.match(climateRoutes({...plan(), exits: {}}).reasons.join(' '), /exits must be an array/);
});

test('Landscape Package: exits, remnants, link widths and stepping stones round-trip and validate against the schema', () => {
  const input = plan();
  const pkg = toLandscapePackage(input, {created: '2026-10-04T00:00:00Z'});
  const validPkg = new Ajv2020({allErrors: true, strict: false}).compile(pkgSchema);
  assert.ok(validPkg(pkg), JSON.stringify(validPkg.errors));
  assert.equal(pkg.layers.exits.length, 1);
  const back = fromLandscapePackage(pkg);
  assert.equal(canonicalJSON(back.exits), canonicalJSON(input.exits));
  assert.equal(back.params.gapCrossingM, 100);
  const bad = structuredClone(pkg);
  bad.layers.coreAreas[0].properties.remnant_source = undefined;
  delete bad.layers.coreAreas[0].properties.remnant_source;
  assert.ok(!validPkg(bad), 'a remnant without a source does not validate');
  const noExit = toLandscapePackage(plan({exits: false}), {created: '2026-10-04T00:00:00Z'});
  assert.ok(!('exits' in noExit.layers), 'exits are written only when present');
});

test('B1 B2 results are deterministic', () => {
  assert.equal(canonicalJSON(checkAt(100)), canonicalJSON(checks[100]));
  assert.equal(canonicalJSON(climateRoutes(plan())), canonicalJSON(climate));
});
