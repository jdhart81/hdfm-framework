// Each test names the DFM Build Spec invariant it verifies.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import Ajv2020 from 'ajv/dist/2020.js';
import {checkConnectivity, toLandscapePackage, fromLandscapePackage, canonicalJSON} from '../src/index.mjs';
import {woodlot} from '../fixtures/woodlot.mjs';

const linked = (r, a = 'core-A', b = 'core-B') => r.linkedAfter.some(p => p.a === a && p.b === b);
const schema = async name => JSON.parse(await readFile(new URL(`../../dfm-schema/${name}.schema.json`, import.meta.url), 'utf8'));

test('baseline woodlot: cores linked across a verified road crossing, check passes', async () => {
  const r = await checkConnectivity(woodlot());
  assert.equal(r.status, 'pass', r.reasons.join(' '));
  assert.ok(linked(r));
  assert.deepEqual(r.lostLinks, []);
});

test('I1 a harvest unit cutting the only corridor fails and is named as the cause', async () => {
  const r = await checkConnectivity(woodlot({treatments: ['harvest-3', 'harvest-1']}));
  assert.equal(r.status, 'fail');
  assert.equal(r.lostLinks.length, 1);
  assert.deepEqual(r.lostLinks[0].causes, ['harvest-3'], 'harvest-1 is away from the corridor and is not a cause');
  assert.ok(!linked(r));
  assert.ok(r.geometry.features.some(f => f.properties.dfm_layer === 'connectivity-loss'));
});

test('I1 narrowing the only link below the minimum width fails', async () => {
  const r = await checkConnectivity(woodlot({treatments: ['edge-cut']}));
  assert.equal(r.status, 'fail');
  assert.deepEqual(r.lostLinks[0].causes, ['edge-cut']);
});

test('I2 unpermitted overlap fails even when a link survives; a recorded light treatment passes with a warning', async () => {
  const wide = await checkConnectivity(woodlot({corridorWidth: 400, treatments: ['edge-cut']}));
  assert.equal(wide.status, 'fail', 'overlap without permission fails');
  assert.ok(linked(wide), 'the link itself survives in a 400 m corridor');
  assert.match(wide.reasons.join(' '), /edge-cut overlaps retained habitat or a core/);

  const thin = await checkConnectivity(woodlot({treatments: ['thin-3']}));
  assert.equal(thin.status, 'pass', thin.reasons.join(' '));
  assert.ok(linked(thin));
  assert.match(thin.warnings.join(' '), /thin-3 overlaps retained habitat or a core as a permitted light treatment/);
});

test('I2 a permission without a recorded intensity and reason is not honored', async () => {
  const input = woodlot({treatments: ['thin-3']});
  delete input.treatments[0].properties.reason;
  const r = await checkConnectivity(input);
  assert.equal(r.status, 'fail');
  assert.match(r.warnings.join(' '), /needs an intensity from .* and a recorded reason/);
});

test('I3 minimum width by hand calculation: 101 m corridor links at 100 m, 99 m does not', async () => {
  const at101 = await checkConnectivity(woodlot({corridorWidth: 101, road: false}));
  const at99 = await checkConnectivity(woodlot({corridorWidth: 99, road: false}));
  assert.ok(linked(at101), '101 m corridor must carry a 100 m link');
  assert.ok(!linked(at99), '99 m corridor must not carry a 100 m link');
  assert.match(at99.warnings.join(' '), /not linked at 100 m in the current state/);
  assert.deepEqual(at101.pinchedLinks.map(p => p.a + p.b), ['core-Acore-B'], '101 m is within 10% of 100 m: a pinch point');
});

test('I3 the minimum width must carry a recorded source', async () => {
  const input = woodlot();
  input.params.minWidthSource = '';
  const r = await checkConnectivity(input);
  assert.equal(r.status, 'incomplete');
  assert.match(r.reasons.join(' '), /minWidthSource/);
});

test('I4 roads sever corridors unless a crossing is recorded; assumed crossings warn', async () => {
  const none = await checkConnectivity(woodlot({crossing: 'none'}));
  const missing = await checkConnectivity(woodlot({crossing: null}));
  const assumed = await checkConnectivity(woodlot({crossing: 'assumed'}));
  assert.ok(!linked(none));
  assert.ok(!linked(missing));
  assert.ok(linked(assumed));
  assert.match(assumed.warnings.join(' '), /assumed, not field-verified/);
});

test('I5 retained habitat is committed only inside parcels with covering consent', async () => {
  const r = await checkConnectivity(woodlot({road: false}));
  // Corridor 400–1600 m x 120 m; the west parcel (x < 1000) has consent: 600 m x 120 m = 72,000 m².
  assert.ok(Math.abs(r.consent.committedM2 - 72_000) / 72_000 < 0.005, r.consent.committedM2);
  assert.ok(Math.abs(r.consent.proposedM2 - 72_000) / 72_000 < 0.005, r.consent.proposedM2);
  const noParcels = await checkConnectivity(woodlot({road: false, parcels: false}));
  assert.equal(noParcels.consent.committedM2, 0);
});

test('I6 old-growth-verified needs an evidence reference', async () => {
  const unsupported = await checkConnectivity(woodlot({coreClassA: 'old-growth-verified'}));
  assert.equal(unsupported.cores.find(c => c.id === 'core-A').coreClass, 'old-growth-candidate');
  assert.match(unsupported.warnings.join(' '), /without an evidence reference/);
  const supported = await checkConnectivity(woodlot({coreClassA: 'old-growth-verified', evidenceA: 'evidence-42'}));
  assert.equal(supported.cores.find(c => c.id === 'core-A').coreClass, 'old-growth-verified');
});

test('I7 missing inputs return incomplete with reasons, never a pass', async () => {
  for (const mutate of [
    i => { i.coreAreas = i.coreAreas.slice(0, 1); },
    i => { i.retained = []; },
    i => { delete i.params.roadWidthM; },
    i => { i.crossings[0].properties.passage = 'maybe'; },
    i => { i.coreAreas[0].properties.core_class = 'ancient'; },
  ]) {
    const input = woodlot(); mutate(input);
    const r = await checkConnectivity(input);
    assert.equal(r.status, 'incomplete');
    assert.ok(r.reasons.length > 0);
  }
  assert.equal((await checkConnectivity(null)).status, 'incomplete');
});

test('I8 same inputs give the same result and checksum; changed inputs change the checksum', async () => {
  const a = await checkConnectivity(woodlot({treatments: ['harvest-3']}));
  const b = await checkConnectivity(woodlot({treatments: ['harvest-3']}));
  assert.equal(canonicalJSON(a), canonicalJSON(b));
  assert.match(a.inputChecksum, /^sha256:[0-9a-f]{64}$/);
  const c = await checkConnectivity(woodlot({treatments: ['harvest-1']}));
  assert.notEqual(a.inputChecksum, c.inputChecksum);
});

test('I10 areas are square meters by hand calculation', async () => {
  const r = await checkConnectivity(woodlot());
  const core = r.cores.find(c => c.id === 'core-A');
  assert.ok(Math.abs(core.areaM2 - 120_000) / 120_000 < 0.005, core.areaM2); // 300 m x 400 m
});

test('Landscape Package round trip validates against the v1 schema; results validate too', async () => {
  const ajv = new Ajv2020({allErrors: true, strict: false});
  const pkg = toLandscapePackage(woodlot({treatments: ['harvest-3']}), {name: 'Fixture woodlot', created: '2026-10-03T00:00:00Z'});
  const validPkg = ajv.compile(await schema('landscape-package'));
  assert.ok(validPkg(pkg), JSON.stringify(validPkg.errors));
  const result = await checkConnectivity(fromLandscapePackage(pkg));
  assert.equal(result.status, 'fail');
  const validResult = ajv.compile(await schema('connectivity-result'));
  assert.ok(validResult(result), JSON.stringify(validResult.errors));
  assert.throws(() => fromLandscapePackage({...pkg, dfm_package: '2.0'}), /Unsupported/);
});

test('digitizing seams under 1 m between adjacent retained polygons do not sever a link; a 5 m gap does', async () => {
  const {rect} = await import('../fixtures/woodlot.mjs');
  const split = gap => {
    const input = woodlot({road: false});
    input.retained = [
      {type: 'Feature', geometry: rect(400, 440, 1000, 560), properties: {dfm_id: 'west'}},
      {type: 'Feature', geometry: rect(1000 + gap, 440, 1600, 560), properties: {dfm_id: 'east'}},
    ];
    return checkConnectivity(input);
  };
  assert.ok(linked(await split(0.2)), 'a 0.2 m seam is closed');
  assert.ok(!linked(await split(5)), 'a 5 m unretained strip is a real gap');
});

// Regression tests from the independent review (2026-10-03).
const F = (geometry, properties) => ({type: 'Feature', geometry, properties});
const fx = await import('../fixtures/woodlot.mjs');

test('review 1: clearing a core area fails, including a core outside retained habitat', async () => {
  const input = woodlot({road: false});
  input.treatments = [F(fx.rect(1600, 300, 1900, 700), {dfm_id: 'clear-B', intensity: 'clearcut'})];
  const r = await checkConnectivity(input);
  assert.equal(r.status, 'fail');
  assert.ok(!linked(r));
  assert.deepEqual(r.lostLinks[0].causes, ['clear-B']);
  assert.ok(r.cores.find(c => c.id === 'core-B').remainingM2 < 1);
});

test('review 2: a clearcut cannot be permitted as a light treatment', async () => {
  const input = woodlot();
  input.treatments = [F(fx.rect(700, 300, 800, 800), {dfm_id: 'x', intensity: 'clearcut', corridor_permitted: true, reason: 'because'})];
  const r = await checkConnectivity(input);
  assert.equal(r.status, 'fail');
  assert.ok(!linked(r));
});

test('review 3: wrong geometry types are incomplete, not silently ignored', async () => {
  const input = woodlot({road: false});
  input.water = [F({type: 'LineString', coordinates: [fx.pt(900, 0), fx.pt(900, 1000)]}, {dfm_id: 'stream'})];
  const r = await checkConnectivity(input);
  assert.equal(r.status, 'incomplete');
  assert.match(r.reasons.join(' '), /water\[0\] must be a polygon/);
});

test('review 4: a crossing on one road does not carry a link across a nearby second road', async () => {
  const input = woodlot();
  input.roads.push(F({type: 'LineString', coordinates: [fx.pt(1015, 0), fx.pt(1015, 1000)]}, {dfm_id: 'road-2'}));
  const r = await checkConnectivity(input);
  assert.ok(!linked(r), 'road-2 has no crossing record');
});

test('review 5: a plan entirely outside habitat passes on a many-core landscape', async () => {
  const cores = [];
  for (let k = 0; k < 5; k++) {
    cores.push(F(fx.rect(0, k * 400, 300, k * 400 + 300), {dfm_id: `W${k}`, core_class: 'reserve'}));
    cores.push(F(fx.rect(3000, k * 400, 3300, k * 400 + 300), {dfm_id: `E${k}`, core_class: 'reserve'}));
  }
  const input = {coreAreas: cores, retained: [F(fx.rect(300, 0, 600, 1900), {}), F(fx.rect(2700, 0, 3000, 1900), {}), F(fx.rect(600, 900, 2700, 1020), {})],
    treatments: [F(fx.rect(1500, 1200, 1600, 1300), {dfm_id: 'outside'})], params: {minWidthM: 100, minWidthSource: 'test'}};
  const r = await checkConnectivity(input);
  assert.equal(r.status, 'pass', r.reasons.join(' '));
  input.treatments = [F(fx.rect(1500, 800, 1600, 1100), {dfm_id: 'cut'})];
  const cut = await checkConnectivity(input);
  assert.equal(cut.status, 'fail', cut.reasons.join(' '));
});

test('review 6: a verified crossing works over a wide road polygon without roadWidthM', async () => {
  const input = woodlot();
  delete input.params.roadWidthM;
  input.roads = [F(fx.rect(985, 0, 1015, 1000), {dfm_id: 'wide-road'})];
  assert.ok(linked(await checkConnectivity(input)), '30 m road with a verified crossing');
  input.crossings = [];
  assert.ok(!linked(await checkConnectivity(input)), 'same road without a crossing');
});

test('review 7: a crossing links habitat drawn up to both road edges', async () => {
  const input = woodlot();
  input.retained = [F(fx.rect(400, 440, 997, 560), {}), F(fx.rect(1003, 440, 1600, 560), {})];
  assert.ok(linked(await checkConnectivity(input)));
  input.crossings = [];
  assert.ok(!linked(await checkConnectivity(input)));
});

test('review 8: core IDs containing "|" are rejected', async () => {
  const input = woodlot();
  input.coreAreas[0].properties.dfm_id = 'a|x';
  assert.equal((await checkConnectivity(input)).status, 'incomplete');
});

test('review: projected coordinates and oversized extents are rejected', async () => {
  const input = woodlot();
  input.retained[0].geometry.coordinates[0] = input.retained[0].geometry.coordinates[0].map(([x, y]) => [x * 111195, y * 111195]);
  assert.match((await checkConnectivity(input)).reasons.join(' '), /longitude\/latitude/);
});

// Second review round (2026-10-03).
test('review 4b: a crossing applies only to the branch it sits on within one road feature', async () => {
  const hairpin = woodlot();
  hairpin.roads = [F({type: 'LineString', coordinates: [fx.pt(1000, 0), fx.pt(1000, 1000), fx.pt(1015, 1000), fx.pt(1015, 0)]}, {dfm_id: 'hairpin'})];
  assert.ok(!linked(await checkConnectivity(hairpin)), 'hairpin return branch has no crossing');
  const multi = woodlot();
  multi.roads = [F({type: 'MultiLineString', coordinates: [[fx.pt(1000, 0), fx.pt(1000, 1000)], [fx.pt(1015, 0), fx.pt(1015, 1000)]]}, {dfm_id: 'network'})];
  assert.ok(!linked(await checkConnectivity(multi)));
  multi.crossings.push(F({type: 'Point', coordinates: fx.pt(1015, 500)}, {dfm_id: 'crossing-2', passage: 'verified'}));
  assert.ok(linked(await checkConnectivity(multi)), 'both branches crossed');
});

test('review new-2: a bridge does not let offset corridors link across a wide road; one-sided habitat is not bridged', async () => {
  const input = woodlot();
  delete input.params.roadWidthM;
  input.roads = [F(fx.rect(980, 0, 1020, 1000), {dfm_id: 'wide'})];
  input.retained = [F(fx.rect(400, 440, 980, 560), {}), F(fx.rect(1020, 450, 1600, 570), {})];
  assert.ok(linked(await checkConnectivity(input)), 'a 10 m offset still leaves a 110 m straight-across strip');
  input.retained = [F(fx.rect(400, 440, 980, 560), {}), F(fx.rect(1020, 500, 1600, 620), {})];
  assert.ok(!linked(await checkConnectivity(input)), 'a 60 m offset leaves only 60 m straight across');
  input.retained = [F(fx.rect(400, 440, 980, 560), {})];
  const oneSided = await checkConnectivity(input);
  assert.ok(!linked(oneSided));
  assert.match(oneSided.warnings.join(' '), /both sides of the road/);
});

test('review new-3: a core counts only while at least half of it remains', async () => {
  const input = woodlot({road: false});
  input.treatments = [F(fx.rect(1600.5, 300, 1900, 700), {dfm_id: 'almost-all-B'})];
  const r = await checkConnectivity(input);
  assert.equal(r.status, 'fail');
  assert.equal(r.lostLinks.length, 1);
});

test('review new-1: road width_m is range-checked; null pinchFraction uses the default', async () => {
  const input = woodlot();
  input.roads[0].properties.width_m = 0.6;
  assert.equal((await checkConnectivity(input)).status, 'incomplete');
  const ok = woodlot();
  ok.params.pinchFraction = null;
  assert.equal((await checkConnectivity(ok)).status, 'pass');
});

test('sync SHA-256 matches the platform implementation; sync and async checks agree', async () => {
  const {sha256Hex, checkConnectivitySync} = await import('../src/index.mjs');
  const {createHash} = await import('node:crypto');
  for (const text of ['', 'abc', 'a'.repeat(55), 'a'.repeat(56), 'a'.repeat(64), 'é∂ƒ — woodlot', 'x'.repeat(100000)])
    assert.equal(sha256Hex(text), createHash('sha256').update(text).digest('hex'));
  const input = woodlot({treatments: ['harvest-3']});
  assert.equal(canonicalJSON(checkConnectivitySync(input)), canonicalJSON(await checkConnectivity(input)));
});
