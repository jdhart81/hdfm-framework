// Time, climate and build-out. Each test names the invariant in src/outlook.mjs it checks.
// Tests marked "review" reproduce findings from the independent review of 2026-10-04.
// The watershed results are computed once and shared: each takes a few seconds.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import Ajv2020 from 'ajv/dist/2020.js';
import {deriveSpine, projectSpine, climateRoutes, buildOutFrontier, checkConnectivitySync, canonicalJSON} from '../src/index.mjs';
import {watershed, withAges, AS_OF} from '../fixtures/watershed.mjs';
import {rect} from '../fixtures/woodlot.mjs';

const schema = JSON.parse(await readFile(new URL('../../dfm-schema/spine-results.schema.json', import.meta.url), 'utf8'));
const ajv = new Ajv2020({allErrors: true, strict: false});
ajv.addSchema(schema);
const valid = (def, r) => { const v = ajv.getSchema(`${schema.$id}#/$defs/${def}`); assert.ok(v(r), JSON.stringify(v.errors)); };
const keys = list => (list ?? []).map(p => `${p.a}|${p.b}`).sort();
const ALL = ['core-e|core-n', 'core-e|core-s', 'core-e|core-w', 'core-n|core-s', 'core-n|core-w', 'core-s|core-w'];
const YEARS = [2026, 2036, 2051, 2101, 2126];

const spineFeatures = withAges(deriveSpine(watershed()).features);
const ws = opts => ({...watershed(opts), retained: structuredClone(spineFeatures)});
const good = ws();
const bad = ws({treatments: ['unit-a', 'cut-main']});
const projection = projectSpine(good, {years: YEARS});
const projectionBad = projectSpine(bad, {years: [2026, 2036, 2126]});
const checkGood = checkConnectivitySync(good), checkBad = checkConnectivitySync(bad);
const climate = climateRoutes(good, checkGood), climateBad = climateRoutes(bad, checkBad);
const frontier = buildOutFrontier(good);
const byYear = r => Object.fromEntries(r.milestones.map(x => [x.year, x]));

test('PJ1 the spine is committed woodlot by woodlot: center, north and south today; east and west by 2036; all six links from then on', () => {
  assert.equal(projection.status, 'ok', projection.reasons.join(' '));
  valid('projection', projection);
  const m = byYear(projection);
  assert.deepEqual(m[2026].committedParcels, ['woodlot-1', 'woodlot-2', 'woodlot-3']);
  assert.deepEqual(keys(m[2026].committedLinks), ['core-n|core-s']);
  assert.deepEqual(m[2036].committedParcels, ['woodlot-1', 'woodlot-2', 'woodlot-3', 'woodlot-4', 'woodlot-5']);
  assert.deepEqual(keys(m[2036].committedLinks), ALL);
  assert.ok(!m[2126].committedParcels.includes('woodlot-6') && !m[2126].committedParcels.includes('woodlot-9'), 'parcels without consent or a planned year never join');
  assert.ok(m[2036].committedM2 > m[2026].committedM2);
});

test('PJ1 review: a parcel without consent cannot plan to join in or before the as-of year', () => {
  const input = ws(); input.parcels.find(p => p.properties.dfm_id === 'woodlot-4').properties.planned_year = 2020;
  const r = projectSpine(input, {years: [2026]});
  assert.equal(r.status, 'incomplete');
  assert.match(r.reasons.join(' '), /woodlot-4 plans to join in 2020, not after params.ageAsOfYear 2026; a parcel that has joined needs covering consent/);
});

test('PJ3 the spine reaches old-growth age as it ages: none linked through it until the century, all six by 2126', () => {
  const m = byYear(projection);
  for (const y of [2026, 2036, 2051]) { assert.deepEqual(m[y].oldGrowthAgeLinks, []); assert.equal(m[y].oldGrowthAgeM2, 0, `${y}: no stand is 150 yet`); }
  assert.deepEqual(m[2101].oldGrowthAgeLinks, [], 'in 2101 only the 90-year headwater stands have reached 150');
  assert.ok(m[2101].oldGrowthAgeM2 > 0);
  assert.deepEqual(keys(m[2126].oldGrowthAgeLinks), ALL);
  for (const x of projection.milestones) assert.equal(x.unknownAgeM2, 0);
  assert.match(projection.limitations.join(' '), /Old-growth condition needs field evidence/);
  assert.match(projection.warnings.join(' '), /youngest age applies where they overlap/, 'the derived spine overlaps at confluences');
});

test('PJ3 review: a young tributary corridor overlapping the main stem at a confluence breaks old-age links through that confluence', () => {
  const input = ws();
  for (const f of input.retained) if (f.properties.source_id === 'e2') f.properties.stand_age = 20;
  const [m] = projectSpine(input, {years: [2126]}).milestones;
  assert.ok(!keys(m.oldGrowthAgeLinks).some(k => k.includes('core-s')), 'the young e2 corridor carves the main stem where they meet, above the riparian core');
  assert.ok(keys(m.oldGrowthAgeLinks).includes('core-n|core-w'));
});

test('PJ3 review: where features overlap, the youngest recorded age applies, so a young main stem breaks old-age links across it', () => {
  const young = ws();
  for (const f of young.retained) if (f.properties.source_id === 'main') f.properties.stand_age = 10;
  const m = byYear(projectSpine(young, {years: [2126]}));
  assert.ok(!keys(m[2126].oldGrowthAgeLinks).some(k => k.includes('core-s')), 'nothing reaches the riparian core across the young stem');
  assert.ok(!keys(m[2126].oldGrowthAgeLinks).includes('core-e|core-w'), 'east and west do not link across the young confluences');
});

test('PJ3 retained habitat without a recorded age never counts as old; its area is reported as unknown', () => {
  const input = ws();
  for (const f of input.retained) if (f.properties.source_id === 'main') delete f.properties.stand_age;
  const [m] = projectSpine(input, {years: [2126]}).milestones;
  assert.ok(m.unknownAgeM2 > 0);
  assert.ok(keys(m.oldGrowthAgeLinks).length < ALL.length);
  assert.ok(!keys(m.oldGrowthAgeLinks).some(k => k.includes('core-s')));
});

test('PJ3 review: cores are endpoints whatever their age, but a path never runs through a core without an old-age record', () => {
  // Two touching cores in one consenting woodlot, with only young retained forest elsewhere.
  const sq = (dfmId, x0, y0, x1, y1, extra = {}) => ({type: 'Feature', geometry: rect(x0, y0, x1, y1), properties: {dfm_id: dfmId, ...extra}});
  const input = {
    coreAreas: [sq('a', 0, 0, 300, 300, {core_class: 'reserve'}), sq('b', 300, 0, 600, 300, {core_class: 'reserve'})],
    retained: [sq('young', 0, 600, 600, 800, {stand_age: 20})],
    parcels: [sq('lot', -100, -100, 700, 900, {consent: 'covered', consent_year: 2020})],
    params: {minWidthM: 100, minWidthSource: 'Test', ageAsOfYear: 2026, oldGrowthAgeYears: 150, oldGrowthAgeSource: 'Test'},
  };
  const [m] = projectSpine(input, {years: [2030]}).milestones;
  assert.deepEqual(keys(m.committedLinks), ['a|b'], 'touching cores are linked');
  assert.deepEqual(m.oldGrowthAgeLinks, [], 'but not through old-age forest: neither core has an old-age record');
  input.coreAreas.forEach(c => { c.properties.stand_age = 200; });
  assert.deepEqual(keys(projectSpine(input, {years: [2030]}).milestones[0].oldGrowthAgeLinks), ['a|b'], 'with old-age records they are');
});

test('PJ4 PJ5 links nest (old-growth age ⊆ committed ⊆ after the plan) and never decrease over time', () => {
  for (const r of [projection, projectionBad]) {
    const after = new Set(keys(r.linkedAfter));
    let prev = {c: new Set(), o: new Set()};
    for (const m of r.milestones) {
      const c = new Set(keys(m.committedLinks)), o = new Set(keys(m.oldGrowthAgeLinks));
      for (const k of c) assert.ok(after.has(k), `${m.year}: committed ${k} holds after the plan`);
      for (const k of o) assert.ok(c.has(k), `${m.year}: old-age ${k} is committed`);
      for (const k of prev.c) assert.ok(c.has(k), `${m.year}: committed ${k} is not lost`);
      for (const k of prev.o) assert.ok(o.has(k), `${m.year}: old-age ${k} is not lost`);
      prev = {c, o};
    }
  }
});

test('PJ2 the plan applies at every milestone: a clearcut across the main stem leaves only north-west and east-south', () => {
  assert.equal(checkBad.status, 'fail');
  assert.deepEqual(keys(projectionBad.linkedAfter), ['core-e|core-s', 'core-n|core-w']);
  const m = byYear(projectionBad);
  assert.deepEqual(keys(m[2026].committedLinks), [], 'the cut breaks the only committed link');
  assert.deepEqual(keys(m[2126].committedLinks), ['core-n|core-w'], 'east-south needs woodlot 9, which never joins');
});

test('PJ2 review: reported areas are after the plan, so a clearcut lowers them', () => {
  const g = byYear(projection), b = byYear(projectionBad);
  assert.ok(b[2026].committedM2 < g[2026].committedM2);
  assert.ok(b[2126].oldGrowthAgeM2 < g[2126].oldGrowthAgeM2);
});

test('PJ1 PJ3 I7 projection inputs are validated', () => {
  const cases = [
    [i => { delete i.params.ageAsOfYear; }, /ageAsOfYear must be the year/],
    [i => { i.params.oldGrowthAgeSource = ''; }, /oldGrowthAgeSource must record/],
    [i => { i.retained[0].properties.stand_age = -3; }, /stand_age -3/],
    [i => { i.parcels[0].properties.consent_year = AS_OF + 1; }, /covering consent from 2027, after params.ageAsOfYear 2026; use planned_year/],
    [i => { i.parcels[3].properties.consent_year = 2020; }, /has a consent_year but its consent is not 'covered'/],
    [i => { i.parcels[1].properties.dfm_id = 'woodlot-1'; }, /Parcel IDs must be unique/],
  ];
  for (const [edit, reason] of cases) {
    const input = ws(); edit(input);
    const r = projectSpine(input, {years: [2036]});
    assert.equal(r.status, 'incomplete');
    assert.match(r.reasons.join(' '), reason);
    valid('projection', r);
  }
  assert.match(projectSpine(ws(), {years: [2020]}).reasons.join(' '), /must be 2026 \(params.ageAsOfYear\) or later/);
  assert.match(projectSpine(ws(), {years: [2050, 2040]}).reasons.join(' '), /must increase/);
  assert.match(projectSpine(ws(), {}).reasons.join(' '), /Milestone years are required/);
  assert.equal(projectSpine({coreAreas: 'x'}, {years: [2030]}).status, 'incomplete', 'malformed input never throws');
});

test('PJ3 without an old-growth age threshold only committed links are projected, and the result says so', () => {
  const input = ws(); delete input.params.oldGrowthAgeYears; delete input.params.oldGrowthAgeSource;
  input.params.milestoneYears = [2026];
  const r = projectSpine(input);
  assert.equal(r.status, 'ok');
  assert.equal(r.milestones[0].oldGrowthAgeLinks, null);
  assert.match(r.warnings.join(' '), /oldGrowthAgeYears is not set/);
});

test('CL1 CL3 climate routes after a passing plan: the warm lowland core has a route north; the headwater core is the coolest', () => {
  assert.equal(climate.status, 'ok', climate.reasons.join(' '));
  valid('climate', climate);
  const c = Object.fromEntries(climate.cores.map(x => [x.id, x]));
  assert.equal(c['core-s'].status, 'route');
  assert.deepEqual(c['core-s'].coolest, {id: 'core-n', tempC: 6, via: ['core-s', 'core-n'], exit: false, toward: null});
  assert.equal(c['core-s'].coolingC, 2.6);
  assert.equal(c['core-n'].status, 'coolest');
  assert.equal(c['core-w'].status, 'short', '1.2 °C cooler is less than the 2 °C target');
  assert.equal(c['core-e'].status, 'short');
  assert.deepEqual(climate.flagged, ['core-e', 'core-w']);
  assert.match(climate.limitations.join(' '), /may cross warmer ground/);
});

test('CL1 CL3 after a plan that cuts the main stem, the east core has no route to cooler ground and is flagged', () => {
  const c = Object.fromEntries(climateBad.cores.map(x => [x.id, x]));
  assert.equal(climateBad.checkStatus, 'fail');
  assert.equal(c['core-e'].status, 'none', 'cooler cores exist, but none is reachable');
  assert.equal(c['core-e'].coolest, null);
  assert.equal(c['core-s'].status, 'short');
  assert.deepEqual(c['core-s'].coolest.via, ['core-s', 'core-e']);
  assert.ok(climateBad.flagged.includes('core-e'));
});

test('CL1 review: a check computed for different inputs is not trusted; it is recomputed', () => {
  const r = climateRoutes(bad, checkGood);
  assert.match(r.warnings.join(' '), /supplied check was computed for different inputs/);
  assert.equal(r.cores.find(x => x.id === 'core-e').status, 'none', 'the routes reflect the plan that cuts the main stem');
  assert.equal(climateRoutes(good, {}).status, 'ok', 'a malformed check is recomputed, not thrown on');
});

test('CL2 CL3 review: missing temperatures leave a core unknown, never coolest, and are flagged; routes still pass through such cores', () => {
  const input = ws(); delete input.coreAreas.find(f => f.properties.dfm_id === 'core-w').properties.temp_c;
  const r = climateRoutes(input);
  const c = Object.fromEntries(r.cores.map(x => [x.id, x]));
  assert.equal(c['core-w'].status, 'unknown');
  assert.equal(c['core-n'].status, 'unknown', 'with core-w unmeasured and reachable, core-n cannot be called the coolest');
  assert.equal(c['core-s'].status, 'route');
  assert.ok(r.flagged.includes('core-w') && r.flagged.includes('core-n'));
});

test('CL2 temperatures and targets need sources; without a target any cooler reachable core counts', () => {
  const noSource = ws(); delete noSource.params.coreTempSource;
  assert.match(climateRoutes(noSource, checkGood).reasons.join(' '), /coreTempSource must record/);
  const noTarget = ws(); delete noTarget.params.climateSource;
  assert.match(climateRoutes(noTarget, checkGood).reasons.join(' '), /climateSource must record/);
  assert.match(climateRoutes(ws({temps: false})).reasons.join(' '), /No core records temp_c/);
  const anyCooler = ws(); delete anyCooler.params.climateWarmingC; delete anyCooler.params.climateSource;
  const r2 = climateRoutes(anyCooler);
  assert.equal(r2.cores.find(x => x.id === 'core-w').status, 'route');
  assert.match(r2.warnings.join(' '), /No warming target is set/);
});

test('FR1 FR2 FR3 the next woodlots: east and west would each complete two links; the northeast does not touch the committed spine', () => {
  assert.equal(frontier.status, 'ok', frontier.reasons.join(' '));
  valid('frontier', frontier);
  assert.deepEqual(frontier.committed.parcels, ['woodlot-1', 'woodlot-2', 'woodlot-3']);
  assert.deepEqual(keys(frontier.committed.links), ['core-n|core-s']);
  assert.ok(frontier.committed.share > 0.4 && frontier.committed.share < 0.6);
  assert.deepEqual(frontier.frontier.map(f => f.parcel), ['woodlot-4', 'woodlot-5', 'woodlot-9', 'woodlot-7', 'woodlot-8']);
  const [east, west] = frontier.frontier;
  assert.deepEqual(keys(east.completesLinks), ['core-e|core-n', 'core-e|core-s']);
  assert.equal(east.direction, 'E');
  assert.deepEqual(keys(west.completesLinks), ['core-n|core-w', 'core-s|core-w']);
  assert.ok(['W', 'NW'].includes(west.direction));
  assert.ok(frontier.frontier.every(f => f.bearingDeg >= 0 && f.bearingDeg <= 359));
  assert.deepEqual(frontier.laterParcels, ['woodlot-6']);
});

test('FR1 with no consent yet, or consent only where there is no spine, every parcel holding spine is listed without a direction', () => {
  const none = ws();
  none.parcels.forEach(p => { p.properties.consent = 'none'; delete p.properties.consent_ref; delete p.properties.consent_year; });
  const r = buildOutFrontier(none);
  assert.equal(r.status, 'ok');
  assert.equal(r.committed.share, 0);
  assert.match(r.warnings.join(' '), /every parcel holding spine can be the first/);
  assert.ok(r.frontier.length >= 8 && r.frontier.every(f => !('direction' in f)));
  // review: consent only on land without spine must not turn every parcel into a neighbor of nothing.
  const bare = ws();
  bare.parcels.forEach(p => { p.properties.consent = 'none'; delete p.properties.consent_ref; delete p.properties.consent_year; });
  bare.parcels.find(p => p.properties.dfm_id === 'woodlot-6').properties.consent = 'covered';
  bare.retained = bare.retained.filter(f => !['e1a'].includes(f.properties.source_id));
  const b = buildOutFrontier(bare);
  assert.match(b.warnings.join(' '), /hold no mapped spine yet/);
  assert.ok(b.frontier.every(f => !('direction' in f)));
});

test('FR1 FR3 review: only features marked spine count when any are marked; parcel IDs must be unique', () => {
  const unmarked = ws(); unmarked.retained.forEach(f => delete f.properties.spine);
  assert.match(buildOutFrontier(unmarked).warnings.join(' '), /No retained feature is marked spine/);
  const dupes = ws(); dupes.parcels[5].properties.dfm_id = 'woodlot-4';
  const r = buildOutFrontier(dupes);
  assert.equal(r.status, 'incomplete');
  assert.match(r.reasons.join(' '), /Parcel IDs must be unique/);
});

test('I8 projections, routes and the frontier are deterministic', () => {
  assert.equal(canonicalJSON(projectSpine(ws(), {years: YEARS})), canonicalJSON(projection));
  assert.equal(canonicalJSON(climateRoutes(ws(), checkGood)), canonicalJSON(climate));
});
