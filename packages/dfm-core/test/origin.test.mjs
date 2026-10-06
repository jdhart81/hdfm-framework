// Stand origin, native composition and native planting. Each test names the invariant it checks:
// O1-O6 in src/outlook.mjs, N1-N4 in src/connectivity.mjs. The aim these serve: true old growth
// regrown from native habitat, in a sea of young forest and plantations, without changing whether
// a link holds today.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import Ajv2020 from 'ajv/dist/2020.js';
import {
  checkConnectivitySync, projectSpine, climateRoutes, buildOutFrontier, spineNetwork, deriveSpine,
  toLandscapePackage, fromLandscapePackage, canonicalJSON, canonicalHashSync, STAND_ORIGINS,
} from '../src/index.mjs';
import {rect, woodlot} from '../fixtures/woodlot.mjs';
import {watershed, withAges} from '../fixtures/watershed.mjs';

const read = async name => JSON.parse(await readFile(new URL(`../../dfm-schema/${name}.schema.json`, import.meta.url), 'utf8'));
const results = await read('spine-results'), pkgSchema = await read('landscape-package');
const ajv = new Ajv2020({allErrors: true, strict: false});
ajv.addSchema(results);
const validPackage = ajv.compile(pkgSchema);
const valid = (def, r) => { const v = ajv.getSchema(`${results.$id}#/$defs/${def}`); assert.ok(v(r), JSON.stringify(v.errors)); };
const keys = list => (list ?? []).map(p => `${p.a}|${p.b}`).sort();
const sansChecksum = r => canonicalJSON({...r, inputChecksum: null, parameters: null});

const sq = (dfmId, x0, y0, x1, y1, extra = {}) => ({type: 'Feature', geometry: rect(x0, y0, x1, y1), properties: {dfm_id: dfmId, ...extra}});
const NATIVE = {native_share: 0.95, composition_source: 'Vegetation plots, fixture'};
/**
 * Two reserves 1,000 m apart, joined by a 200 m wide corridor of 200-year-old habitat, all in one
 * consenting woodlot. At a 150-year threshold the corridor is at old-growth age from the start.
 */
const corridor = (props = {}, {params = {}, treatments = [], overlay = []} = {}) => ({
  coreAreas: [sq('a', 0, 0, 300, 300, {core_class: 'reserve', stand_age: 200, ...NATIVE}), sq('b', 1300, 0, 1600, 300, {core_class: 'reserve', stand_age: 200, ...NATIVE})],
  retained: [sq('corridor', 300, 50, 1300, 250, {stand_age: 200, ...props}), ...overlay],
  parcels: [sq('lot', -100, -100, 1700, 400, {consent: 'covered', consent_year: 2020})],
  treatments,
  params: {minWidthM: 100, minWidthSource: 'Fixture value for tests', ageAsOfYear: 2026, oldGrowthAgeYears: 150, oldGrowthAgeSource: 'Fixture value for tests', ...params},
});
const ORIGIN = o => ({stand_origin: o, origin_source: 'Stand inventory, fixture'});
const year = (input, y = 2030) => {
  const r = projectSpine(input, {years: [y]});
  assert.equal(r.status, 'ok', r.reasons.join(' '));
  return r.milestones[0];
};

// --- Stand origin (O1, O2) ---

test('O2 a plantation never reaches old-growth age: the same 200-year-old corridor links at old-growth age only when it is not a plantation', () => {
  for (const origin of ['natural', 'planted']) {
    const m = year(corridor(ORIGIN(origin)));
    assert.deepEqual(keys(m.oldGrowthAgeLinks), ['a|b'], `${origin} habitat counts by its recorded age`);
    assert.equal(m.plantationM2, 0, `${origin}: no plantation area`);
  }
  const plantation = projectSpine(corridor(ORIGIN('plantation')), {years: [2030, 2126, 2526]});
  assert.equal(plantation.status, 'ok', plantation.reasons.join(' '));
  valid('projection', plantation);
  for (const m of plantation.milestones) {
    assert.deepEqual(keys(m.committedLinks), ['a|b'], `${m.year}: the plantation still carries today's structural link`);
    assert.deepEqual(m.oldGrowthAgeLinks, [], `${m.year}: but never an old-growth-age link, however old it gets`);
    assert.ok(Math.abs(m.plantationM2 - 200_000) < 500, `${m.year}: plantation area ${m.plantationM2} m² (1,000 m × 200 m)`);
  }
  assert.match(plantation.warnings.join(' '), /corridor is recorded as plantation, which never counts at old-growth age whatever its stand_age/);
  assert.match(plantation.limitations.join(' '), /a plantation never counts at old-growth age/);
});

test('O2 where a plantation overlaps old natural habitat it carves the old-age area, as a younger stand would', () => {
  const across = sq('pines', 600, 0, 700, 300, {stand_age: 300, ...ORIGIN('plantation')});
  const m = year(corridor(ORIGIN('natural'), {overlay: [across]}));
  assert.deepEqual(keys(m.committedLinks), ['a|b'], 'committed: the plantation is still habitat');
  assert.deepEqual(m.oldGrowthAgeLinks, [], 'old-growth age: the band of plantation breaks the old link');
  const planted = sq('oaks', 600, 0, 700, 300, {stand_age: 300, ...ORIGIN('planted')});
  assert.deepEqual(keys(year(corridor(ORIGIN('natural'), {overlay: [planted]})).oldGrowthAgeLinks), ['a|b'], 'planted native habitat of the same age does not');
});

test('O1 stand origin needs a known value and a source; a remnant cannot be a plantation', () => {
  const cases = [
    [{stand_origin: 'monoculture', origin_source: 'x'}, /stand_origin "monoculture"; use natural, planted, plantation/],
    [{stand_origin: 'plantation'}, /records stand_origin; record origin_source/],
    [{stand_origin: 'plantation', origin_source: 'Planting record', remnant: true, remnant_source: 'Survey'}, /marked both remnant .* and plantation; it cannot be both/],
  ];
  for (const [props, reason] of cases) {
    const r = projectSpine(corridor(props), {years: [2030]});
    assert.equal(r.status, 'incomplete', JSON.stringify(props));
    assert.match(r.reasons.join(' '), reason);
  }
  assert.deepEqual(STAND_ORIGINS, ['natural', 'planted', 'plantation']);
});

// --- Native composition (O3) ---

test('O3 with a minimum native share, habitat counts at old-growth age only where it records at least that share', () => {
  const params = {nativeShareMin: 0.8, nativeShareSource: 'Co-op policy, fixture'};
  const high = projectSpine(corridor({native_share: 0.92, composition_source: 'Vegetation plots'}, {params}), {years: [2030]});
  assert.equal(high.status, 'ok', high.reasons.join(' '));
  valid('projection', high);
  assert.deepEqual(keys(high.milestones[0].oldGrowthAgeLinks), ['a|b']);
  assert.equal(high.milestones[0].belowNativeShareM2, 0);
  assert.deepEqual([high.parameters.nativeShareMin, high.parameters.nativeShareSource], [0.8, 'Co-op policy, fixture']);
  assert.match(high.limitations.join(' '), /at least 0.8 of its cover or basal area is recorded as native species/);

  // Old, but mostly introduced species (say, a stand taken over by Norway maple, or a never-plowed
  // prairie overrun by smooth brome): committed, never old-growth age.
  const low = year(corridor({native_share: 0.4, composition_source: 'Vegetation plots'}, {params}));
  assert.deepEqual(keys(low.committedLinks), ['a|b']);
  assert.deepEqual(low.oldGrowthAgeLinks, []);
  assert.ok(Math.abs(low.belowNativeShareM2 - 200_000) < 500, `${low.belowNativeShareM2}`);

  // No recorded share never counts as old, like a missing age (PJ3).
  const none = projectSpine(corridor({}, {params}), {years: [2030]});
  assert.deepEqual(none.milestones[0].oldGrowthAgeLinks, []);
  assert.match(none.warnings.join(' '), /corridor does not record a native_share of at least 0.8 \(params.nativeShareMin\), so it never counts at old-growth age/);
});

test('O3 a never-plowed remnant counts from the start only when its native share is recorded at or above the minimum', () => {
  const params = {nativeShareMin: 0.8, nativeShareSource: 'Co-op policy, fixture'};
  const remnant = share => corridor({remnant: true, remnant_source: 'Land survey: never plowed', native_share: share, composition_source: 'Plant survey'}, {params});
  assert.deepEqual(keys(year(remnant(0.9)).oldGrowthAgeLinks), ['a|b']);
  assert.deepEqual(year(remnant(0.3)).oldGrowthAgeLinks, [], 'a remnant overrun by introduced grasses does not');
});

test('O3 native share needs a share from 0 to 1 and a source; the minimum needs a source', () => {
  const cases = [
    [corridor({native_share: 1.5, composition_source: 'x'}), /native_share 1.5; use a share from 0 to 1/],
    [corridor({native_share: 0.5}), /records native_share; record composition_source/],
    [corridor({}, {params: {nativeShareMin: 0.8}}), /params.nativeShareSource must record where the minimum native share comes from/],
    [corridor({}, {params: {nativeShareMin: 0, nativeShareSource: 'x'}}), /params.nativeShareMin must be a share above 0 and at most 1/],
  ];
  for (const [input, reason] of cases) {
    const r = projectSpine(input, {years: [2030]});
    assert.equal(r.status, 'incomplete');
    assert.match(r.reasons.join(' '), reason);
  }
});

test('O3 O6 without a minimum, a recorded native share changes nothing but the checksum', () => {
  const plain = projectSpine(corridor(), {years: [2030]});
  const recorded = projectSpine(corridor({native_share: 0.1, composition_source: 'Vegetation plots'}), {years: [2030]});
  assert.notEqual(plain.inputChecksum, recorded.inputChecksum);
  assert.equal(sansChecksum(recorded), sansChecksum(plain));
  assert.ok(!('belowNativeShareM2' in recorded.milestones[0]) && !('plantationM2' in recorded.milestones[0]), 'no new fields without the records they report');
});

// --- Structure is unchanged (O4) ---

test('O4 stand origin and native share never change the check, the network, climate routes or the frontier', () => {
  const plain = {...watershed(), retained: withAges(deriveSpine(watershed()).features)};
  const labelled = structuredClone(plain);
  labelled.retained.forEach((f, i) => Object.assign(f.properties, ORIGIN(i % 2 ? 'plantation' : 'natural'), {native_share: (i % 10) / 10, composition_source: 'Fixture'}));
  labelled.coreAreas.forEach(f => Object.assign(f.properties, ORIGIN('natural'), NATIVE));
  labelled.params.nativeShareMin = 0.5; labelled.params.nativeShareSource = 'Fixture';
  assert.equal(sansChecksum(checkConnectivitySync(labelled)), sansChecksum(checkConnectivitySync(plain)));
  assert.equal(sansChecksum(climateRoutes(labelled)), sansChecksum(climateRoutes(plain)));
  assert.equal(sansChecksum(buildOutFrontier(labelled)), sansChecksum(buildOutFrontier(plain)));
  assert.equal(sansChecksum(spineNetwork(labelled, {cuts: false})), sansChecksum(spineNetwork(plain, {cuts: false})));
});

test('O5 PJ4 PJ5 with plantations and a native minimum, links still nest and never decrease, and committed links are unchanged', () => {
  const plain = {...watershed(), retained: withAges(deriveSpine(watershed()).features)};
  const mixed = structuredClone(plain);
  // The main stem's corridor was replanted as a plantation; everything else is natural and mostly native.
  for (const f of mixed.retained) Object.assign(f.properties, f.properties.source_id === 'main' ? ORIGIN('plantation') : ORIGIN('natural'), NATIVE);
  for (const f of mixed.coreAreas) Object.assign(f.properties, NATIVE);
  Object.assign(mixed.params, {nativeShareMin: 0.8, nativeShareSource: 'Co-op policy, fixture'});
  const years = [2026, 2126];
  const [a, b] = [projectSpine(plain, {years}), projectSpine(mixed, {years})];
  assert.equal(b.status, 'ok', b.reasons.join(' '));
  valid('projection', b);
  const after = new Set(keys(b.linkedAfter));
  let prev = new Set();
  b.milestones.forEach((m, i) => {
    assert.deepEqual(keys(m.committedLinks), keys(a.milestones[i].committedLinks), `${m.year}: committed links do not read origin`);
    const o = new Set(keys(m.oldGrowthAgeLinks));
    for (const k of o) assert.ok(new Set(keys(m.committedLinks)).has(k) && after.has(k), `${m.year}: ${k} nests`);
    for (const k of prev) assert.ok(o.has(k), `${m.year}: ${k} is not lost`);
    assert.ok(o.size <= keys(a.milestones[i].oldGrowthAgeLinks).length, `${m.year}: a plantation can only remove old-age links`);
    prev = o;
  });
  for (const m of b.milestones) assert.ok(m.plantationM2 > 0, `${m.year}: the committed plantation is reported`);
  assert.equal(b.milestones[1].belowNativeShareM2, 0, 'everything records a native share above the minimum');
  assert.ok(b.milestones[1].oldGrowthAgeM2 < a.milestones[1].oldGrowthAgeM2, 'the plantation is left out of the old-age area');
  assert.ok(!keys(b.milestones[1].oldGrowthAgeLinks).some(k => k.includes('core-s')), 'nothing reaches the riparian core at old-growth age through the plantation stem');
  assert.ok(keys(a.milestones[1].oldGrowthAgeLinks).some(k => k.includes('core-s')), 'as natural forest of the same age, it does');
});

// --- Native planting (N1-N4) ---

const PLANT = {dfm_id: 'plant-3', period: '2027', intensity: 'restoration-planting', corridor_permitted: true, reason: 'Underplanting a canopy gap'};
const planting = (props = {}, params = {}) => {
  const input = woodlot();
  input.treatments = [{type: 'Feature', geometry: rect(700, 300, 800, 800), properties: {...PLANT, ...props}}];
  Object.assign(input.params, params);
  return input;
};
const SOURCE = {native_status_source: 'USDA PLANTS native status for Vermont'};
const OAK_PINE = [{name: 'Quercus rubra', native: true}, {name: 'Pinus strobus', native: true}];

test('N1 a corridor planting of native species stays permitted', () => {
  const r = checkConnectivitySync(planting({species: OAK_PINE, ...SOURCE}));
  assert.equal(r.status, 'pass', r.reasons.join(' '));
  assert.match(r.warnings.join(' '), /plant-3 overlaps retained habitat or a core as a permitted light treatment/);
});

test('N1 a corridor planting that includes an introduced species is not permitted: it removes habitat and the check says why', () => {
  const r = checkConnectivitySync(planting({species: [...OAK_PINE, {name: 'Acer platanoides', native: false}], ...SOURCE}));
  assert.equal(r.status, 'fail');
  assert.deepEqual(r.lostLinks[0].causes, ['plant-3'], 'the unit spans the corridor, so removing it breaks the link');
  assert.match(r.reasons.join(' '), /Treatment plant-3 overlaps retained habitat or a core by \d+ m²; its light-treatment permission does not hold because it plants species recorded as not native \(Acer platanoides\), and only native planting is permitted inside corridors and cores\./);
  assert.match(r.warnings.join(' '), /plant-3 is marked corridor_permitted but plants species recorded as not native \(Acer platanoides\)/);
});

test('N1 a species list needs a native-status source, on the unit or in the parameters', () => {
  const unsourced = checkConnectivitySync(planting({species: OAK_PINE}));
  assert.equal(unsourced.status, 'fail');
  assert.match(unsourced.reasons.join(' '), /lists its species without a native-status source/);
  const fromParams = checkConnectivitySync(planting({species: OAK_PINE}, {nativeStatusSource: 'Flora of Vermont, 2026'}));
  assert.equal(fromParams.status, 'pass', fromParams.reasons.join(' '));
});

test('N2 with params.nativeStatusSource, a corridor planting must list its species', () => {
  const r = checkConnectivitySync(planting({}, {nativeStatusSource: 'Flora of Vermont, 2026'}));
  assert.equal(r.status, 'fail');
  assert.match(r.reasons.join(' '), /lists no species, and params.nativeStatusSource requires a species list for planting inside corridors and cores/);
});

test('N4 a planting with no species list and no native-status policy is permitted exactly as in engine 0.2.0', () => {
  const r = checkConnectivitySync(planting());
  assert.equal(r.status, 'pass', r.reasons.join(' '));
  assert.doesNotMatch(r.warnings.join(' '), /species|native/);
  // Other light treatments never read a species list.
  const thin = checkConnectivitySync(planting({intensity: 'single-tree-selection', species: [{name: 'Acer platanoides', native: false}]}));
  assert.equal(thin.status, 'pass', thin.reasons.join(' '));
});

test('N3 a malformed species list or native-status source is incomplete, naming the unit', () => {
  const cases = [
    [{species: []}, /Treatment plant-3 species must list each planted species as \{name, native: true or false\}/],
    [{species: 'oak'}, /plant-3 species must list/],
    [{species: [{name: 'Quercus rubra'}]}, /plant-3 species must list/],
    [{species: [{name: '', native: true}]}, /plant-3 species must list/],
    [{species: OAK_PINE, native_status_source: ' '}, /plant-3 native_status_source must record where/],
  ];
  for (const [props, reason] of cases) {
    const r = checkConnectivitySync(planting(props));
    assert.equal(r.status, 'incomplete', JSON.stringify(props));
    assert.match(r.reasons.join(' '), reason);
  }
  const p = checkConnectivitySync(planting({}, {nativeStatusSource: ''}));
  assert.equal(p.status, 'incomplete');
  assert.match(p.reasons.join(' '), /params.nativeStatusSource must record where native status comes from/);
});

test('N1 PJ2 projections apply the same planting rule: an introduced-species planting across the corridor removes it at every milestone', () => {
  const unit = props => [{type: 'Feature', geometry: rect(600, 0, 700, 300), properties: {...PLANT, dfm_id: 'plant-x', ...props}}];
  const native = year(corridor({}, {treatments: unit({species: OAK_PINE, ...SOURCE})}));
  assert.deepEqual(keys(native.committedLinks), ['a|b']);
  const introduced = year(corridor({}, {treatments: unit({species: [{name: 'Robinia pseudoacacia', native: false}], ...SOURCE})}));
  assert.deepEqual(introduced.committedLinks, []);
});

// --- Package and schema ---

test('the Landscape Package carries the new records, and the schema checks them', () => {
  const input = corridor({...ORIGIN('natural'), native_share: 0.9, composition_source: 'Plots'}, {
    params: {nativeShareMin: 0.8, nativeShareSource: 'Policy', nativeStatusSource: 'USDA PLANTS'},
    treatments: [{type: 'Feature', geometry: rect(600, 0, 700, 300), properties: {...PLANT, species: OAK_PINE, ...SOURCE}}],
  });
  const pkg = toLandscapePackage(input, {name: 'Origin fixture', created: '2026-10-06T00:00:00Z'});
  assert.ok(validPackage(pkg), JSON.stringify(validPackage.errors));
  const back = fromLandscapePackage(JSON.parse(JSON.stringify(pkg)));
  assert.equal(canonicalHashSync(back), canonicalHashSync(fromLandscapePackage(pkg)));
  assert.deepEqual(back.retained[0].properties, input.retained[0].properties);

  const broken = [
    p => { p.layers.retained[0].properties.stand_origin = 'monoculture'; },
    p => { delete p.layers.retained[0].properties.origin_source; },
    p => { p.layers.retained[0].properties.native_share = 1.2; },
    p => { delete p.layers.retained[0].properties.composition_source; },
    p => { Object.assign(p.layers.retained[0].properties, {stand_origin: 'plantation', remnant: true, remnant_source: 'Survey'}); },
    p => { delete p.params.nativeShareSource; },
    p => { p.layers.treatments[0].properties.species = [{name: 'Quercus rubra'}]; },
    p => { p.layers.treatments[0].properties.species = []; },
  ];
  for (const [i, change] of broken.entries()) {
    const p = structuredClone(pkg);
    change(p);
    assert.ok(!validPackage(p), `case ${i} should be rejected`);
  }
});
