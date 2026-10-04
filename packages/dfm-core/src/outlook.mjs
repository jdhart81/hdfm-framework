// Time, climate and build-out for the old-growth spine (DFM Build Spec, "Old-growth spine").
//
//   projectSpine(input, {years})   at each milestone year: core pairs linked through committed
//                                  habitat, and through habitat at old-growth age
//   climateRoutes(input, check?)   for each core, the coolest core or exit it can reach through
//                                  links that hold after the plan
//   buildOutFrontier(input)        neighbors whose land would extend the committed spine: the
//                                  links each would complete and its direction
//
// Invariants (tests in test/outlook.test.mjs):
//   PJ1 Commitment by year. A parcel counts from its consent_year when consent covers it (no
//       year: from params.ageAsOfYear; never later than it), or from its planned_year, which must be
//       after params.ageAsOfYear, when it plans to join. Consent never lapses in a projection;
//       revocation is a present-tense change the check handles.
//   PJ2 The proposed treatment units apply at every milestone, to links and to the areas reported.
//   PJ3 Old-growth age at year Y: stand_age + (Y - params.ageAsOfYear) >= params.oldGrowthAgeYears,
//       which needs a recorded source. A remnant (remnant: true with remnant_source: never plowed
//       or clear-cut) is at old-growth age in every year. Where features overlap, the youngest
//       recorded age applies, and a feature with neither an age nor remnant status counts as young
//       there and never as old. That applies to core areas too: a core is a link's endpoint whatever
//       its age, but a path runs only through old-age habitat. Age is reported, never condition (I6).
//   PJ4 Nesting: links at old-growth age ⊆ committed links ⊆ links after the plan, every year.
//   PJ5 Monotone in time: committed and old-growth-age links never decrease from one milestone to
//       the next (no loss to fire, storm, conversion or a new road is predicted).
//   CL1 Climate routes use only the links the check finds after the plan, chained through linked
//       cores. A supplied check must match the input's checksum; otherwise it is recomputed.
//   CL2 A core without temp_c is 'unknown'. Temperatures need params.coreTempSource; a warming
//       target (params.climateWarmingC) needs params.climateSource.
//   CL3 'route': a reachable core or exit is cooler by at least the warming target (by any amount
//       if no target is set); 'short': cooler, but by less; 'none': nothing reachable is cooler
//       although a cooler core or exit exists; 'coolest': every core has a temperature and nothing
//       is cooler; 'unknown': missing temperatures leave it undecided. 'none', 'short' and
//       'unknown' are flagged.
//   CL4 Exits (input.exits) are where the spine continues into a neighboring landscape, each with
//       the temp_c of the ground it leads to. In flat country cooler ground is usually beyond one
//       landscape, so routes chain across landscapes through exits. A core reaches an exit by the
//       same rule as another core (minimum width, stepping stones, half the exit remaining), on the
//       habitat after the plan. Exits end routes; they never pass one on and get no status.
//   FR1 Frontier parcels have no covering consent, hold mapped spine (retained habitat; only the
//       features marked spine when any are), and that spine touches the committed spine. With no
//       committed spine yet, every parcel holding spine is listed, without a direction.
//   FR2 Each lists the core pairs that would become linked through committed forest if it alone
//       joined now, its spine area, and its direction from the committed piece it adjoins.
//   FR3 Ordered by links completed, then spine area, then ID. Parcel IDs must be unique.
//   I7, I8 Invalid inputs return 'incomplete' with reasons; results are deterministic.

import * as turf from './turf.mjs';
import {canonicalHashSync} from './hash.mjs';
import {id, isPoly, union, intersect, difference, areaM2, buffer, parts, centroid, compass} from './geo.mjs';
import {validateInput, effectiveHabitat, linkedPairs, pairObj, checkConnectivitySync, gapOf, LIMITS} from './connectivity.mjs';

export const OUTLOOK_VERSION = 'dfm-outlook-0.2.0';
const MAX_MILESTONES = 12;
const isYear = y => Number.isInteger(y) && y >= 1800 && y <= 3000;
const hasText = s => typeof s === 'string' && s.trim().length > 0;
const byId = (a, b) => (id(a) < id(b) ? -1 : id(a) > id(b) ? 1 : 0);
const hasAge = f => Number.isFinite(f?.properties?.stand_age);
/** Never plowed or clear-cut, as recorded: at old-growth age in every year (PJ3). */
const isRemnant = f => f?.properties?.remnant === true;
const hasAgeRecord = f => hasAge(f) || isRemnant(f);

/** Clip features to an area, keeping their properties. */
function clip(features, area) {
  if (!area) return [];
  const out = [];
  for (const f of features) {
    const g = intersect(f, area);
    if (g) out.push({type: 'Feature', geometry: g.geometry, properties: f.properties});
  }
  return out;
}

/** Start the result: checksum first, so even malformed input gets a well-formed answer. */
function begin(engine, payload) {
  try { return {engine, inputChecksum: canonicalHashSync(payload ?? null)}; }
  catch { return {engine, inputChecksum: null}; }
}

/** Shared checks: the connectivity input, then (where parcels matter) unique parcel IDs. Never throws (I7). */
function validateBase(input, {parcels = true} = {}) {
  try {
    const v = validateInput(input);
    const ids = parcels ? (input?.parcels ?? []).map(id) : [];
    if (ids.some(x => !x)) v.errors.push('Every parcel needs properties.dfm_id.');
    else if (new Set(ids).size !== ids.length) v.errors.push('Parcel IDs must be unique.');
    return v;
  } catch (e) {
    return {errors: [`Invalid input: ${e.message}`], warnings: [], cores: []};
  }
}

/**
 * Habitat and links through the layers inside `area`, with the plan's treatment units applied.
 * Memoized by key. Returns {habitat, pairs}.
 */
function linker(input, cores) {
  const cache = new Map();
  const run = (key, retained, coreAreas, coreHabitat = null) => {
    if (cache.has(key)) return cache.get(key);
    let habitat = null, pairs = [];
    if (retained.length || coreAreas.length) {
      const e = effectiveHabitat({...input, retained, coreAreas}, input.treatments ?? [], []);
      habitat = e.habitat;
      pairs = linkedPairs(habitat, cores, input.params.minWidthM, {coreHabitat: coreHabitat ?? habitat, gapM: gapOf(input.params), barrier: e.barrier}).pairs;
    }
    const out = {habitat, pairs};
    cache.set(key, out);
    return out;
  };
  return {
    committed: (key, area) => run(`c:${key}`, clip(input.retained, area), clip(input.coreAreas, area)),
    // Paths only through old-age habitat (youngest age wins where features overlap); endpoints are
    // measured against the committed habitat, so a core need not itself be old to be reached.
    old: (key, area, qualifies, committedHabitat) => {
      const all = [...input.retained, ...input.coreAreas];
      const oldArea = difference(union(clip(all.filter(qualifies), area)), union(clip(all.filter(f => !qualifies(f)), area)));
      const retained = oldArea ? [{type: 'Feature', geometry: oldArea.geometry, properties: {}}] : [];
      return run(`o:${key}`, retained, [], committedHabitat);
    },
  };
}

/** Projection-specific checks on top of the connectivity input checks. */
function validateProjection(input, years, errors, warnings) {
  const p = input?.params ?? {};
  if (!isYear(p.ageAsOfYear)) errors.push('params.ageAsOfYear must be the year (1800-3000) that ages and consent are recorded for.');
  if (p.oldGrowthAgeYears != null) {
    if (!(Number.isFinite(p.oldGrowthAgeYears) && p.oldGrowthAgeYears > 0 && p.oldGrowthAgeYears <= 2000)) errors.push('params.oldGrowthAgeYears must be an age in years between 0 and 2,000.');
    if (!hasText(p.oldGrowthAgeSource)) errors.push('params.oldGrowthAgeSource must record where the old-growth age threshold comes from.');
  }
  if (!Array.isArray(years) || !years.length) errors.push('Milestone years are required: pass {years} or set params.milestoneYears.');
  else {
    if (years.length > MAX_MILESTONES) errors.push(`Use at most ${MAX_MILESTONES} milestone years.`);
    for (const y of years) if (!isYear(y)) errors.push(`Milestone ${y} is not a year between 1800 and 3000.`);
    for (let i = 1; i < years.length; i++) if (!(years[i] > years[i - 1])) errors.push('Milestone years must increase.');
    if (isYear(p.ageAsOfYear) && years.some(y => y < p.ageAsOfYear)) errors.push(`Milestone years must be ${p.ageAsOfYear} (params.ageAsOfYear) or later; the projection runs forward only.`);
  }
  for (const f of [...(input?.retained ?? []), ...(input?.coreAreas ?? [])]) {
    const a = f?.properties?.stand_age, rem = f?.properties?.remnant, name = id(f) || '(no id)';
    if (a != null && !(Number.isFinite(a) && a >= 0 && a <= 3000)) errors.push(`Feature ${name} has stand_age ${a}; use an age in years from 0 to 3,000.`);
    if (rem != null && typeof rem !== 'boolean') errors.push(`Feature ${name} remnant must be true or false.`);
    else if (rem === true && !hasText(f.properties.remnant_source)) errors.push(`Feature ${name} is marked remnant; record remnant_source, such as a survey or land record showing it was never plowed or clear-cut.`);
    else if (rem === true && a != null) warnings.push(`Feature ${name} is marked remnant and also records stand_age ${a}; remnant status applies.`);
  }
  for (const f of input?.parcels ?? []) {
    const pr = f?.properties ?? {}, name = id(f) || '(no id)';
    if (pr.consent_year != null) {
      if (!isYear(pr.consent_year)) errors.push(`Parcel ${name} consent_year must be a year.`);
      else if (pr.consent !== 'covered') errors.push(`Parcel ${name} has a consent_year but its consent is not 'covered'.`);
      else if (isYear(p.ageAsOfYear) && pr.consent_year > p.ageAsOfYear) errors.push(`Parcel ${name} has covering consent from ${pr.consent_year}, after params.ageAsOfYear ${p.ageAsOfYear}; use planned_year for a future join.`);
    }
    if (pr.planned_year != null) {
      if (!isYear(pr.planned_year)) errors.push(`Parcel ${name} planned_year must be a year.`);
      else if (pr.consent !== 'covered' && isYear(p.ageAsOfYear) && pr.planned_year <= p.ageAsOfYear) errors.push(`Parcel ${name} plans to join in ${pr.planned_year}, not after params.ageAsOfYear ${p.ageAsOfYear}; a parcel that has joined needs covering consent.`);
    }
  }
}

/**
 * Project the spine forward (PJ1-PJ5).
 * @param {object} input - connectivity input plus params.ageAsOfYear, optional
 *   params.oldGrowthAgeYears + oldGrowthAgeSource, params.milestoneYears; retained features and
 *   cores may carry stand_age; parcels may carry consent_year (covered) or planned_year (joining later).
 * @param {{years?: number[]}} [options] milestone years (default params.milestoneYears)
 * Areas are retained habitat outside core areas, after the plan's treatment units, roads and open water.
 */
export function projectSpine(input, {years} = {}) {
  const list = years ?? input?.params?.milestoneYears;
  const base = begin(OUTLOOK_VERSION, {input: input ?? null, years: list ?? null});
  const {errors, warnings, cores} = validateBase(input);
  if (!errors.length) validateProjection(input, list, errors, warnings);
  if (errors.length) return {...base, status: 'incomplete', reasons: errors, warnings};
  try {
    const p = input.params, asOf = p.ageAsOfYear, ogAge = p.oldGrowthAgeYears ?? null;
    const parcels = (input.parcels ?? []).slice().sort(byId);
    const startOf = f => (f.properties?.consent === 'covered' ? (f.properties.consent_year ?? asOf) : (f.properties?.planned_year ?? null));
    if (!parcels.length) warnings.push('No parcels are mapped, so nothing is committed and no link is projected.');
    for (const f of parcels) if (f.properties?.consent === 'covered' && f.properties.planned_year != null) warnings.push(`Parcel ${id(f)} already has covering consent; its planned_year is ignored.`);
    if (ogAge == null) warnings.push('params.oldGrowthAgeYears is not set, so only committed links are projected.');
    else if (![...input.retained, ...input.coreAreas].some(hasAgeRecord)) warnings.push('No retained habitat or core records a stand_age or remnant status, so no habitat reaches old-growth age in the projection.');

    const links = linker(input, cores);
    const coresUnion = union(input.coreAreas);
    const outsideCores = h => areaM2(difference(h, coresUnion));
    const eAfter = effectiveHabitat(input, input.treatments ?? [], []);
    const after = new Set(linkedPairs(eAfter.habitat, cores, p.minWidthM, {gapM: gapOf(p), barrier: eAfter.barrier}).pairs);
    const unknownGross = union(input.retained.filter(f => !hasAgeRecord(f)));
    const conflicts = [];
    const milestones = [];
    let prevCommitted = new Set(), prevOld = new Set();
    for (const year of list) {
      const joined = parcels.filter(f => { const s = startOf(f); return s != null && s <= year; });
      const joinedKey = joined.map(id).join(',');
      const area = joined.length ? union(joined) : null;
      const committed = links.committed(joinedKey, area);
      // PJ4: a pair cannot be linked through part of the habitat unless it is linked through all of it.
      const anomalies = [];
      const committedPairs = committed.pairs.filter(k => after.has(k) || (anomalies.push(k), false));
      let old = null;
      if (ogAge != null) {
        const qualifies = f => isRemnant(f) || (hasAge(f) && f.properties.stand_age + (year - asOf) >= ogAge);
        const all = [...input.retained, ...input.coreAreas];
        const young = union(all.filter(f => hasAgeRecord(f) && !qualifies(f)));
        if (young && areaM2(intersect(union(all.filter(qualifies)), young)) >= 1) conflicts.push(year);
        const keep = all.map((f, i) => (qualifies(f) ? i : -1)).filter(i => i >= 0).join(',');
        const r = links.old(`${joinedKey}:${keep}`, area, qualifies, committed.habitat);
        old = {pairs: r.pairs.filter(k => committedPairs.includes(k) || (anomalies.push(k), false)), m2: outsideCores(r.habitat)};
      }
      if (anomalies.length) warnings.push(`Year ${year}: ${[...new Set(anomalies)].join(', ')} linked only within numerical tolerance through part of the habitat; not counted.`);
      const cSet = new Set(committedPairs), oSet = new Set(old?.pairs ?? []);
      if ([...prevCommitted].some(k => !cSet.has(k)) || [...prevOld].some(k => !oSet.has(k))) warnings.push(`Year ${year}: a link projected earlier is missing; check consent and planned years.`);
      prevCommitted = cSet; prevOld = oSet;
      milestones.push({
        year, committedParcels: joined.map(id), committedLinks: committedPairs.map(pairObj), committedM2: outsideCores(committed.habitat),
        oldGrowthAgeLinks: old ? old.pairs.map(pairObj) : null, oldGrowthAgeM2: old ? old.m2 : null,
        unknownAgeM2: unknownGross && committed.habitat ? outsideCores(intersect(committed.habitat, unknownGross)) : 0,
      });
    }
    if (conflicts.length) warnings.push(`Features with different recorded ages overlap (milestones ${conflicts.join(', ')}); the youngest age applies where they overlap.`);
    return {
      ...base, status: 'ok', reasons: [], warnings: [...new Set(warnings)],
      parameters: {ageAsOfYear: asOf, oldGrowthAgeYears: ogAge, oldGrowthAgeSource: p.oldGrowthAgeSource ?? null, minWidthM: p.minWidthM},
      linkedAfter: [...after].sort().map(pairObj),
      milestones,
      assumptions: [
        'Woodlots join in the years recorded (consent_year, planned_year) and consent does not lapse.',
        `Recorded ages grow one year per year from ${asOf} and remnants stay at old-growth age; no loss to fire, storm, pests, conversion or a new road is predicted.`,
        'The proposed treatment units apply at every milestone.',
      ],
      limitations: [
        'Old-growth age means a recorded age at or above the threshold, or recorded remnant status; where features overlap the youngest age applies, and features with neither never count. Old-growth condition needs field evidence.',
        'Areas are retained habitat outside core areas, after the treatment units, roads and open water.',
        gapOf(p) ? `Structural connectivity at the minimum width, with stepping stones up to ${gapOf(p)} m apart; not species movement or genetics.` : 'Structural connectivity at the minimum width only; not species movement or genetics.',
      ],
    };
  } catch (e) {
    return {...base, status: 'incomplete', reasons: [`Geometry engine error: ${e.message}`], warnings: [...new Set(warnings)]};
  }
}

/** Exits (CL4): polygons where the spine leaves the landscape, each with the temp_c it leads to. */
function readExits(input, errors) {
  const list = input.exits ?? [];
  if (!Array.isArray(list)) { errors.push('exits must be an array of GeoJSON features.'); return []; }
  const coreIds = new Set((input.coreAreas ?? []).map(id)), out = [];
  for (const [i, f] of list.entries()) {
    if (!f?.geometry || !Array.isArray(f.geometry.coordinates) || !isPoly(f)) { errors.push(`exits[${i}] must be a polygon where the spine leaves the landscape.`); continue; }
    const x = id(f);
    if (!x || x.includes('|')) { errors.push(`exits[${i}] needs properties.dfm_id without a "|" character.`); continue; }
    if (coreIds.has(x)) { errors.push(`Exit ${x} has the same ID as a core area.`); continue; }
    const t = f.properties?.temp_c;
    if (!(Number.isFinite(t) && t >= -60 && t <= 60)) { errors.push(`Exit ${x} needs temp_c: the temperature of the ground the spine continues into, in °C between -60 and 60.`); continue; }
    let lonLat = true;
    turf.coordEach(f, c => { if (!(Math.abs(c[0]) <= 180 && Math.abs(c[1]) <= 85)) lonLat = false; });
    if (!lonLat || !turf.booleanValid(f)) { errors.push(`Exit ${x} needs valid WGS84 polygon geometry.`); continue; }
    out.push({id: x, feature: f, tempC: t, toward: hasText(f.properties.toward) ? f.properties.toward : null});
  }
  if (new Set(out.map(e => e.id)).size !== out.length) errors.push('Exit IDs must be unique.');
  if (out.length) {
    // Exits sit at the landscape's edge, so the landscape with its exits keeps the engine's extent limit.
    const layers = ['coreAreas', 'retained', 'roads', 'water', 'crossings', 'treatments', 'parcels'].flatMap(k => input[k] ?? []).filter(f => f?.geometry);
    const [w, s, e, n] = turf.bbox(turf.featureCollection([...layers, ...out.map(x => x.feature)]));
    if (e - w > LIMITS.extentDegrees || n - s > LIMITS.extentDegrees) errors.push(`With its exits the landscape spans more than ${LIMITS.extentDegrees}° (about 50 km); draw each exit where the spine leaves the mapped landscape.`);
  }
  return out.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/**
 * Climate routes for each core (CL1-CL4).
 * @param {object} input - connectivity input; cores may carry temp_c (with params.coreTempSource);
 *   optional params.climateWarmingC (degrees C) with params.climateSource; optional exits
 *   (polygons with dfm_id, temp_c and toward) where the spine continues into the next landscape.
 * @param {object} [check] - a checkConnectivity result for this same input, to avoid recomputing.
 */
export function climateRoutes(input, check = null) {
  const base = begin(OUTLOOK_VERSION, input);
  const {errors, warnings} = validateBase(input, {parcels: false}); // climate routes do not use parcels
  try {
    const p = input?.params ?? {};
    const temps = new Map();
    let exits = [];
    if (!errors.length) {
      for (const f of input.coreAreas ?? []) {
        const t = f?.properties?.temp_c;
        if (t == null) continue;
        if (!(Number.isFinite(t) && t >= -60 && t <= 60)) errors.push(`Core ${id(f) || '(no id)'} temp_c must be a temperature in °C between -60 and 60.`);
        else temps.set(id(f), t);
      }
      exits = readExits(input, errors);
      if (!temps.size) errors.push('No core records temp_c, so no climate route can be found.');
      else if (!hasText(p.coreTempSource)) errors.push('params.coreTempSource must record where core and exit temperatures come from.');
      if (p.climateWarmingC != null) {
        if (!(Number.isFinite(p.climateWarmingC) && p.climateWarmingC > 0 && p.climateWarmingC <= 10)) errors.push('params.climateWarmingC must be a warming in °C between 0 and 10.');
        if (!hasText(p.climateSource)) errors.push('params.climateSource must record where the warming target comes from.');
      }
    }
    if (errors.length) return {...base, status: 'incomplete', reasons: errors, warnings};

    let result = check;
    if (result && result.inputChecksum !== base.inputChecksum) { warnings.push('The supplied check was computed for different inputs; it was recomputed.'); result = null; }
    result ??= checkConnectivitySync(input);
    if (result.status === 'incomplete') return {...base, status: 'incomplete', reasons: result.reasons, warnings};

    const ids = (input.coreAreas ?? []).map(id).sort();
    const adj = new Map([...ids, ...exits.map(e => e.id)].map(c => [c, []]));
    const link = (a, b) => { adj.get(a)?.push(b); adj.get(b)?.push(a); };
    for (const {a, b} of result.linkedAfter) link(a, b);
    // CL4: links to exits by the same rule, on the habitat after the plan.
    const exitIds = new Set(exits.map(e => e.id)), exitOf = new Map(exits.map(e => [e.id, e]));
    if (exits.length) {
      const e = effectiveHabitat(input, input.treatments ?? [], []);
      const ends = [...(input.coreAreas ?? []).map(f => ({id: id(f), feature: f})), ...exits.map(x => ({id: x.id, feature: x.feature}))];
      for (const k of e.habitat ? linkedPairs(e.habitat, ends, p.minWidthM, {gapM: gapOf(p), barrier: e.barrier}).pairs : []) {
        const [a, b] = k.split('|');
        if (exitIds.has(a) !== exitIds.has(b)) link(a, b); // core to exit only: core links come from the check
      }
    }
    for (const nbrs of adj.values()) nbrs.sort();
    const known = new Map([...temps, ...exits.map(e => [e.id, e.tempC])]);
    const target = p.climateWarmingC ?? null, allKnown = ids.every(c => temps.has(c));
    const cores = ids.map(c => {
      if (!temps.has(c)) return {id: c, tempC: null, status: 'unknown', coolest: null, coolingC: null};
      const t = temps.get(c);
      // Breadth-first from the core, neighbors in ID order: shortest chains, deterministic ties. Exits end a chain.
      const parent = new Map([[c, null]]), order = [c];
      for (let i = 0; i < order.length; i++) {
        if (exitIds.has(order[i])) continue;
        for (const w of adj.get(order[i])) if (!parent.has(w)) { parent.set(w, order[i]); order.push(w); }
      }
      const reachable = order.slice(1);
      const cooler = reachable.filter(w => known.has(w) && known.get(w) < t);
      if (!cooler.length) {
        const status = reachable.some(w => !known.has(w)) ? 'unknown' : [...known.values()].some(x => x < t) ? 'none' : allKnown ? 'coolest' : 'unknown';
        return {id: c, tempC: t, status, coolest: null, coolingC: null};
      }
      const best = cooler.reduce((m, w) => (known.get(w) < known.get(m) ? w : m)); // first in breadth-first order wins ties
      const via = [];
      for (let x = best; x != null; x = parent.get(x)) via.unshift(x);
      const cooling = t - known.get(best);
      return {
        id: c, tempC: t, status: target == null || cooling >= target - 1e-9 ? 'route' : 'short',
        coolest: {id: best, tempC: known.get(best), via, exit: exitIds.has(best), toward: exitOf.get(best)?.toward ?? null},
        coolingC: Math.round(cooling * 100) / 100,
      };
    });
    if (target == null) warnings.push('No warming target is set (params.climateWarmingC); any cooler reachable core or exit counts as a route.');
    const exitList = exits.map(e => ({id: e.id, tempC: e.tempC, toward: e.toward, linkedCores: adj.get(e.id).filter(w => !exitIds.has(w))}));
    for (const e of exitList) if (!e.linkedCores.length) warnings.push(`Exit ${e.id} links to no core after the plan; draw it on the spine where the spine leaves the landscape.`);
    return {
      ...base, status: 'ok', reasons: [], warnings: [...new Set(warnings)],
      checkStatus: result.status,
      target: target == null ? null : {warmingC: target, source: p.climateSource},
      temperatureSource: p.coreTempSource,
      cores,
      exits: exitList,
      flagged: cores.filter(e => ['none', 'short', 'unknown'].includes(e.status)).map(e => e.id),
      limitations: [
        'Routes follow links that hold after the plan, chained through linked cores; they do not predict whether species move.',
        "A route is judged by its end cores' temperatures; the corridor between them may cross warmer ground.",
        'Core temperatures are as supplied; one value per core does not capture the microclimate within it.',
        ...(exits.length ? ['An exit stands for the spine continuing into the next landscape at the temperature supplied; the route beyond it is checked in that landscape, not here.'] : []),
      ],
    };
  } catch (e) {
    return {...base, status: 'incomplete', reasons: [`Invalid input or check result: ${e.message}`], warnings};
  }
}

/**
 * Neighbors whose land would extend the committed spine (FR1-FR3).
 * @param {object} input - connectivity input with parcels (consent 'covered' or not).
 */
export function buildOutFrontier(input) {
  const base = begin(OUTLOOK_VERSION, input);
  const {errors, warnings, cores} = validateBase(input);
  if (!errors.length && !(input.parcels ?? []).length) errors.push('Parcels are required to find the next woodlots.');
  if (errors.length) return {...base, status: 'incomplete', reasons: errors, warnings};
  try {
    const parcels = input.parcels.slice().sort(byId);
    const marked = input.retained.filter(f => f.properties?.spine === true);
    const spineFeatures = marked.length ? marked : input.retained;
    if (!marked.length) warnings.push('No retained feature is marked spine: true, so all retained habitat counts as spine.');
    const spine = union(spineFeatures);
    const covered = parcels.filter(f => f.properties?.consent === 'covered');
    const coveredArea = covered.length ? union(covered) : null;
    let committedSpine = coveredArea ? intersect(spine, coveredArea) : null;
    if (committedSpine && areaM2(committedSpine) < 1) committedSpine = null;
    if (!covered.length) warnings.push('No parcel has covering consent yet: every parcel holding spine can be the first.');
    else if (!committedSpine) warnings.push('Parcels with covering consent hold no mapped spine yet: every parcel holding spine can extend it.');
    const links = linker(input, cores);
    const now = links.committed('now', coveredArea).pairs;
    const nowSet = new Set(now);
    const committedPieces = committedSpine ? parts(committedSpine) : [];
    const frontier = [], later = [];
    for (const f of parcels) {
      if (f.properties?.consent === 'covered') continue;
      const piece = intersect(spine, f);
      const spineM2 = areaM2(piece);
      if (spineM2 < 1) continue;
      const near = committedSpine ? buffer(piece, 1) : null;
      const adjoining = near ? committedPieces.filter(c => turf.booleanIntersects(near, c)) : [];
      if (committedSpine && !adjoining.length) { later.push(id(f)); continue; }
      const withIt = links.committed(`with:${id(f)}`, coveredArea ? union([coveredArea, f]) : turf.feature(f.geometry)).pairs;
      const entry = {parcel: id(f), spineM2: Math.round(spineM2), completesLinks: withIt.filter(k => !nowSet.has(k)).map(pairObj)};
      if (adjoining.length) {
        const from = centroid(union(adjoining)), at = centroid(piece);
        const bearing = ((Math.round(turf.bearing(turf.point(from), turf.point(at))) % 360) + 360) % 360;
        Object.assign(entry, {direction: compass(bearing), bearingDeg: bearing, distanceM: Math.round(turf.rhumbDistance(from, at, {units: 'meters'}))});
      }
      frontier.push(entry);
    }
    frontier.sort((a, b) => b.completesLinks.length - a.completesLinks.length || b.spineM2 - a.spineM2 || (a.parcel < b.parcel ? -1 : 1));
    const mappedM2 = areaM2(spine), committedM2 = areaM2(committedSpine);
    return {
      ...base, status: 'ok', reasons: [], warnings: [...new Set(warnings)],
      committed: {parcels: covered.map(id), spineM2: Math.round(committedM2), mappedSpineM2: Math.round(mappedM2), share: mappedM2 ? committedM2 / mappedM2 : 0, links: now.map(pairObj)},
      frontier,
      laterParcels: later,
      limitations: [
        'Links a parcel would complete assume it alone joins now, with the proposed treatment units applied.',
        'Direction is from the committed piece of spine the parcel adjoins to the spine inside the parcel.',
      ],
    };
  } catch (e) {
    return {...base, status: 'incomplete', reasons: [`Geometry engine error: ${e.message}`], warnings: [...new Set(warnings)]};
  }
}
