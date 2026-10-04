// Corridor connectivity check (DFM Build Spec, "Corridor connectivity engine").
//
// Question answered: which core areas stay linked by retained habitat at least
// `minWidthM` wide, before versus after a proposed set of treatment units?
//
// Invariants implemented here (tests in test/connectivity.test.mjs):
//   I1  A plan that disconnects two core areas linked today, narrows the only link
//       below the minimum width, or removes a core area, returns 'fail' and names
//       the responsible treatment units.
//   I2  A treatment unit overlapping retained habitat or a core area fails the
//       check unless it is a permitted light treatment: corridor_permitted, an
//       intensity from LIGHT_INTENSITIES and a recorded reason. Permitted units
//       keep habitat in place and are reported as warnings.
//   I3  Width is enforced after road surfaces and open water are removed; the
//       minimum width must carry a recorded source.
//   I4  Roads sever habitat; a crossing recorded as 'verified' or 'assumed' on a
//       specific road restores a link across that road only ('assumed' warns).
//   I5  Retained habitat is 'committed' only inside parcels whose consent covers
//       it; elsewhere it is 'proposed'.
//   I6  'old-growth-verified' requires an evidence reference; otherwise the core is
//       reported as 'old-growth-candidate'.
//   I7  Missing, invalid or wrong-type inputs return 'incomplete' with reasons,
//       never a partial pass.
//   I8  Results are deterministic; an input checksum and engine version are
//       recorded.
//   I10 Lengths are meters and areas square meters throughout.
//
// Method: a minimum-width corridor exists between two cores when a disk of
// diameter minWidthM can travel from one to the other inside the habitat:
// the cores (what remains of them) touch the same connected part of the habitat
// eroded by minWidthM / 2, within that radius.

import * as turf from './turf.mjs';
import {canonicalHashSync} from './hash.mjs';
import {fc, isPoly, isLine, id, union, intersect, difference, buffer, close, areaM2, meanWidthM, parts} from './geo.mjs';

export const ENGINE_VERSION = 'dfm-connectivity-0.1.0';
export const CORE_CLASSES = ['old-growth-candidate', 'old-growth-verified', 'riparian-core', 'reserve'];
export const CROSSING_STATUS = ['verified', 'assumed', 'none'];
/** Treatments that may be permitted inside corridors and cores. Anything else removes habitat. */
export const LIGHT_INTENSITIES = ['single-tree-selection', 'light-thinning', 'invasive-removal', 'restoration-planting'];
export const LIMITS = {features: 2000, treatments: 100, cores: 50, extentDegrees: 0.5};
/** Validate a connectivity input. Returns {errors, warnings, cores}. */
export function validateInput(input) {
  const errors = [], warnings = [];
  if (!input || typeof input !== 'object') return {errors: ['A connectivity input object is required.'], warnings, cores: []};
  // Structure first, so malformed layers are reported in plain words rather than failing later.
  for (const k of ['coreAreas', 'retained', 'roads', 'water', 'crossings', 'treatments', 'parcels']) {
    if (input[k] == null) continue;
    if (!Array.isArray(input[k])) { errors.push(`${k} must be an array of GeoJSON features.`); continue; }
    for (const [i, f] of input[k].entries())
      if (!f || typeof f !== 'object' || !f.geometry || !Array.isArray(f.geometry.coordinates)) errors.push(`${k}[${i}] must be a GeoJSON feature with geometry coordinates.`);
  }
  if (input.params != null && (typeof input.params !== 'object' || Array.isArray(input.params))) errors.push('params must be an object.');
  if (errors.length) return {errors, warnings, cores: []};
  const p = input.params ?? {};
  if (!Number.isFinite(p.minWidthM) || p.minWidthM <= 0 || p.minWidthM > 2000) errors.push('params.minWidthM must be a width in meters between 0 and 2,000.');
  if (typeof p.minWidthSource !== 'string' || !p.minWidthSource.trim()) errors.push('params.minWidthSource must record where the minimum width comes from.');
  if (p.roadWidthM != null && (!Number.isFinite(p.roadWidthM) || p.roadWidthM < 1 || p.roadWidthM > 100)) errors.push('params.roadWidthM must be between 1 and 100 m.');
  if (p.pinchFraction != null && (!Number.isFinite(p.pinchFraction) || p.pinchFraction < 0 || p.pinchFraction > 1)) errors.push('params.pinchFraction must be between 0 and 1.');

  const polygonsOnly = {coreAreas: 'core areas', retained: 'retained habitat', water: 'open water (buffer stream centerlines into polygons first)', parcels: 'parcels', treatments: 'treatment units'};
  for (const [k, label] of Object.entries(polygonsOnly))
    for (const [i, f] of (input[k] ?? []).entries()) if (!isPoly(f)) errors.push(`${k}[${i}] must be a polygon: ${label}.`);
  for (const [i, f] of (input.roads ?? []).entries()) {
    if (!isLine(f) && !isPoly(f)) errors.push(`roads[${i}] must be a centerline or a road-surface polygon.`);
    else if (f.properties?.width_m != null && !(f.properties.width_m >= 1 && f.properties.width_m <= 100)) errors.push(`roads[${i}] properties.width_m must be between 1 and 100 m.`);
    else if (isLine(f) && p.roadWidthM == null && f.properties?.width_m == null) errors.push(`roads[${i}] is a centerline: set params.roadWidthM or properties.width_m.`);
  }
  for (const [i, f] of (input.crossings ?? []).entries()) {
    if (f?.geometry?.type !== 'Point') errors.push(`crossings[${i}] must be a point.`);
    else if (!CROSSING_STATUS.includes(f.properties?.passage)) errors.push(`crossings[${i}] passage must be one of ${CROSSING_STATUS.join(', ')}.`);
  }

  const cores = [];
  for (const f of input.coreAreas ?? []) {
    if (!isPoly(f)) continue;
    if (!id(f) || id(f).includes('|')) { errors.push('Every core area needs properties.dfm_id without a "|" character.'); continue; }
    let coreClass = f.properties.core_class;
    if (!CORE_CLASSES.includes(coreClass)) { errors.push(`Core ${id(f)} has unknown core_class "${coreClass}".`); continue; }
    if (coreClass === 'old-growth-verified' && !f.properties.evidence_id) {
      warnings.push(`Core ${id(f)} is marked old-growth-verified without an evidence reference; reported as old-growth-candidate.`);
      coreClass = 'old-growth-candidate';
    }
    cores.push({id: id(f), coreClass, feature: f});
  }
  if (new Set(cores.map(c => c.id)).size !== cores.length) errors.push('Core area IDs must be unique.');
  if (cores.length < 2) errors.push('At least two core areas are needed to check connections.');
  if (cores.length > LIMITS.cores) errors.push(`Use at most ${LIMITS.cores} core areas.`);
  if (!(input.retained ?? []).some(isPoly)) errors.push('Retained habitat polygons are required.');

  const treatments = input.treatments ?? [];
  if (treatments.length > LIMITS.treatments) errors.push(`Use at most ${LIMITS.treatments} treatment units per check.`);
  if (treatments.some(t => isPoly(t) && !id(t))) errors.push('Every treatment unit needs properties.dfm_id.');
  if (new Set(treatments.map(id)).size !== treatments.length) errors.push('Treatment unit IDs must be unique.');

  const all = ['coreAreas', 'retained', 'roads', 'water', 'crossings', 'treatments', 'parcels'].flatMap(k => input[k] ?? []).filter(f => f?.geometry);
  if (all.length > LIMITS.features) errors.push(`Use at most ${LIMITS.features} features.`);
  let lonLatOk = true;
  for (const f of all) turf.coordEach(f, c => { if (!(Math.abs(c[0]) <= 180 && Math.abs(c[1]) <= 85)) lonLatOk = false; });
  if (!lonLatOk) errors.push('Coordinates must be WGS84 longitude/latitude (EPSG:4326); reproject projected data first.');
  else if (all.length) {
    const [w, s, e, n] = turf.bbox(fc(all));
    if (e - w > LIMITS.extentDegrees || n - s > LIMITS.extentDegrees) errors.push(`The landscape spans more than ${LIMITS.extentDegrees}° (about 50 km); this engine is for woodlot-scale extents.`);
  }
  for (const f of all) if (!turf.booleanValid(f)) { errors.push(`Feature ${id(f) || '(no id)'} has invalid geometry.`); break; }
  return {errors, warnings, cores};
}

/** Recorded light treatment that may stay inside corridors and cores (I2). */
export function isPermittedLight(f, warnings) {
  const pr = f.properties ?? {};
  if (pr.corridor_permitted !== true) return false;
  if (!LIGHT_INTENSITIES.includes(pr.intensity) || !pr.reason) {
    warnings.push(`Treatment ${id(f)} is marked corridor_permitted but needs an intensity from ${LIGHT_INTENSITIES.join(', ')} and a recorded reason; treated as not permitted.`);
    return false;
  }
  return true;
}

/** Split roads into single parts so a crossing applies only to the piece it sits on. */
export function roadParts(input) {
  const p = input.params, out = [];
  for (const f of input.roads ?? []) {
    const width = f.properties?.width_m ?? p.roadWidthM ?? null;
    if (f.geometry.type === 'LineString') out.push({line: turf.lineString(f.geometry.coordinates), width});
    else if (f.geometry.type === 'MultiLineString') for (const c of f.geometry.coordinates) out.push({line: turf.lineString(c), width});
    else if (f.geometry.type === 'Polygon') out.push({poly: turf.polygon(f.geometry.coordinates), width});
    else for (const c of f.geometry.coordinates) out.push({poly: turf.polygon(c), width});
  }
  for (const r of out) r.surface = r.line ? buffer(r.line, r.width / 2) : r.poly;
  return out;
}

/** Principal axis bearing (degrees) of a road surface near a point, from its vertices. */
function axisBearing(surface) {
  const pts = [];
  turf.coordEach(surface, c => pts.push(c));
  const mx = pts.reduce((a, c) => a + c[0], 0) / pts.length, my = pts.reduce((a, c) => a + c[1], 0) / pts.length;
  let sxx = 0, syy = 0, sxy = 0;
  for (const [x, y] of pts) { const dx = (x - mx) * Math.cos(my * Math.PI / 180), dy = y - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
  const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy); // radians from east
  return 90 - angle * 180 / Math.PI;
}

/**
 * Road surfaces with recorded crossings applied, plus bridge pieces across roads (I4).
 * A bridge is the road surface near the crossing where habitat lies straight across the
 * road on BOTH sides, so it cannot shift a corridor sideways or link a one-sided edge.
 */
function roadsAndBridges(input, habitatGross, warnings) {
  const p = input.params;
  const roads = roadParts(input);
  const bridges = [];
  for (const c of input.crossings ?? []) {
    if (c.properties.passage === 'none') continue;
    const probe = buffer(c, 1);
    const at = roads.filter(r => r.surface && turf.booleanIntersects(probe, r.surface));
    if (!at.length) { warnings.push(`Crossing ${id(c) || '(no id)'} is not on a road; ignored.`); continue; }
    let used = false;
    for (const r of at) {
      let local, bearing, width = r.width;
      if (r.line) {
        const near = turf.nearestPointOnLine(r.line, c, {units: 'meters'});
        const reach = p.minWidthM / 2 + width + 2;
        const along = near.properties.location;
        const slice = turf.lineSliceAlong(r.line, Math.max(0, along - reach), along + reach, {units: 'meters'});
        local = buffer(slice, width / 2);
        const coords = slice.geometry.coordinates;
        bearing = turf.bearing(coords[0], coords[coords.length - 1]);
      } else {
        local = intersect(r.surface, buffer(c, Math.max(100, p.minWidthM)));
        if (!local) continue;
        bearing = axisBearing(local);
        if (width == null) {
          // Width = local area / local length along the axis.
          const pts = []; turf.coordEach(local, q => pts.push(turf.rhumbDistance(c, q, {units: 'meters'}) * Math.cos((turf.rhumbBearing(c, q) - bearing) * Math.PI / 180)));
          const length = Math.max(...pts) - Math.min(...pts);
          width = Math.min(100, areaM2(local) / Math.max(length, 1));
        }
      }
      const disk = buffer(c, p.minWidthM / 2 + width + 1);
      const cut = intersect(local, disk);
      if (!cut) continue;
      const shift = width + 1;
      const sideA = turf.transformTranslate(habitatGross, shift, bearing + 90, {units: 'meters'});
      const sideB = turf.transformTranslate(habitatGross, shift, bearing - 90, {units: 'meters'});
      const bridge = intersect(intersect(cut, sideA), sideB);
      if (!bridge || areaM2(bridge) < 1) continue;
      r.surface = difference(r.surface, cut);
      bridges.push(bridge);
      used = true;
    }
    if (!used) { warnings.push(`Crossing ${id(c) || '(no id)'} is not where retained habitat lies on both sides of the road; ignored.`); continue; }
    if (c.properties.passage === 'assumed') warnings.push(`Crossing ${id(c) || '(no id)'} passage is assumed, not field-verified.`);
  }
  return {roads: union(roads.map(r => r.surface).filter(Boolean)), bridges};
}

/** Build effective habitat for one plan state. Internal: also used by projections on filtered layers. */
export function effectiveHabitat(input, treatments, warnings) {
  const habitatGross = close(union([...input.retained, ...input.coreAreas]));
  if (!habitatGross) return {habitat: null, removed: []};
  const {roads, bridges} = roadsAndBridges(input, habitatGross, warnings);
  // Closing runs once more in every case (roads are at least 1 m wide, so it cannot heal one).
  let habitat = close(union([difference(habitatGross, roads), ...bridges].filter(Boolean)));
  habitat = difference(habitat, union(input.water ?? []));
  const removed = treatments.filter(t => !isPermittedLight(t, warnings));
  if (removed.length) habitat = difference(habitat, union(removed));
  return {habitat, removed};
}

export const pairKey = (a, b) => (a < b ? `${a}|${b}` : `${b}|${a}`);
export const pairObj = k => { const [a, b] = k.split('|'); return {a, b}; };

/**
 * Pairs of cores linked by a corridor at least `widthM` wide. Uses what remains of each core.
 * `coreHabitat` (default `habitat`) is where a core's remaining part is measured: projections pass
 * the full habitat so an endpoint core need not itself qualify as old forest to be reached.
 */
export function linkedPairs(habitat, cores, widthM, coreHabitat = habitat) {
  const r = widthM / 2;
  const eroded = buffer(habitat, -r);
  const components = parts(eroded);
  const touch = cores.map(c => {
    // A core counts only while at least half of it remains as habitat.
    const remaining = intersect(c.feature, coreHabitat);
    if (!remaining || areaM2(remaining) < 0.5 * areaM2(c.feature)) return new Set();
    const reach = buffer(remaining, r + 0.5);
    return new Set(components.map((comp, i) => (turf.booleanIntersects(comp, reach) ? i : -1)).filter(i => i >= 0));
  });
  const pairs = [];
  for (let a = 0; a < cores.length; a++) for (let b = a + 1; b < cores.length; b++)
    if ([...touch[a]].some(i => touch[b].has(i))) pairs.push(pairKey(cores[a].id, cores[b].id));
  return {pairs: pairs.sort(), eroded};
}

/** Retained habitat split into committed (consent covers) and proposed area (I5). */
function consentStatus(input) {
  const retained = union(input.retained);
  const covered = union((input.parcels ?? []).filter(f => f.properties?.consent === 'covered'));
  const committed = intersect(retained, covered);
  const proposed = covered ? difference(retained, covered) : retained;
  return {committedM2: areaM2(committed), proposedM2: areaM2(proposed), committed, proposed};
}

/**
 * Check corridor connectivity for a proposed set of treatment units.
 * @param {object} input - {coreAreas, retained, roads?, water?, crossings?, treatments?, parcels?, params}
 *   params: {minWidthM, minWidthSource, roadWidthM?, pinchFraction? (default 0.1)}
 *   treatments are the PROPOSED units; the current state is checked without them.
 * Synchronous: safe to call inside a host's synchronous command handler.
 * @returns {object} check result (schema: dfm-schema/connectivity-result.schema.json)
 */
export function checkConnectivitySync(input) {
  let base;
  try { base = {engine: ENGINE_VERSION, inputChecksum: canonicalHashSync(input ?? null), parameters: input?.params ?? null}; }
  catch (e) { return {engine: ENGINE_VERSION, inputChecksum: null, parameters: null, status: 'incomplete', reasons: [`Invalid input: ${e.message}`], warnings: []}; }
  let errors, warnings, cores;
  // I7: malformed input (wrong types, missing coordinates) is 'incomplete', never a thrown error.
  try { ({errors, warnings, cores} = validateInput(input)); }
  catch (e) { return {...base, status: 'incomplete', reasons: [`Invalid input: ${e.message}`], warnings: []}; }
  if (errors.length) return {...base, status: 'incomplete', reasons: errors, warnings};
  try {
    const p = input.params, treatments = input.treatments ?? [];
    const before = effectiveHabitat(input, [], warnings);
    const after = effectiveHabitat(input, treatments, warnings);
    const linkedBefore = linkedPairs(before.habitat, cores, p.minWidthM);
    const linkedAfter = linkedPairs(after.habitat, cores, p.minWidthM);
    const lost = linkedBefore.pairs.filter(k => !linkedAfter.pairs.includes(k));

    let lostGeometry = null;
    if (lost.length) {
      try {
        // Drop slivers under 1 m wide left by differing buffer approximations; keep the real loss.
        const raw = difference(buffer(linkedBefore.eroded, p.minWidthM / 2), buffer(linkedAfter.eroded, p.minWidthM / 2));
        const kept = parts(raw).filter(part => meanWidthM(part) >= 1);
        lostGeometry = kept.length ? union(kept) : null;
      }
      catch { warnings.push('The map of lost corridor could not be drawn for this geometry; the link results are unaffected.'); }
    }
    // Each removed unit is tested alone once; a unit "causes" a lost link if removing it alone breaks that link.
    const single = lost.length ? new Map(after.removed.map(t => [t, new Set(linkedPairs(effectiveHabitat(input, [t], []).habitat, cores, p.minWidthM).pairs)])) : new Map();
    const breaks = lost.map(k => {
      const causes = after.removed.filter(t => !single.get(t).has(k)).map(id);
      const near = lostGeometry ?? before.habitat;
      return {...pairObj(k), causes, contributing: after.removed.filter(t => intersect(t, near)).map(id)};
    });

    // I2: unpermitted overlap with retained habitat or cores (net of roads and water) fails even when links hold.
    const overlaps = treatments.map(t => ({unit: id(t), overlapM2: areaM2(intersect(t, before.habitat)), permitted: !after.removed.includes(t)}))
      .filter(o => o.overlapM2 > 0.01);
    for (const o of overlaps) if (o.permitted) warnings.push(`Treatment ${o.unit} overlaps retained habitat or a core as a permitted light treatment (${o.overlapM2.toFixed(0)} m²).`);
    const violations = overlaps.filter(o => !o.permitted);

    // Pinch points: links that hold at the minimum width but not with the margin.
    const margin = p.pinchFraction ?? 0.1;
    const linkedMargin = margin > 0 ? linkedPairs(after.habitat, cores, p.minWidthM * (1 + margin)) : linkedAfter;
    const pinched = linkedAfter.pairs.filter(k => !linkedMargin.pairs.includes(k)).map(pairObj);

    for (const k of cores.flatMap((c, i) => cores.slice(i + 1).map(d => pairKey(c.id, d.id))))
      if (!linkedBefore.pairs.includes(k)) warnings.push(`Cores ${k.replace('|', ' and ')} are not linked at ${p.minWidthM} m in the current state.`);
    if (!linkedBefore.pairs.length) warnings.push('No core areas are linked in the current state, so this plan cannot break a link; review the corridor design before relying on a pass.');

    const consent = consentStatus(input);
    const status = lost.length || violations.length ? 'fail' : 'pass';
    return {
      ...base,
      status,
      reasons: [
        ...breaks.map(b => `Link ${b.a}–${b.b} is lost ${b.causes.length ? `(caused by ${b.causes.join(', ')})` : `(by the combined effect of ${b.contributing.join(', ') || 'the plan'})`}.`),
        ...violations.map(v => `Treatment ${v.unit} overlaps retained habitat or a core by ${v.overlapM2.toFixed(0)} m² without a recorded light-treatment permission.`),
      ],
      warnings: [...new Set(warnings)],
      cores: cores.map(c => ({id: c.id, coreClass: c.coreClass, areaM2: areaM2(c.feature), remainingM2: areaM2(intersect(c.feature, after.habitat))})),
      linkedBefore: linkedBefore.pairs.map(pairObj),
      linkedAfter: linkedAfter.pairs.map(pairObj),
      lostLinks: breaks,
      overlaps,
      pinchedLinks: pinched,
      consent: {committedM2: consent.committedM2, proposedM2: consent.proposedM2},
      areas: {habitatBeforeM2: areaM2(before.habitat), habitatAfterM2: areaM2(after.habitat)},
      geometry: fc([
        lostGeometry && {...lostGeometry, properties: {dfm_layer: 'connectivity-loss', name: 'Functional corridor lost under the plan'}},
        consent.proposed && {...consent.proposed, properties: {dfm_layer: 'corridor-proposed', name: 'Retained habitat without covering consent'}},
        consent.committed && {...consent.committed, properties: {dfm_layer: 'corridor-committed', name: 'Retained habitat with covering consent'}},
      ]),
      limitations: [
        'Structural connectivity at a minimum width only; not species movement, genetics or verified old-growth condition.',
        'Widths are horizontal, on a spherical Earth (within about 0.5%), after road surfaces and open water are removed; slope and edge effects are not modeled.',
        'Crossing passage is as recorded; assumed crossings need field verification.',
      ],
    };
  } catch (e) {
    return {...base, status: 'incomplete', reasons: [`Geometry engine error: ${e.message}`], warnings: [...new Set(warnings)]};
  }
}

/** Promise-returning form of checkConnectivitySync. */
export const checkConnectivity = async input => checkConnectivitySync(input);
