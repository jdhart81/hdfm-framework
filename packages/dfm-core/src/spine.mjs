// The old-growth spine (DFM Build Spec, "Old-growth spine").
//
// The spine is the dendritic network of retained forest that follows a landscape's natural
// corridors: streams set its branches and it is widest along the largest rivers (a river network
// is already a terrain-routed tree); links over ridges, along dry valleys and across saddles
// close loops. Loops can give a second route around a single disturbance; this module finds where
// one disturbance would still cut a link. Two functions:
//
//   deriveSpine(input)   draft retained-habitat polygons from stream and link lines, for a steward
//                        to review and save as the retained layer;
//   spineNetwork(input)  test that draft as a network: which cores it links (by the connectivity
//                        check's own rules), its loops, and its single points of failure.
//
// Invariants (tests in test/spine.test.mjs):
//   SP1 Every derived section is at least params.minWidthM wide. Widths come from
//       params.spineWidthByOrderM (meters per stream order, each at least the minimum, with
//       params.spineWidthSource) or, when no table is recorded, the minimum width plus the pinch
//       margin, rounded up past the next 5 m, recorded as a default. A width under the minimum plus
//       the pinch margin warns: the check flags such corridors as pinch points.
//   SP2 Width never decreases with stream order: a larger river never gets a narrower corridor.
//   SP3 Streams need an integer stream_order from 1 to 12 and a dfm_id; links need a kind (ridge,
//       valley or saddle) and a dfm_id; every line is at least 1 m long; width-table keys are whole
//       numbers. Anything else returns 'incomplete' with reasons, never a thrown error (I7).
//   SP4 Each derived section records its origin, source line, width and width source, and is
//       retained habitat only: derivation never assigns a core class (I6). A derived corridor that
//       overlaps mapped open water warns, for every kind of line.
//   SP5 Links come only from the connectivity check's rules applied to the derived corridors and
//       the core areas: roads sever unless a crossing on that road part carries the corridor (I4),
//       open water is removed, and the minimum width holds (I3). The line graph never adds a link.
//   SP6 Single points of failure are tested on that same habitat. A disturbance is a disk
//       params.disturbanceWidthM wide (default: the widest corridor plus 2 m), centered anywhere;
//       it may overlap a core but does not remove core habitat. Centers are tested on a grid with
//       an enlargement that covers every position (src/robust.mjs), so a pair reported robust
//       cannot be separated by any such disturbance under the check's rules. Each place where a
//       tested cut separates linked cores is reported, with the pairs it separates, and re-tested
//       exactly at nominal width (verified or not).
//   SP7 rho2 = robust pairs / all core pairs (the HDFM paper's 2-edge connectivity, over cores,
//       tested on the habitat); pFail1 = share of sections whose corridor some separating
//       disturbance overlaps; exposedShare = that length / total length. Pairs the raster cannot
//       resolve are never reported robust. Robustness is reported only from a completed search
//       (null when cuts are skipped or the cut budget runs out), and exposure is null when the
//       network has no sections to measure.
//   SP8 Loops are counted on the line graph of sections between junctions; routes that close only
//       through a core area are counted separately and do not make a network non-dendritic.
//   I8  Deterministic; an input checksum and engine version are recorded.

import * as turf from './turf.mjs';
import {canonicalHashSync} from './hash.mjs';
import {fc, isPoly, isLine, id, buffer, intersect, areaM2, lineParts} from './geo.mjs';
import {CROSSING_STATUS, LIMITS, effectiveHabitat, linkedPairs, pairObj} from './connectivity.mjs';
import {disturbanceModel, searchDisturbances, zones, exactCut} from './robust.mjs';

export const SPINE_VERSION = 'dfm-spine-0.1.0';
/** Kinds of non-stream links in the spine. */
export const SPINE_LINK_KINDS = ['ridge', 'valley', 'saddle'];
const MAX_ORDER = 12, DEFAULT_SNAP_M = 5, MAX_SNAP_M = 10, MIN_LINE_M = 1;
const ATTACH_STEP_M = 5; // spacing of samples used to find where a line enters a core's reach

/** Width table lookup: the entry for the largest order key <= order. */
function widthFor(table, order) {
  let best = null;
  for (const [k, w] of table) if (k <= order) best = w;
  return best;
}

const featureList = (input, k, errors) => {
  const v = input[k];
  if (v == null) return [];
  if (!Array.isArray(v)) { errors.push(`${k} must be an array of GeoJSON features.`); return []; }
  for (const [i, f] of v.entries())
    if (!f || typeof f !== 'object' || !f.geometry || !Array.isArray(f.geometry.coordinates)) { errors.push(`${k}[${i}] must be a GeoJSON feature with geometry coordinates.`); return []; }
  return v;
};

/** Shared validation for spine inputs. */
function validateSpine(input, {needCores = false} = {}) {
  const errors = [], warnings = [];
  if (!input || typeof input !== 'object') return {errors: ['A spine input object is required.'], warnings};
  const layers = {};
  for (const k of ['streams', 'connectors', 'coreAreas', 'roads', 'crossings', 'water']) layers[k] = featureList(input, k, errors);
  if (input.params != null && (typeof input.params !== 'object' || Array.isArray(input.params))) errors.push('params must be an object.');
  if (errors.length) return {errors, warnings};
  const p = input.params ?? {};
  if (!Number.isFinite(p.minWidthM) || p.minWidthM <= 0 || p.minWidthM > 2000) errors.push('params.minWidthM must be a width in meters between 0 and 2,000.');
  if (typeof p.minWidthSource !== 'string' || !p.minWidthSource.trim()) errors.push('params.minWidthSource must record where the minimum width comes from.');
  if (p.pinchFraction != null && (!Number.isFinite(p.pinchFraction) || p.pinchFraction < 0 || p.pinchFraction > 1)) errors.push('params.pinchFraction must be between 0 and 1.');
  if (p.junctionSnapM != null && !(p.junctionSnapM >= 0.1 && p.junctionSnapM <= MAX_SNAP_M)) errors.push(`params.junctionSnapM must be between 0.1 and ${MAX_SNAP_M} m.`);
  if (p.disturbanceWidthM != null && !(Number.isFinite(p.disturbanceWidthM) && p.disturbanceWidthM >= 1 && p.disturbanceWidthM <= 2000)) errors.push('params.disturbanceWidthM must be a width in meters between 1 and 2,000.');
  const mw = p.minWidthM, pinch = p.pinchFraction ?? 0.1;

  const streams = layers.streams, links = layers.connectors;
  if (!streams.length && !links.length) errors.push('Spine lines are required: streams (with stream_order) and optional connectors over ridges, along valleys or across saddles.');
  for (const [i, f] of streams.entries()) {
    if (!isLine(f)) { errors.push(`streams[${i}] must be a line.`); continue; }
    if (!id(f) || id(f).includes('|')) errors.push(`streams[${i}] needs properties.dfm_id without a "|" character.`);
    const o = f.properties?.stream_order;
    if (!Number.isInteger(o) || o < 1 || o > MAX_ORDER) errors.push(`Stream ${id(f) || `[${i}]`} needs an integer stream_order from 1 to ${MAX_ORDER} (Strahler order).`);
  }
  for (const [i, f] of links.entries()) {
    if (!isLine(f)) { errors.push(`connectors[${i}] must be a line.`); continue; }
    if (!id(f) || id(f).includes('|')) errors.push(`connectors[${i}] needs properties.dfm_id without a "|" character.`);
    if (!SPINE_LINK_KINDS.includes(f.properties?.kind)) errors.push(`Connector ${id(f) || `[${i}]`} needs a kind: ${SPINE_LINK_KINDS.join(', ')}.`);
  }
  for (const f of [...streams, ...links].filter(isLine)) {
    let shortest = 0;
    try { shortest = Math.min(...lineParts(f).map(c => (c.length >= 2 ? turf.length(turf.lineString(c), {units: 'meters'}) : 0))); } catch { shortest = 0; }
    if (!(shortest >= MIN_LINE_M)) errors.push(`Line ${id(f) || '(no id)'} has a part shorter than ${MIN_LINE_M} m; remove it or fix its coordinates.`);
  }
  const ids = [...streams, ...links].map(id).filter(Boolean);
  if (new Set(ids).size !== ids.length) errors.push('Stream and connector IDs must be unique.');

  // SP1 and SP2: the width table.
  let table = null, widthSource = null;
  const marginWidth = Number.isFinite(mw) ? mw * (1 + pinch) : null;
  const defaultWidth = Number.isFinite(mw) ? (Math.floor(marginWidth / 5) + 1) * 5 : null;
  if (p.spineWidthByOrderM != null) {
    const raw = p.spineWidthByOrderM;
    if (typeof raw !== 'object' || Array.isArray(raw)) errors.push('params.spineWidthByOrderM must map stream orders ("1", "2", ...) to widths in meters.');
    else {
      table = [];
      for (const [key, w] of Object.entries(raw)) {
        if (!/^(?:[1-9]|1[0-2])$/.test(key)) { errors.push(`params.spineWidthByOrderM key "${key}" must be a whole stream order from 1 to ${MAX_ORDER}.`); continue; }
        if (!Number.isFinite(w) || w > 2000) errors.push(`params.spineWidthByOrderM["${key}"] must be a width in meters up to 2,000.`);
        else if (Number.isFinite(mw) && w < mw) errors.push(`params.spineWidthByOrderM["${key}"] is ${w} m, narrower than the ${mw} m minimum width; every section must be at least the minimum.`);
        else if (Number.isFinite(mw) && w === mw) warnings.push(`params.spineWidthByOrderM["${key}"] is exactly the ${mw} m minimum: the check finds no link through a corridor exactly that wide. Use at least the minimum plus the ${Math.round(pinch * 100)}% pinch margin.`);
        else if (Number.isFinite(marginWidth) && w <= marginWidth) warnings.push(`params.spineWidthByOrderM["${key}"] is ${w} m, within the ${Math.round(pinch * 100)}% pinch margin of the ${mw} m minimum: the check will flag these corridors as pinch points.`);
        table.push([Number(key), w]);
      }
      table.sort((a, b) => a[0] - b[0]);
      for (let i = 1; i < table.length; i++) if (table[i][1] < table[i - 1][1]) errors.push(`Spine widths must not decrease with stream order: order ${table[i][0]} (${table[i][1]} m) is narrower than order ${table[i - 1][0]} (${table[i - 1][1]} m).`);
      if (typeof p.spineWidthSource !== 'string' || !p.spineWidthSource.trim()) errors.push('params.spineWidthSource must record where the spine widths come from.');
      widthSource = p.spineWidthSource;
      for (const f of streams) {
        const o = f.properties?.stream_order;
        if (Number.isInteger(o) && widthFor(table, o) == null) errors.push(`Stream ${id(f)} is order ${o}, but params.spineWidthByOrderM has no width for order ${o} or below.`);
      }
    }
  } else if (Number.isFinite(defaultWidth)) {
    widthSource = `Default: the ${mw} m minimum width plus the ${Math.round(pinch * 100)}% pinch margin, rounded up to ${defaultWidth} m. No width table is recorded.`;
    if (streams.length) warnings.push(`No spine width table is recorded (params.spineWidthByOrderM); every stream section uses ${defaultWidth} m. Record widths by stream order with a source.`);
  }
  let linkWidth = defaultWidth, linkSource = `Default: the ${mw} m minimum width plus the ${Math.round(pinch * 100)}% pinch margin, rounded up to ${defaultWidth} m.`;
  if (p.connectorWidthM != null) {
    if (!Number.isFinite(p.connectorWidthM) || p.connectorWidthM > 2000 || (Number.isFinite(mw) && p.connectorWidthM < mw)) errors.push('params.connectorWidthM must be a width in meters, at least the minimum width.');
    else if (Number.isFinite(mw) && p.connectorWidthM === mw) warnings.push(`params.connectorWidthM is exactly the ${mw} m minimum: the check finds no link through a corridor exactly that wide.`);
    else if (Number.isFinite(marginWidth) && p.connectorWidthM <= marginWidth) warnings.push(`params.connectorWidthM is ${p.connectorWidthM} m, within the pinch margin of the minimum: the check will flag these corridors as pinch points.`);
    if (typeof p.connectorWidthSource !== 'string' || !p.connectorWidthSource.trim()) errors.push('params.connectorWidthSource must record where the connector width comes from.');
    linkWidth = p.connectorWidthM; linkSource = p.connectorWidthSource;
  }

  for (const [i, f] of layers.water.entries()) if (!isPoly(f)) errors.push(`water[${i}] must be a polygon (buffer stream centerlines into polygons first).`);
  if (p.roadWidthM != null && (!Number.isFinite(p.roadWidthM) || p.roadWidthM < 1 || p.roadWidthM > 100)) errors.push('params.roadWidthM must be between 1 and 100 m.');
  for (const [i, f] of layers.roads.entries()) {
    if (!isLine(f) && !isPoly(f)) errors.push(`roads[${i}] must be a centerline or a road-surface polygon.`);
    else if (f.properties?.width_m != null && !(f.properties.width_m >= 1 && f.properties.width_m <= 100)) errors.push(`roads[${i}] properties.width_m must be between 1 and 100 m.`);
    else if (isLine(f) && p.roadWidthM == null && f.properties?.width_m == null) errors.push(`roads[${i}] is a centerline: set params.roadWidthM or properties.width_m.`);
  }
  for (const [i, f] of layers.crossings.entries())
    if (f.geometry.type !== 'Point' || !CROSSING_STATUS.includes(f.properties?.passage)) errors.push(`crossings[${i}] must be a point with passage ${CROSSING_STATUS.join(', ')}.`);

  if (needCores) {
    const cores = layers.coreAreas;
    for (const [i, f] of cores.entries()) {
      if (!isPoly(f)) errors.push(`coreAreas[${i}] must be a polygon.`);
      else if (!id(f) || id(f).includes('|')) errors.push('Every core area needs properties.dfm_id without a "|" character.');
    }
    if (new Set(cores.map(id)).size !== cores.length) errors.push('Core area IDs must be unique.');
    if (cores.length < 2) errors.push('At least two core areas are needed to analyze the network.');
    if (cores.length > LIMITS.cores) errors.push(`Use at most ${LIMITS.cores} core areas.`);
  }

  if (!errors.length) {
    const all = Object.values(layers).flat().filter(f => f?.geometry);
    if (all.length > LIMITS.features) errors.push(`Use at most ${LIMITS.features} features.`);
    let lonLatOk = true;
    for (const f of all) turf.coordEach(f, c => { if (!(Math.abs(c[0]) <= 180 && Math.abs(c[1]) <= 85)) lonLatOk = false; });
    if (!lonLatOk) errors.push('Coordinates must be WGS84 longitude/latitude (EPSG:4326); reproject projected data first.');
    else if (all.length) {
      const [w, s, e, n] = turf.bbox(fc(all));
      if (e - w > LIMITS.extentDegrees || n - s > LIMITS.extentDegrees) errors.push(`The landscape spans more than ${LIMITS.extentDegrees}° (about 50 km); this engine is for woodlot and watershed-scale extents.`);
    }
    for (const f of all) if (!turf.booleanValid(f)) { errors.push(`Feature ${id(f) || '(no id)'} has invalid geometry.`); break; }
  }
  return {errors, warnings, table, defaultWidth, widthSource, linkWidth, linkSource};
}

/** Width and its source for one stream or connector feature. */
function sectionWidth(f, kind, v) {
  if (kind === 'stream') return {width: v.table ? widthFor(v.table, f.properties.stream_order) : v.defaultWidth, source: v.widthSource};
  return {width: v.linkWidth, source: v.linkSource};
}

const byId = (a, b) => (id(a) < id(b) ? -1 : id(a) > id(b) ? 1 : 0);
const sortedLines = input => [...(input.streams ?? []).slice().sort(byId).map(f => [f, 'stream']), ...(input.connectors ?? []).slice().sort(byId).map(f => [f, f.properties.kind])];

/** Derived corridors (inputs already validated): features in canonical order plus warnings. */
function derive(input, v) {
  const warnings = [], features = [];
  for (const [f, kind] of sortedLines(input)) {
    const {width, source} = sectionWidth(f, kind, v);
    const poly = buffer(turf.feature(f.geometry), width / 2);
    if (!poly) { warnings.push(`Line ${id(f)} produced no corridor.`); continue; }
    features.push({type: 'Feature', geometry: poly.geometry, properties: {
      dfm_id: `spine-${id(f)}`, spine: true, origin: kind, source_id: id(f),
      ...(kind === 'stream' ? {stream_order: f.properties.stream_order} : {}),
      width_m: width, width_source: source,
    }});
    for (const w of input.water ?? []) {
      const overlap = areaM2(intersect(poly, w));
      if (overlap >= 1) warnings.push(`The corridor of ${kind === 'stream' ? 'stream' : 'link'} ${id(f)} overlaps mapped open water${id(w) ? ` ${id(w)}` : ''} by ${Math.round(overlap)} m². The check removes open water, so make sure the forest left on at least one side is the minimum width.`);
    }
  }
  return {features, warnings};
}

/**
 * Draft spine sections from stream and connector lines (SP1-SP4).
 * @param {object} input - {streams, connectors?, water?, params: {minWidthM, minWidthSource,
 *   spineWidthByOrderM?, spineWidthSource?, connectorWidthM?, connectorWidthSource?, pinchFraction?}}
 * @returns {object} {engine, inputChecksum, status: 'ok'|'incomplete', reasons, warnings, features}
 *   features are retained-habitat polygons with properties {dfm_id, spine: true, origin,
 *   source_id, stream_order?, width_m, width_source}. Review them before saving as retained habitat.
 */
export function deriveSpine(input) {
  let base;
  try { base = {engine: SPINE_VERSION, inputChecksum: canonicalHashSync(input ?? null)}; }
  catch (e) { return {engine: SPINE_VERSION, inputChecksum: null, status: 'incomplete', reasons: [`Invalid input: ${e.message}`], warnings: [], features: []}; }
  try {
    const v = validateSpine(input);
    if (v.errors.length) return {...base, status: 'incomplete', reasons: v.errors, warnings: v.warnings, features: []};
    const d = derive(input, v);
    return {...base, status: 'ok', reasons: [], warnings: [...new Set([...v.warnings, ...d.warnings])], features: d.features};
  } catch (e) {
    return {...base, status: 'incomplete', reasons: [`Geometry engine error: ${e.message}`], warnings: [], features: []};
  }
}

// ---------------------------------------------------------------------------
// Network analysis

/** Local metric frame (equirectangular around the data's center): lon/lat to meters and back. */
function frame(features) {
  const [w, s, e, n] = turf.bbox(fc(features));
  const lon0 = (w + e) / 2, lat0 = (s + n) / 2, R = 6371008.8, k = Math.cos(lat0 * Math.PI / 180), d = Math.PI / 180;
  return {toXY: c => [(c[0] - lon0) * d * R * k, (c[1] - lat0) * d * R], fromXY: ([x, y]) => [lon0 + x / (d * R * k), lat0 + y / (d * R)]};
}

/** Union-find with path halving. */
function unionFind(n) {
  const parent = Array.from({length: n}, (_, i) => i);
  const find = i => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  return {find, join: (a, b) => { const ra = find(a), rb = find(b); if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb); }};
}

/** Cyclomatic number E - V + C of a graph given by its edges (only vertices on an edge count). */
function cyclomatic(edges) {
  const verts = [...new Set(edges.flatMap(e => [e.u, e.v]))];
  const index = new Map(verts.map((x, i) => [x, i]));
  const uf = unionFind(verts.length);
  for (const e of edges) uf.join(index.get(e.u), index.get(e.v));
  return edges.length - verts.length + new Set(verts.map((_, i) => uf.find(i))).size;
}

const pointAlong = (line, at) => (at <= 0 ? line.geometry.coordinates[0] : turf.lineSliceAlong(line, 0, at, {units: 'meters'}).geometry.coordinates.at(-1));

/**
 * Analyze the spine as a network (SP5-SP8).
 * @param {object} input - {streams, connectors?, coreAreas, roads?, crossings?, water?, params:
 *   {minWidthM, minWidthSource, spineWidthByOrderM?, connectorWidthM?, roadWidthM?, junctionSnapM?, disturbanceWidthM?}}
 * @param {{cuts?: boolean, cutBudget?: number}} [options] - cuts: false skips the disturbance tests
 *   (fast; robustness is then reported as null, never as true). cutBudget (default 20,000) caps the
 *   number of disturbance positions tested; a search that needs more reports robustness as null.
 * @returns {object} {engine, inputChecksum, status, reasons, warnings, summary, coreLinks,
 *   singlePointsOfFailure, severed, sections, limitations}
 */
export function spineNetwork(input, {cuts = true, cutBudget = 20000} = {}) {
  let base;
  try { base = {engine: SPINE_VERSION, inputChecksum: canonicalHashSync(input ?? null)}; }
  catch (e) { return {engine: SPINE_VERSION, inputChecksum: null, status: 'incomplete', reasons: [`Invalid input: ${e.message}`], warnings: []}; }
  const warnings = [];
  try {
    const v = validateSpine(input, {needCores: true});
    warnings.push(...v.warnings);
    if (!v.errors.length && input.params.minWidthM < 5) v.errors.push('The network analysis needs params.minWidthM of at least 5 m; a narrower strip is not a forest corridor.');
    if (v.errors.length) return {...base, status: 'incomplete', reasons: v.errors, warnings};
    const p = input.params, snapM = p.junctionSnapM ?? DEFAULT_SNAP_M, mw = p.minWidthM;
    const lines = sortedLines(input);
    const cores = (input.coreAreas ?? []).slice().sort(byId);
    const coreIds = cores.map(id);
    const {toXY, fromXY} = frame([...lines.map(l => l[0]), ...cores]);

    // SP5: the habitat is the derived spine plus the cores, under the check's own rules.
    const derived = derive(input, v);
    warnings.push(...derived.warnings);
    const checkCores = cores.map(c => ({id: id(c), feature: c}));
    const {habitat} = effectiveHabitat({...input, retained: derived.features, coreAreas: cores, treatments: []}, [], warnings);
    const linked = new Set(linkedPairs(habitat, checkCores, mw).pairs);

    // The line graph: parts, junctions, sections.
    const partsList = [];
    for (const [f, kind] of lines) {
      const {width} = sectionWidth(f, kind, v);
      lineParts(f).forEach((coords, k) => {
        const line = turf.lineString(coords);
        partsList.push({sourceId: id(f), part: k, origin: kind, order: kind === 'stream' ? f.properties.stream_order : null, width, line, length: turf.length(line, {units: 'meters'}), cuts: []});
      });
    }
    const points = [];
    const addPoint = c => { points.push({xy: toXY(c), c}); return points.length - 1; };
    for (const pr of partsList) {
      const coords = pr.line.geometry.coordinates;
      pr.cuts.push({at: 0, point: addPoint(coords[0])}, {at: pr.length, point: addPoint(coords[coords.length - 1])});
      const b = turf.bbox(pr.line); pr.box = [toXY([b[0], b[1]]), toXY([b[2], b[3]])];
    }
    const near = (xy, box, pad) => xy[0] >= box[0][0] - pad && xy[0] <= box[1][0] + pad && xy[1] >= box[0][1] - pad && xy[1] <= box[1][1] + pad;
    const boxesMeet = (a, b) => a[1][0] >= b[0][0] && b[1][0] >= a[0][0] && a[1][1] >= b[0][1] && b[1][1] >= a[0][1];
    const loose = [], joined = new Set();
    for (const [i, pr] of partsList.entries()) {
      for (const end of [pr.cuts[0], pr.cuts[1]]) {
        const P = points[end.point];
        let attached = false, closest = Infinity, closestId = null;
        for (const [j, other] of partsList.entries()) {
          if (j === i || !near(P.xy, other.box, 10 * snapM)) continue;
          const np = turf.nearestPointOnLine(other.line, turf.point(P.c), {units: 'meters'});
          const d = np.properties.dist;
          if (d <= snapM) {
            attached = true; joined.add(`${Math.min(i, j)}:${Math.max(i, j)}`);
            other.cuts.push({at: Math.min(Math.max(np.properties.location, 0), other.length), point: addPoint(np.geometry.coordinates)});
          } else if (d < closest) { closest = d; closestId = other.sourceId; }
        }
        if (!attached && closest <= 10 * snapM) loose.push({source: pr.sourceId, other: closestId, gapM: closest});
      }
    }
    // Lines that cross mid-way are not joined; say so rather than assume either way.
    for (let i = 0; i < partsList.length; i++) for (let j = i + 1; j < partsList.length; j++) {
      const a = partsList[i], b = partsList[j];
      if (a.sourceId === b.sourceId || joined.has(`${i}:${j}`) || !boxesMeet(a.box, b.box)) continue;
      if (turf.booleanIntersects(a.line, b.line)) warnings.push(`Lines ${a.sourceId} and ${b.sourceId} cross without a junction; split them where they meet if the corridors join there.`);
    }

    // Cores attach to line ends and junctions inside their reach; where a line only passes
    // through the reach, at the middle of each stretch of the line that lies inside it.
    const attachments = [];
    for (const [ci, core] of cores.entries()) {
      const cbox = turf.bbox(core), cmin = toXY([cbox[0], cbox[1]]), cmax = toXY([cbox[2], cbox[3]]);
      const reachOf = new Map();
      for (const pr of partsList) {
        const pad = pr.width / 2 + 1;
        if (pr.box[1][0] + pad < cmin[0] || pr.box[0][0] - pad > cmax[0] || pr.box[1][1] + pad < cmin[1] || pr.box[0][1] - pad > cmax[1]) continue;
        if (!reachOf.has(pr.width)) reachOf.set(pr.width, buffer(core, pr.width / 2 + 0.5));
        const reach = reachOf.get(pr.width);
        if (!reach || !turf.booleanIntersects(pr.line, reach)) continue;
        const inside = pr.cuts.filter(cut => turf.booleanIntersects(turf.point(points[cut.point].c), reach));
        if (inside.length) { for (const cut of inside) attachments.push({core: ci, point: cut.point}); continue; }
        const steps = Math.max(2, Math.ceil(pr.length / ATTACH_STEP_M));
        let run = null;
        const flush = () => {
          if (!run) return;
          const at = (run[0] + run[1]) / 2;
          const point = addPoint(pointAlong(pr.line, at));
          pr.cuts.push({at, point}); attachments.push({core: ci, point}); run = null;
        };
        for (let s = 0; s <= steps; s++) {
          const at = (pr.length * s) / steps;
          if (turf.booleanIntersects(turf.point(pointAlong(pr.line, at)), reach)) run = run ? [run[0], at] : [at, at]; else flush();
        }
        flush();
      }
    }

    // Cluster points within snapM into nodes.
    const uf = unionFind(points.length), cell = new Map(), key = (x, y) => `${x},${y}`;
    points.forEach((pt, i) => {
      const gx = Math.floor(pt.xy[0] / snapM), gy = Math.floor(pt.xy[1] / snapM);
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++)
        for (const j of cell.get(key(gx + dx, gy + dy)) ?? [])
          if (Math.hypot(pt.xy[0] - points[j].xy[0], pt.xy[1] - points[j].xy[1]) <= snapM) uf.join(i, j);
      const k = key(gx, gy); if (!cell.has(k)) cell.set(k, []); cell.get(k).push(i);
    });
    const nodeOf = new Map(), nodePoint = []; let nNodes = 0;
    points.forEach((_, i) => { const r = uf.find(i); if (!nodeOf.has(r)) { nodeOf.set(r, nNodes++); nodePoint.push(points[r].c); } });
    const node = i => nodeOf.get(uf.find(i));

    const sections = [];
    for (const pr of partsList) {
      const cuts = pr.cuts.slice().sort((a, b) => a.at - b.at);
      let k = 0;
      for (let c = 1; c < cuts.length; c++) {
        const from = cuts[c - 1], to = cuts[c];
        if (to.at - from.at < 0.01 || (node(from.point) === node(to.point) && to.at - from.at <= snapM)) continue;
        const geom = turf.lineSliceAlong(pr.line, from.at, to.at, {units: 'meters'});
        sections.push({id: `${pr.sourceId}#${pr.part}.${k++}`, sourceId: pr.sourceId, origin: pr.origin, order: pr.order, widthM: pr.width, lengthM: to.at - from.at, u: node(from.point), v: node(to.point), geom});
      }
    }

    // Roads cut sections at each place a section crosses a road: every such place needs a
    // crossing recorded on that road within the corridor's reach of it (as I4). Reported for the
    // line graph; links come from SP5 regardless.
    const segX = (a, b, c, d) => {
      const r = [b[0] - a[0], b[1] - a[1]], q = [d[0] - c[0], d[1] - c[1]], den = r[0] * q[1] - r[1] * q[0];
      if (Math.abs(den) < 1e-12) return null;
      const t = ((c[0] - a[0]) * q[1] - (c[1] - a[1]) * q[0]) / den, u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / den;
      return t >= 0 && t <= 1 && u >= 0 && u <= 1 ? [a[0] + t * r[0], a[1] + t * r[1]] : null;
    };
    const roadsXY = [];
    for (const f of input.roads ?? []) {
      const width = f.properties?.width_m ?? p.roadWidthM;
      const surface = isLine(f) ? buffer(turf.feature(f.geometry), width / 2) : turf.feature(f.geometry);
      if (!surface) throw new Error(`Road ${id(f) || '(no id)'} has no surface; check its width.`);
      const carried = (input.crossings ?? []).filter(c => c.properties.passage !== 'none' && turf.booleanIntersects(buffer(c, 1), surface)).map(c => ({c, xy: toXY(c.geometry.coordinates)}));
      roadsXY.push({road: id(f) || '(no id)', width: width ?? 0, surface, line: isLine(f) ? lineParts(f).map(cs => cs.map(toXY)) : null, carried});
    }
    const severed = [];
    for (const s of sections) {
      const sxy = s.geom.geometry.coordinates.map(toXY);
      const by = new Set();
      for (const r of roadsXY) {
        if (!turf.booleanIntersects(s.geom, r.surface)) continue;
        // Places where the section crosses this road: exact for centerlines; sampled every 2 m for surfaces.
        const places = [];
        if (r.line) { for (const part of r.line) for (let i = 1; i < sxy.length; i++) for (let j = 1; j < part.length; j++) { const x = segX(sxy[i - 1], sxy[i], part[j - 1], part[j]); if (x) places.push(x); } }
        if (!places.length) {
          const steps = Math.max(2, Math.ceil(s.lengthM / 2));
          let run = null;
          for (let k = 0; k <= steps; k++) {
            const at = (s.lengthM * k) / steps, inside = turf.booleanIntersects(turf.point(pointAlong(s.geom, at)), r.surface);
            if (inside) run = run ? [run[0], at] : [at, at];
            if ((!inside || k === steps) && run) { places.push(toXY(pointAlong(s.geom, (run[0] + run[1]) / 2))); run = null; }
          }
        }
        const reach = s.widthM / 2 + r.width + 1;
        for (const x of places) {
          const carriers = r.carried.filter(k => Math.hypot(k.xy[0] - x[0], k.xy[1] - x[1]) <= reach);
          if (!carriers.length) by.add(r.road);
          else if (carriers.every(k => k.c.properties.passage === 'assumed')) warnings.push(`Crossing ${carriers.map(k => id(k.c) || '(no id)').join(', ')} on road ${r.road} carries spine line ${s.sourceId}; its passage is assumed, not field-verified.`);
        }
      }
      if (by.size) { s.severedBy = [...by].sort(); severed.push({section: s.id, sourceId: s.sourceId, roads: s.severedBy}); }
    }
    const live = sections.filter(s => !s.severedBy);

    // SP8: loops on the line graph; routes closing only through cores are counted separately.
    const coreVertex = ci => nNodes + ci;
    const attachKeys = [...new Set(attachments.map(a => `${a.core}:${node(a.point)}`))].sort();
    const sectionEdges = live.map(s => ({u: s.u, v: s.v}));
    const loops = cyclomatic(sectionEdges);
    const loopsThroughCores = cyclomatic([...sectionEdges, ...attachKeys.map(k => { const [c, n] = k.split(':').map(Number); return {u: coreVertex(c), v: n}; })]) - loops;

    // SP6: one disturbance, centered anywhere, tested on the habitat (src/robust.mjs).
    const widths = derived.features.map(f => f.properties.width_m);
    const widest = widths.length ? Math.max(...widths) : null, narrowest = widths.length ? Math.min(...widths) : null;
    const disturbanceWidthM = p.disturbanceWidthM ?? (widest != null ? widest + 2 : null), rho = disturbanceWidthM / 2;
    // Raster spacing: fine enough for the narrowest corridor, and at most r / 2 so no gap in E can be stepped over.
    const resolutionM = narrowest != null ? Math.min(2.5, Math.max(0.5, (narrowest - mw) / 6), mw / 4) : null;
    const failures = [], separated = new Set(), untested = new Set(), failingCenters = [];
    let cutsTested = 0, fineSpacing = null, testedWidths = null, robustnessTested = false;
    if (cuts && linked.size) {
      if (!Number.isFinite(disturbanceWidthM) || !Number.isFinite(resolutionM)) warnings.push('No corridor could be derived to test, so robustness is not reported.');
      else {
        const model = disturbanceModel({habitat, cores: checkCores, minWidthM: mw, resolutionM, toXY});
        for (const k of linked) if (!model?.baseline.has(k)) { untested.add(k); separated.add(k); }
        if (untested.size) warnings.push(`${[...untested].join(', ')} ${untested.size === 1 ? 'is' : 'are'} linked by the check through habitat too narrow for the ${resolutionM.toFixed(1)} m disturbance raster, so ${untested.size === 1 ? 'it is' : 'they are'} not reported as robust.`);
        // The local frame stretches east-west distances away from its center latitude; inflate for the worst case.
        const [, south, , north] = turf.bbox(fc([...lines.map(l => l[0]), ...cores]));
        const lat0 = (south + north) / 2, latFar = Math.max(Math.abs(south), Math.abs(north));
        const scale = Math.max(1, Math.cos(lat0 * Math.PI / 180) / Math.cos(latFar * Math.PI / 180));
        const search = model ? searchDisturbances(model, {rhoM: rho, minWidthM: mw, pairs: linked, scale, budget: cutBudget}) : null;
        if (search?.exhausted) {
          cutsTested = search.tested;
          warnings.push(`The disturbance search needs more than ${cutBudget} cuts at this size, so robustness is not reported. Analyze one watershed at a time, or raise the cut budget.`);
        } else if (search) {
          robustnessTested = search.tested > 0 || !model.baseline.size;
          if (!robustnessTested) warnings.push('No disturbance position could be tested, so robustness is not reported.');
          cutsTested = search.tested; fineSpacing = search.fineSpacing;
          testedWidths = {coarseM: Math.round(2 * search.testedRadius.coarse), fineM: Math.round(2 * search.testedRadius.fine)};
          for (const members of zones(search.failures)) {
            const lost = new Set(members.flatMap(m => m.lost));
            // Confirm exactly at nominal width: the cuts nearest the zone's middle first, then a few
            // offsets around the best ones, since a tight separating spot can lie between grid points.
            const mx = members.reduce((t, m) => t + m.x, 0) / members.length, my = members.reduce((t, m) => t + m.y, 0) / members.length;
            const order = members.slice().sort((a, b) => b.lost.length - a.lost.length || Math.hypot(a.x - mx, a.y - my) - Math.hypot(b.x - mx, b.y - my) || a.fj - b.fj || a.fi - b.fi);
            const tries = order.slice(0, 4).map(m => ({x: m.x, y: m.y, lost: m.lost}));
            const h = fineSpacing / 2;
            for (const m of order.slice(0, 2)) for (const [dx, dy] of [[h, 0], [-h, 0], [0, h], [0, -h]]) tries.push({x: m.x + dx, y: m.y + dy, lost: m.lost});
            let rep = order[0], verified = false;
            for (const cand of tries) {
              cutsTested++;
              const exactAfter = exactCut({habitat, cores: checkCores, minWidthM: mw, center: fromXY([cand.x, cand.y]), rhoM: rho});
              if (cand.lost.some(k => !exactAfter.has(k))) { rep = cand; verified = true; break; }
            }
            const location = fromXY([rep.x, rep.y]);
            const at = turf.point(location);
            // Source lines whose corridor the disturbance reaches (whole lines, so a line shorter
            // than the junction distance, which has no section, is still named).
            const nearLines = [...new Set(partsList.filter(x => turf.nearestPointOnLine(x.line, at, {units: 'meters'}).properties.dist <= rho + x.width / 2).map(x => x.sourceId))].sort();
            let extent = 0;
            for (const a of members) for (const b of members) extent = Math.max(extent, Math.hypot(a.x - b.x, a.y - b.y));
            lost.forEach(k => separated.add(k));
            failures.push({kind: 'zone', location: location.map(v => Math.round(v * 1e7) / 1e7), separates: [...lost].sort().map(pairObj), verified, nearLines, extentM: Math.round(extent + fineSpacing)});
          }
          failingCenters.push(...search.failures.map(f => [f.x, f.y]));
        }
      }
    }
    failures.sort((x, y) => y.separates.length - x.separates.length || Number(y.verified) - Number(x.verified) || x.location[1] - y.location[1] || x.location[0] - y.location[0]);
    // Exposure: spine length whose corridor a separating disturbance overlaps (its center within
    // the disturbance radius plus the corridor half-width, plus the fine grid's half-diagonal).
    const sectionExposed = new Map();
    if (failingCenters.length) {
      const reachMax = rho + widest / 2 + fineSpacing;
      const bins = new Map(), bin = (x, y) => `${Math.floor(x / reachMax)},${Math.floor(y / reachMax)}`;
      for (const c of failingCenters) { const k = bin(c[0], c[1]); if (!bins.has(k)) bins.set(k, []); bins.get(k).push(c); }
      // All sections count: links come from the habitat, where a section the line graph marks as
      // severed by a road may still carry habitat (for example through a core).
      for (const x of sections) {
        const reach = rho + x.widthM / 2 + (fineSpacing * Math.SQRT2) / 2;
        const n = Math.max(2, Math.ceil(x.lengthM / 5));
        let hit = 0;
        for (let i = 0; i < n; i++) {
          const [px, py] = toXY(pointAlong(x.geom, ((i + 0.5) * x.lengthM) / n));
          const bx = Math.floor(px / reachMax), by = Math.floor(py / reachMax);
          let exposedHere = false;
          for (let dx = -1; dx <= 1 && !exposedHere; dx++) for (let dy = -1; dy <= 1 && !exposedHere; dy++)
            for (const c of bins.get(`${bx + dx},${by + dy}`) ?? []) if (Math.hypot(c[0] - px, c[1] - py) <= reach) { exposedHere = true; break; }
          if (exposedHere) hit++;
        }
        if (hit) sectionExposed.set(x.id, (hit / n) * x.lengthM);
      }
    }
    // Robustness is reported only from a completed search (or when nothing is linked, trivially).
    const tested = cuts && (robustnessTested || !linked.size);
    // Exposure is a share of sections; with none (every line shorter than the junction distance)
    // it cannot be measured, so it is unknown rather than zero.
    const measured = tested && sections.length > 0;
    const totalPairs = cores.length * (cores.length - 1) / 2;
    const robust = [...linked].filter(k => !separated.has(k)).length;
    const totalLength = sections.reduce((t, s) => t + s.lengthM, 0);
    const exposed = [...sectionExposed.values()].reduce((t, v) => t + v, 0);
    if (!sections.length) warnings.push(`Every spine line is shorter than the ${snapM} m junction distance, so the network has no sections; exposure is not reported. Check that the lines are drawn at full length.`);
    for (const g of loose) warnings.push(`Line ${g.source} ends ${g.gapM.toFixed(1)} m from line ${g.other}, beyond the ${snapM} m junction distance; join them if they meet on the ground.`);
    for (const c of coreIds.filter((_, ci) => !attachments.some(a => a.core === ci))) warnings.push(`Core ${c} is not reached by any spine line.`);
    if (!linked.size) warnings.push('The spine links no pair of cores under the check rules.');
    if (!cuts) warnings.push('Cut tests were skipped, so robustness and single points of failure are not reported.');

    return {
      ...base, status: 'ok', reasons: [], warnings: [...new Set(warnings)],
      summary: {
        sections: sections.length, severedSections: severed.length, nodes: nNodes, cores: cores.length, loops, loopsThroughCores, dendritic: loops === 0,
        linkedPairs: linked.size, robustPairs: tested ? robust : null, totalPairs,
        rho2: !tested ? null : totalPairs ? robust / totalPairs : 0,
        pFail1: measured ? sectionExposed.size / sections.length : null,
        exposedShare: measured && totalLength ? exposed / totalLength : null,
        lengthM: Math.round(totalLength), cutsTested,
        disturbanceWidthM, resolutionM: resolutionM == null ? null : Math.round(resolutionM * 100) / 100, testedWidths,
      },
      coreLinks: [...linked].sort().map(k => ({...pairObj(k), robust: tested ? !separated.has(k) : null})),
      singlePointsOfFailure: tested ? failures : null,
      severed,
      sections: sections.map(s => ({id: s.id, sourceId: s.sourceId, origin: s.origin, ...(s.order ? {order: s.order} : {}), widthM: s.widthM, lengthM: Math.round(s.lengthM * 10) / 10, ...(s.severedBy ? {severedBy: s.severedBy} : {})})),
      limitations: [
        'Links follow the connectivity check: the corridors derived from these lines, with roads, crossings, open water and the minimum width applied. Run the check on the saved retained layer for a treatment plan.',
        `Robust means no single disturbance up to ${disturbanceWidthM} m wide, centered anywhere, separates the pair under the check's rules. Disturbances may overlap core areas but do not remove core habitat in this test.`,
        `Reported failures are places where a disturbance up to ${testedWidths ? testedWidths.fineM : disturbanceWidthM} m wide (the tested width with its safety margin) separates a pair. Those marked verified separate it at ${disturbanceWidthM} m exactly; unverified ones were not confirmed at nominal width and may still be real.`,
        'Robust does not mean the corridor resists fire, storm or pests, or that species use it; two disturbances at once are not tested.',
        `Lines join where one ends within ${snapM} m of another; lines that cross mid-way are reported, not joined.`,
      ],
    };
  } catch (e) {
    return {...base, status: 'incomplete', reasons: [`Geometry engine error: ${e.message}`], warnings: [...new Set(warnings)]};
  }
}
