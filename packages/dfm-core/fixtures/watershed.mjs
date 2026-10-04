// Fictional watershed for the spine, projection, climate and build-out tests. It is the
// site's illustration at 5 m per drawing unit (2,800 m x 2,600 m), placed at the equator
// like fixtures/woodlot.mjs so it cannot be mistaken for a real place. North is up.
//
//   Headwaters (cool, north): two first-order streams meet and run south as the main stem,
//   joined by second-order tributaries from the west and east and first-order streams
//   lower down. Two saddle links close loops: northwest (headwater to west tributary) and
//   east (east tributary to the southeast stream). A road crosses the main stem and the
//   east link at recorded crossings. Nine woodlots tile the map; the middle, north and
//   south ones have consent, others plan to join later.
//
// Drawing units: x right, y down, 0..560 x 0..520. Curves are the illustration's cubic
// Beziers, sampled finely enough that junctions stay well inside the 5 m snap distance.

const M_PER_DEG = (2 * Math.PI * 6371008.8) / 360;
const S = 5; // meters per drawing unit
/** Drawing units to [lon, lat]. */
export const P = (x, y) => [(x * S) / M_PER_DEG, ((520 - y) * S) / M_PER_DEG];

function cubic(p0, c1, c2, p1, n = 32) {
  const out = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n, u = 1 - t;
    out.push([u ** 3 * p0[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t ** 3 * p1[0], u ** 3 * p0[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t ** 3 * p1[1]]);
  }
  return out;
}
/** A path of chained cubics: [[p0, c1, c2, p1], [c1, c2, p2], ...] in drawing units. */
function path(first, ...rest) {
  let pts = cubic(...first), end = first[3];
  for (const [c1, c2, p] of rest) { pts = pts.concat(cubic(end, c1, c2, p).slice(1)); end = p; }
  return pts.map(([x, y]) => P(x, y));
}
const line = (coords, properties) => ({type: 'Feature', geometry: {type: 'LineString', coordinates: coords}, properties});

/** First intersection of two polylines ([lon, lat] arrays), by segment intersection. */
function crossingPoint(a, b) {
  for (let i = 1; i < a.length; i++) for (let j = 1; j < b.length; j++) {
    const [p, r] = [a[i - 1], [a[i][0] - a[i - 1][0], a[i][1] - a[i - 1][1]]];
    const [q, s] = [b[j - 1], [b[j][0] - b[j - 1][0], b[j][1] - b[j - 1][1]]];
    const den = r[0] * s[1] - r[1] * s[0];
    if (den === 0) continue;
    const t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / den, u = ((q[0] - p[0]) * r[1] - (q[1] - p[1]) * r[0]) / den;
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) return [p[0] + t * r[0], p[1] + t * r[1]];
  }
  throw new Error('Fixture lines do not cross.');
}
const poly = (pts, properties) => {
  const ring = pts.map(([x, y]) => P(x, y));
  return {type: 'Feature', geometry: {type: 'Polygon', coordinates: [[...ring, ring[0]]]}, properties};
};

export const STREAMS = [
  ['h1', 1, [[238, 38], [246, 62], [262, 92], [282, 112]]],
  ['h2', 1, [[338, 28], [330, 60], [306, 92], [282, 112]]],
  ['main-upper', 2, [[282, 112], [276, 150], [280, 196], [288, 232]]],
  ['w1a', 1, [[78, 62], [96, 100], [122, 140], [150, 168]]],
  ['w1b', 1, [[34, 188], [72, 176], [112, 170], [150, 168]]],
  ['w1', 2, [[150, 168], [196, 178], [246, 202], [288, 232]]],
  ['e1a', 1, [[476, 48], [466, 88], [446, 128], [430, 160]]],
  ['e1b', 1, [[532, 182], [498, 168], [462, 160], [430, 160]]],
  ['e1', 2, [[430, 160], [392, 200], [340, 250], [300, 306]]],
  ['w2', 1, [[52, 352], [122, 362], [230, 372], [306, 398]]],
  ['e2', 1, [[530, 362], [460, 380], [380, 418], [322, 452]]],
];
const MAIN = [[[288, 232], [292, 258], [296, 282], [300, 306]], [[302, 338], [302, 370], [306, 398]], [[310, 418], [316, 436], [322, 452]], [[328, 472], [334, 496], [338, 520]]];
const LINKS = [
  ['saddle-nw', 'saddle', [[[251, 69], [222, 70], [196, 76], [176, 82]], [[150, 90], [124, 100], [103, 108]]]],
  ['saddle-e', 'saddle', [[[501, 171], [503, 210], [496, 240], [490, 268]], [[484, 300], [480, 340], [476, 379]]]],
];

/**
 * Stand ages (years at AS_OF) by spine line: older headwater stands, younger lowland forest, and
 * young saddle links. Streams that meet in the lowland share the lowland's age, because derived
 * corridors overlap at confluences and the youngest recorded age applies there.
 */
export const AGES = {h1: 90, h2: 90, w1a: 90, w1b: 90, e1a: 90, e1b: 90, 'main-upper': 70, w1: 70, e1: 70, main: 60, w2: 60, e2: 60, 'saddle-nw': 30, 'saddle-e': 30};
export const AS_OF = 2026;

export const PARCELS = [
  // [id, polygon, consent, consent_year | planned_year]
  ['woodlot-1', [[190, 130], [385, 125], [395, 325], [210, 330]], 'covered', {consent_year: 2024}],
  ['woodlot-2', [[200, 0], [395, 0], [385, 125], [190, 130]], 'covered', {consent_year: 2025}],
  ['woodlot-3', [[210, 330], [395, 325], [410, 520], [215, 520]], 'covered', {consent_year: 2026}],
  ['woodlot-4', [[385, 125], [560, 115], [560, 330], [395, 325]], 'none', {planned_year: 2030}],
  ['woodlot-5', [[0, 120], [190, 130], [210, 330], [0, 335]], 'none', {planned_year: 2032}],
  ['woodlot-6', [[395, 0], [560, 0], [560, 115], [385, 125]], 'none', {}],
  ['woodlot-7', [[0, 335], [210, 330], [215, 520], [0, 520]], 'none', {planned_year: 2040}],
  ['woodlot-8', [[0, 0], [200, 0], [190, 130], [0, 120]], 'none', {planned_year: 2045}],
  ['woodlot-9', [[395, 325], [560, 330], [560, 520], [410, 520]], 'none', {}],
];

export const CORES = [
  ['core-n', [[244, 18], [322, 10], [342, 40], [308, 66], [250, 52]], 'old-growth-candidate', 6.0],
  ['core-w', [[100, 124], [146, 130], [156, 158], [124, 172], [98, 152]], 'old-growth-candidate', 7.2],
  ['core-e', [[446, 104], [494, 110], [500, 142], [468, 152], [444, 134]], 'reserve', 7.0],
  ['core-s', [[346, 470], [392, 474], [398, 506], [354, 514], [340, 494]], 'riparian-core', 8.6],
];

/** Harvest units: away from the spine, plus one that cuts the main stem between the west and east tributaries. */
export const UNITS = {
  'unit-a': poly([[222, 244], [258, 240], [262, 266], [226, 270]], {dfm_id: 'unit-a', period: '2027', intensity: 'patch cut'}),
  'unit-b': poly([[420, 214], [458, 210], [462, 240], [424, 244]], {dfm_id: 'unit-b', period: '2027', intensity: 'shelterwood'}),
  'cut-main': poly([[268, 262], [330, 262], [330, 284], [268, 284]], {dfm_id: 'cut-main', period: '2028', intensity: 'clearcut'}),
};

/** Spine lines (streams and saddle links) as GeoJSON. */
export function spineLines() {
  return {
    streams: [
      ...STREAMS.map(([id, order, pts]) => line(path(pts), {dfm_id: id, stream_order: order})),
      line(path(...MAIN), {dfm_id: 'main', stream_order: 3}),
    ],
    connectors: LINKS.map(([id, kind, segs]) => line(path(...segs), {dfm_id: id, kind})),
  };
}

/**
 * The watershed as one input. Options:
 *   crossingE ('assumed' | 'verified' | 'none' | null): the road crossing on the east saddle link
 *   treatments: unit names from UNITS
 *   temps (true): core temp_c values
 *   retained: retained habitat features (tests usually pass the derived spine, with ages)
 */
export function watershed({crossingE = 'assumed', treatments = ['unit-a', 'unit-b'], temps = true, retained = null} = {}) {
  const lines = spineLines();
  const road = path([[0, 318], [120, 312], [220, 318], [300, 326]], [[390, 334], [480, 326], [560, 318]]);
  const at = sourceId => crossingPoint(road, [...lines.streams, ...lines.connectors].find(f => f.properties.dfm_id === sourceId).geometry.coordinates);
  return {
    ...lines,
    coreAreas: CORES.map(([id, pts, coreClass, t]) => poly(pts, {dfm_id: id, core_class: coreClass, ...(temps ? {temp_c: t} : {})})),
    retained: retained ?? [],
    roads: [line(road, {dfm_id: 'town-road'})],
    water: [],
    crossings: [
      {type: 'Feature', geometry: {type: 'Point', coordinates: at('main')}, properties: {dfm_id: 'culvert-main', passage: 'verified'}},
      ...(crossingE ? [{type: 'Feature', geometry: {type: 'Point', coordinates: at('saddle-e')}, properties: {dfm_id: 'underpass-e', passage: crossingE}}] : []),
    ],
    treatments: treatments.map(n => structuredClone(UNITS[n])),
    parcels: PARCELS.map(([id, pts, consent, years]) => poly(pts, {dfm_id: id, consent, ...(consent === 'covered' ? {consent_ref: `fixture-${id}`} : {}), ...years})),
    params: {
      minWidthM: 100, minWidthSource: 'Fixture value for tests', roadWidthM: 6,
      spineWidthByOrderM: {1: 115, 2: 140, 3: 180}, spineWidthSource: 'Fixture widths by stream order',
      connectorWidthM: 115, connectorWidthSource: 'Fixture link width',
      ageAsOfYear: AS_OF, oldGrowthAgeYears: 150, oldGrowthAgeSource: 'Fixture threshold',
      coreTempSource: 'Fixture temperatures', climateWarmingC: 2, climateSource: 'Fixture warming target',
    },
  };
}

/** Add stand ages to derived spine features by their source line. */
export const withAges = features => features.map(f => ({...f, properties: {...f.properties, stand_age: AGES[f.properties.source_id]}}));
