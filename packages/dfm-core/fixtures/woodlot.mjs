// Fictional test woodlot, drawn in local meters and placed at the equator so it
// cannot be mistaken for a real site. Turf measures on a sphere of radius
// 6,371,008.8 m, so 1 degree = 111,194.93 m in both directions at the equator;
// the fixture uses the same factor, making hand-calculated widths exact.
//
//   y (m)
//  1000 +----------------------------------------------------------+
//       |              road (x=1000) |                             |
//   700 |  [core A    ]              |                [core B    ] |
//       |  [old-growth]==== corridor (width W) ====== [riparian  ] |
//   300 |  [candidate ]     harvest-3 cuts here   |   [core      ] |
//     0 +----------------------------------------------------------+
//       0    100      400   700  800           1000  1600     1900  2000  x (m)

const M_PER_DEG = (2 * Math.PI * 6371008.8) / 360, LON = 1 / M_PER_DEG, LAT = 1 / M_PER_DEG;
export const pt = (x, y) => [x * LON, y * LAT];
export const rect = (x0, y0, x1, y1) => ({type: 'Polygon', coordinates: [[pt(x0, y0), pt(x1, y0), pt(x1, y1), pt(x0, y1), pt(x0, y0)]]});
const feature = (geometry, properties) => ({type: 'Feature', geometry, properties});

/**
 * Build the fixture. Options:
 *   corridorWidth (m, default 120), road (true), crossing ('verified'|'assumed'|'none'|null),
 *   treatments (array of unit names from UNITS), minWidthM (100), parcels (true)
 */
export function woodlot({corridorWidth = 120, road = true, crossing = 'verified', treatments = [], minWidthM = 100, parcels = true, coreClassA = 'old-growth-candidate', evidenceA} = {}) {
  const yMid = 500, half = corridorWidth / 2;
  return {
    coreAreas: [
      feature(rect(100, 300, 400, 700), {dfm_id: 'core-A', core_class: coreClassA, ...(evidenceA ? {evidence_id: evidenceA} : {})}),
      feature(rect(1600, 300, 1900, 700), {dfm_id: 'core-B', core_class: 'riparian-core'}),
    ],
    retained: [feature(rect(400, yMid - half, 1600, yMid + half), {dfm_id: 'corridor-1'})],
    roads: road ? [feature({type: 'LineString', coordinates: [pt(1000, 0), pt(1000, 1000)]}, {dfm_id: 'road-1'})] : [],
    water: [],
    crossings: road && crossing ? [feature({type: 'Point', coordinates: pt(1000, yMid)}, {dfm_id: 'crossing-1', passage: crossing})] : [],
    treatments: treatments.map(name => structuredClone(UNITS[name])),
    parcels: parcels ? [
      feature(rect(0, 0, 1000, 1000), {dfm_id: 'parcel-west', consent: 'covered', consent_ref: 'fixture-consent-1'}),
      feature(rect(1000, 0, 2000, 1000), {dfm_id: 'parcel-east', consent: 'none'}),
    ] : [],
    params: {minWidthM, minWidthSource: 'Fixture value for tests', roadWidthM: 6},
  };
}

export const UNITS = {
  // Cuts straight across the corridor: 100 m wide, from y=300 to y=800.
  'harvest-3': feature(rect(700, 300, 800, 800), {dfm_id: 'harvest-3', period: '2027', intensity: 'clearcut'}),
  // Same footprint, recorded as a permitted light treatment.
  'thin-3': feature(rect(700, 300, 800, 800), {dfm_id: 'thin-3', period: '2027', intensity: 'single-tree-selection', corridor_permitted: true, reason: 'Release of advance regeneration; canopy retained above 70%.'}),
  // Away from the corridor.
  'harvest-1': feature(rect(1200, 700, 1500, 950), {dfm_id: 'harvest-1', period: '2027', intensity: 'shelterwood'}),
  // Takes a 30 m bite from the corridor edge, leaving 90 m of a 120 m corridor.
  'edge-cut': feature(rect(1200, 530, 1300, 800), {dfm_id: 'edge-cut', period: '2028', intensity: 'patch cut'}),
};
