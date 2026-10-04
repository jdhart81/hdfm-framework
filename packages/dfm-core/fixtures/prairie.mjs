// Fictional flat prairie for the biome tests (B1-B7). No ridges or valleys: a low-gradient creek,
// a railroad right-of-way that was never plowed, and a chain of wet-meadow patches (stepping
// stones) north across cropland to a remnant on a moraine. Local meters, placed at the equator like
// fixtures/woodlot.mjs so it cannot be mistaken for a real place. North is up.
//
//   y (m)
//  2400 +----------------- exit-n (to the next landscape north) ------------------+
//       |                        [remnant-n]  moraine link                          |
//  1775 |                            o patch-4      gaps between patches 58-90 m    |
//       |                            o patch-3      (cropland)                      |
//       |                            o patch-2                                      |
//  1000 |[remnant-w]==== creek ===== o patch-1 ===================== [marsh-e]      |
//       |          road |                              // rail right-of-way          |
//     0 |               |                           [preserve-s]                    |
//       -300           500           1000           1500           2000         2300  x (m)
//
// Cores are nearly the same temperature, as in flat country; the cooler ground is beyond the
// landscape, reached through exit-n.

import {pt, rect} from './woodlot.mjs';

export const AS_OF = 2026;
const feature = (geometry, properties) => ({type: 'Feature', geometry, properties});
const line = (coords, properties) => feature({type: 'LineString', coordinates: coords.map(([x, y]) => pt(x, y))}, properties);
const box = (x0, y0, x1, y1, properties) => feature(rect(x0, y0, x1, y1), properties);

export const CREEK = [[0, 900], [500, 950], [1000, 900], [1500, 960], [2000, 1000]];
/** Stepping stones: 110 m wet-meadow patches north of the creek; the largest gap (to remnant-n) is 90 m. */
export const PATCHES = [
  ['patch-1', 1010, 1120],
  ['patch-2', 1195, 1305],
  ['patch-3', 1385, 1495],
  ['patch-4', 1575, 1685],
];
/** Recorded ages (years as of AS_OF) by source line or patch; remnants are marked separately. */
export const AGES = {creek: 31, moraine: 31, 'patch-1': 16, 'patch-2': 16, 'patch-3': 16, 'patch-4': 16};
export const REMNANT_SOURCE = 'Fixture record: never plowed (fictional)';

export const UNITS = {
  // A spring burn over patches 2 and 3: upkeep, permitted inside the habitat.
  'burn-2': box(900, 1180, 1100, 1510, {dfm_id: 'burn-2', period: '2027', intensity: 'prescribed-burn', corridor_permitted: true, reason: 'Spring burn to set back woody encroachment (fixture).'}),
  // Plowing patch 3 for crops: removes a stepping stone.
  'plow-3': box(930, 1370, 1070, 1510, {dfm_id: 'plow-3', period: '2027', intensity: 'conversion'}),
};

/**
 * Build the fixture. Options: gap (params.gapCrossingM; 0 leaves it unset), exits (true),
 * treatments (names from UNITS), retained (habitat already mapped; default the patches).
 */
export function prairie({gap = 100, exits = true, treatments = [], retained = null} = {}) {
  return {
    streams: [line(CREEK, {dfm_id: 'creek', stream_order: 2})],
    connectors: [
      line([[1700, 0], [1650, 972]], {dfm_id: 'rail', kind: 'right-of-way', width_m: 75, width_source: 'Fixture: 75 m railroad right-of-way'}),
      line([[1000, 2075], [1000, 2400]], {dfm_id: 'moraine', kind: 'moraine'}),
    ],
    coreAreas: [
      box(-300, 700, 0, 1100, {dfm_id: 'remnant-w', core_class: 'old-growth-candidate', remnant: true, remnant_source: REMNANT_SOURCE, temp_c: 9.7}),
      box(2000, 800, 2300, 1200, {dfm_id: 'marsh-e', core_class: 'riparian-core', temp_c: 9.9}),
      box(800, 1775, 1200, 2075, {dfm_id: 'remnant-n', core_class: 'old-growth-candidate', remnant: true, remnant_source: REMNANT_SOURCE, temp_c: 9.4}),
      box(1500, -300, 1900, 0, {dfm_id: 'preserve-s', core_class: 'reserve', stand_age: 46, temp_c: 10.0}),
    ],
    retained: retained ?? patches(),
    roads: [line([[500, -300], [500, 2400]], {dfm_id: 'county-road', width_m: 8})],
    water: [],
    crossings: [feature({type: 'Point', coordinates: pt(500, 950)}, {dfm_id: 'creek-bridge', passage: 'verified'})],
    treatments: treatments.map(n => structuredClone(UNITS[n])),
    parcels: [
      box(-300, -300, 500, 2400, {dfm_id: 'farm-a', consent: 'covered', consent_ref: 'fixture-farm-a', consent_year: 2024}),
      box(500, -300, 1300, 1150, {dfm_id: 'farm-b', consent: 'none', planned_year: 2030}),
      box(500, 1150, 1300, 2400, {dfm_id: 'farm-n', consent: 'none', planned_year: 2035}),
      box(1300, -300, 2300, 2400, {dfm_id: 'farm-c', consent: 'none'}),
    ],
    ...(exits ? {exits: [box(950, 2330, 1050, 2400, {dfm_id: 'exit-n', temp_c: 7.2, toward: 'The next landscape north (fictional)'})]} : {}),
    params: {
      minWidthM: 60, minWidthSource: 'Fixture value for tests (grassland)', roadWidthM: 8,
      spineWidthByOrderM: {1: 75, 2: 100}, spineWidthSource: 'Fixture widths by stream order',
      connectorWidthM: 75, connectorWidthSource: 'Fixture link width',
      ...(gap ? {gapCrossingM: gap, gapCrossingSource: 'Fixture value for tests: gaps grassland birds cross'} : {}),
      disturbanceWidthM: 120,
      ageAsOfYear: AS_OF, oldGrowthAgeYears: 100, oldGrowthAgeSource: 'Fixture threshold (grassland)',
      coreTempSource: 'Fixture temperatures', climateWarmingC: 2, climateSource: 'Fixture warming target',
    },
  };
}

/** The stepping-stone patches as retained habitat, with their ages. */
export const patches = () => PATCHES.map(([id, y0, y1]) => box(945, y0, 1055, y1, {dfm_id: id, stand_age: AGES[id]}));

/** Ages for derived spine features: the right-of-way is a remnant; the rest by source line. */
export const withPrairieAges = features => features.map(f => ({
  ...f,
  properties: {...f.properties, ...(f.properties.source_id === 'rail' ? {remnant: true, remnant_source: REMNANT_SOURCE} : {stand_age: AGES[f.properties.source_id]})},
}));
