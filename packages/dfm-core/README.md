# @viridis/dfm-core

Pure analysis functions for Dendritic Forest Management (DFM). No storage, no network, no UI. Used by the DFM workspace and by VergeCommon's woodland projects, so both run the same calculation.

**Status: 0.1.0, experimental.** Not yet published to npm. Structural connectivity only; it does not establish species movement, genetic viability, regulatory compliance or old-growth condition.

## Corridor connectivity check

```js
import {checkConnectivity} from '@viridis/dfm-core';

const result = await checkConnectivity({
  coreAreas,   // polygons: properties.dfm_id, core_class (old-growth-candidate | old-growth-verified | riparian-core | reserve), evidence_id?
  retained,    // polygons: retained habitat (riparian buffers, corridors, retained stands)
  roads,       // centerlines (with params.roadWidthM) or road-surface polygons
  water,       // open-water polygons
  crossings,   // points: properties.passage = verified | assumed | none
  treatments,  // PROPOSED units: dfm_id, period, intensity, corridor_permitted?, reason?
  parcels,     // polygons: dfm_id, consent = covered | none
  params: {minWidthM: 100, minWidthSource: 'Co-op charter 2026, section 4', roadWidthM: 6},
});
// result.status: 'pass' | 'fail' | 'incomplete'
```

The check compares the current state (no proposed treatments) with the proposed plan:

1. Habitat = retained polygons and core areas, unioned; seams under 0.8 m between adjacent polygons are closed.
2. Road surfaces and open water are removed. A crossing recorded as `verified` or `assumed` (assumed warns) restores the road strip only on the road part it sits on, and only where habitat lies straight across the road on both sides.
3. Proposed treatment units are removed, except permitted light treatments: `corridor_permitted: true`, an `intensity` from `LIGHT_INTENSITIES` (`single-tree-selection`, `light-thinning`, `invasive-removal`, `restoration-planting`) and a `reason`.
4. Two cores are linked when a disk `minWidthM` across can travel between them inside the habitat (morphological erosion by half the width). A core counts only while at least half of it remains.
5. The check fails when a link is lost, naming each unit that alone breaks it (`causes`) and the units overlapping the lost corridor (`contributing`). Any unpermitted overlap with retained habitat or a core also fails, even when every link holds.
6. Links that hold at the minimum but not at the minimum + 10% are reported as `pinchedLinks`.

Inputs of the wrong geometry type, projected coordinates, extents over 0.5° and unsupported values return `incomplete` with reasons. Polygon operations retry on a 1 mm grid when edges nearly coincide.

All lengths are meters and all areas square meters. Results include the engine version and a SHA-256 checksum of the canonical input, so a stored result can be reproduced.

## Landscape Package

`toLandscapePackage(input, meta)` and `fromLandscapePackage(pkg)` read and write the exchange file defined in [`../dfm-schema`](../dfm-schema/README.md).

## Known limits

- A single road polygon that doubles back on itself (a U-shaped surface) is treated as one piece at a crossing; supply such roads as centerlines or split parts.
- Road-polygon width at a crossing is estimated from the local surface when not recorded; landings and junctions inflate it. Record `width_m` where it matters.
- Turf.js measures on a sphere (within about 0.5% of ellipsoidal distances).

## Development

```sh
npm ci
npm test
```

Tests use a fictional fixture woodlot (`fixtures/woodlot.mjs`) placed at the equator. Each test names the DFM Build Spec invariant it checks. Turf.js computes buffers in a local azimuthal projection around each feature, which is accurate for woodlot-scale extents (a few kilometers); larger landscapes need a projected-CRS engine.
