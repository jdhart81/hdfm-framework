# @viridis/dfm-core

Pure analysis functions for Dendritic Forest Management (DFM). No storage, no network, no UI. Used by the DFM workspace and by VergeCommon's woodland projects, so both run the same calculation.

**Status: 0.2.0, experimental.** Not yet published to npm. Structural connectivity only; it does not establish species movement, genetic viability, regulatory compliance or old-growth condition.

- [Corridor connectivity check](#corridor-connectivity-check): does a harvest plan break a link between core areas?
- [The old-growth spine](#the-old-growth-spine): draft the spine from streams and ridges, test it as a network, project it forward in time, find climate routes and the next woodlots to join.
- [Landscape Package](#landscape-package): the exchange file.

## Corridor connectivity check

```js
import {checkConnectivity, checkConnectivitySync} from '@viridis/dfm-core';

// checkConnectivitySync gives the same result without a Promise, for synchronous command handlers.
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

All lengths are meters and all areas square meters. Results include the engine version and a SHA-256 checksum of the canonical input (computed in plain JS, identical to Node's `crypto`), so a stored result can be reproduced.

Dependencies are individual Turf.js 7 modules, so a bundle includes only what the check uses (about 100 KB gzipped, mostly the polygon clipping and buffering libraries).

## The old-growth spine

The spine is the dendritic network of retained forest that follows a landscape's natural corridors. Streams set its branches, and it is widest along the largest rivers; links over ridges, along dry valleys and across saddles close loops. A watershed is mapped whole, then built woodlot by woodlot: each woodlot that joins commits its stretch, and the spine ages toward old growth while the woods around it are worked.

Five functions, all synchronous. Each returns `status: 'ok' | 'incomplete'` with `reasons` and `warnings`, the engine version, an input checksum and its `limitations`; none throws on bad input. Result shapes are in [`../dfm-schema/spine-results.schema.json`](../dfm-schema/spine-results.schema.json).

```js
import {deriveSpine, spineNetwork, projectSpine, climateRoutes, buildOutFrontier} from '@viridis/dfm-core';

const input = {
  streams,     // lines: dfm_id, stream_order (Strahler, 1-12)
  connectors,  // lines: dfm_id, kind = ridge | valley | saddle
  coreAreas, roads, crossings, water, parcels,
  params: {
    minWidthM: 100, minWidthSource: 'Co-op charter 2026, section 4',
    spineWidthByOrderM: {1: 115, 2: 140, 3: 180}, spineWidthSource: 'Forester recommendation, 2026',
    connectorWidthM: 115, connectorWidthSource: 'Forester recommendation, 2026',
  },
};
const draft = deriveSpine(input);       // corridor polygons, for a steward to review
const network = spineNetwork(input);    // links, loops, single points of failure
const plan = {...input, retained: reviewedFeatures}; // the reviewed draft, with stand_age recorded
const outlook = projectSpine(plan, {years: [2026, 2036, 2051, 2076, 2126]});
const routes = climateRoutes(plan);
const next = buildOutFrontier(plan);
```

### `deriveSpine(input)`

Drafts retained-habitat polygons from stream and link lines. Each corridor is the line buffered to its width; its properties record `dfm_id` (`spine-<line id>`), `spine: true`, `origin` (stream, ridge, valley or saddle), `source_id`, `stream_order`, `width_m` and `width_source`. Review the draft before saving it as the retained layer.

- **SP1** Every section is at least `minWidthM` wide. Widths come from `spineWidthByOrderM` (each at least the minimum, with `spineWidthSource`) or, with no table, the minimum plus the pinch margin rounded up past the next 5 m, recorded as a default. A width at the minimum, or within the pinch margin, warns.
- **SP2** Width never decreases with stream order.
- **SP3** Streams need a whole `stream_order` from 1 to 12; links need a `kind`; every line is at least 1 m long; IDs are unique. Anything else is `incomplete`, naming the line.
- **SP4** Provenance is recorded on every section, and derivation never assigns a core class. A corridor that overlaps mapped open water warns.

### `spineNetwork(input, {cuts = true, cutBudget = 20000})`

Tests the draft as a network, on the habitat the connectivity check would see.

- **SP5** Links come only from the check's rules applied to the derived corridors and the cores: roads sever unless a crossing on that road carries the corridor, open water is removed, and the minimum width holds. The line graph never adds a link.
- **SP6** Single points of failure. A disturbance is a disk `disturbanceWidthM` across (default: the widest corridor plus 2 m), centered anywhere; it may overlap a core but does not remove core habitat. Centers are tested on a grid whose disks are enlarged to cover every position in between (see `src/robust.mjs` for the argument), so **a pair reported robust cannot be separated by any such disturbance under the check's rules**. Each place where a tested cut separates linked cores is reported as a zone with the pairs it separates, then re-tested exactly at nominal width and marked `verified` or not. An unverified zone may still be real; it was not confirmed at nominal width.
- **SP7** `rho2` = robust pairs / all core pairs (the HDFM paper's 2-edge connectivity, over cores, tested on the habitat); `pFail1` = share of sections some separating disturbance reaches; `exposedShare` = that length / total length. Robustness is reported only from a completed search: `cuts: false`, or a search that needs more than `cutBudget` cuts, reports it as `null`, never as true.
- **SP8** Loops are counted on the line graph of sections between junctions (lines join where one ends within `junctionSnapM`, default 5 m, at most 10 m). Routes that close only through a core are counted separately as `loopsThroughCores`. Lines that cross mid-way are reported, not joined.

`minWidthM` must be at least 5 m for the network analysis. The disturbance raster spacing is at most 2.5 m and at most a quarter of the minimum width.

### `projectSpine(input, {years})`

At each milestone year, the core pairs linked through committed forest and through forest at old-growth age. Uses `params.ageAsOfYear`, `oldGrowthAgeYears` with `oldGrowthAgeSource`, and `milestoneYears` (or `years`, up to 12, from `ageAsOfYear` on); `stand_age` on retained features and cores; `consent_year` or `planned_year` on parcels.

- **PJ1** A covered parcel counts from its `consent_year` (no later than `ageAsOfYear`); a parcel without consent counts from its `planned_year`, which must be after `ageAsOfYear`. Consent never lapses in a projection.
- **PJ2** The proposed treatment units apply at every milestone.
- **PJ3** Forest is at old-growth age in year Y when `stand_age + (Y - ageAsOfYear) >= oldGrowthAgeYears`. Where features overlap, the youngest recorded age applies; a feature without an age never counts as old. A core is a link's endpoint whatever its age. Age is reported, never condition.
- **PJ4** Old-age links ⊆ committed links ⊆ links after the plan, in every year.
- **PJ5** Committed and old-age links never decrease from one milestone to the next: no fire, storm or new road is predicted.

### `climateRoutes(input, check?)`

For each core, the coolest core it can reach through links that hold after the plan. Cores carry `temp_c` (with `params.coreTempSource`); an optional warming target `params.climateWarmingC` needs `params.climateSource`.

- **CL1** Routes use only the links the check finds after the plan, chained through linked cores. A supplied check result must match the input's checksum; otherwise it is recomputed, with a warning.
- **CL2** A core without `temp_c` is `unknown`, never `coolest`.
- **CL3** `route`: a reachable core is cooler by at least the target (by any amount with no target); `short`: cooler, by less; `none`: nothing reachable is cooler, although a cooler core exists; `coolest`: every core has a temperature and none is cooler. `none`, `short` and `unknown` are flagged.

### `buildOutFrontier(input)`

The parcels without consent whose spine touches the committed spine, with the core pairs each would link if it alone joined now, its spine area and its direction (8-point compass and bearing) from the committed piece it adjoins.

- **FR1** Only features marked `spine: true` count as spine when any are marked. With no committed spine yet, every parcel holding spine is listed, without a direction.
- **FR2** Links each parcel would complete are computed through committed forest, with the proposed treatment units applied.
- **FR3** Ordered by links completed, then spine area, then ID. Parcel IDs must be unique.

### Performance

The polygon operations dominate. On the watershed fixture (`fixtures/watershed.mjs`: 14 lines, 4 cores, about 2.5 km across), measured on one core of a cloud build machine:

| Call | Time |
|---|---|
| `deriveSpine` | under 0.1 s |
| `spineNetwork` | about 4 s (about 2,200 cuts); 1 s with `cuts: false` |
| `checkConnectivitySync` on the derived spine | about 3 s |
| `projectSpine` (5 milestones) | about 5 s |
| `climateRoutes`, `buildOutFrontier` | about 3 s each |

All calls block the thread they run on. A server should run them in a worker thread or a job queue, not in a request handler. Larger landscapes raise the cut count roughly with area; past `cutBudget` the network reports robustness as `null` with a warning. Analyze one watershed at a time.

## Landscape Package

`toLandscapePackage(input, meta)` and `fromLandscapePackage(pkg)` read and write the exchange file defined in [`../dfm-schema`](../dfm-schema/README.md). The optional `streams` and `connectors` layers are written only when they hold features and read back only when present, so a package without them, and its checksum, are unchanged from 0.1.0.

## Known limits

- A single road polygon that doubles back on itself (a U-shaped surface) is treated as one piece at a crossing; supply such roads as centerlines or split parts.
- Road-polygon width at a crossing is estimated from the local surface when not recorded; landings and junctions inflate it. Record `width_m` where it matters.
- Turf.js measures on a sphere (within about 0.5% of ellipsoidal distances).
- The spine network tests one disturbance at a time, shaped as a disk. Two disturbances at once, or one long disturbance such as a new road across two corridors, are not tested.
- Projections take ages and join years as recorded and predict no loss. Climate routes use the temperatures supplied; they do not model climate or species.

## Development

```sh
npm ci
npm test
```

Tests use fictional fixtures placed at the equator: a woodlot (`fixtures/woodlot.mjs`) and a watershed (`fixtures/watershed.mjs`, the illustration on dendriticforest.com, at 5 m per drawing unit). Each test names the DFM Build Spec invariant it checks. A soundness test compares every "robust" claim with an independent exact check at sampled disturbance positions. Turf.js computes buffers in a local azimuthal projection around each feature, which is accurate for woodlot and watershed-scale extents (a few kilometers); larger landscapes need a projected-CRS engine.
