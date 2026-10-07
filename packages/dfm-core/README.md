# @viridis/dfm-core

Pure analysis functions for Dendritic Forest Management (DFM). No storage, no network, no UI. Used by the DFM workspace and by VergeCommon's woodland projects, so both run the same calculation.

**Status: 0.3.0, experimental.** Not yet published to npm. Structural connectivity only; it does not establish species movement, genetic viability, regulatory compliance or old-growth condition.

- [Corridor connectivity check](#corridor-connectivity-check): does a treatment plan break a link between core areas?
- [The old-growth spine](#the-old-growth-spine): draft the spine from streams, ridges and other natural lines, test it as a network, project it forward in time, find climate routes and the next holdings to join.
- [Any biome](#any-biome-forest-prairie-and-flat-country): forest, prairie, savanna, wetland and flat country, with stepping stones, flat-land links, remnants, upkeep by fire and grazing, and exits to the next landscape.
- [Native old growth, not plantations](#native-old-growth-not-plantations): stand origin, native composition and native-only planting, so a plantation or a stand of introduced species is never projected as old growth and a corridor is never replanted with introduced species.
- [Landscape Package](#landscape-package): the exchange file.

## Corridor connectivity check

```js
import {checkConnectivity, checkConnectivitySync} from '@viridis/dfm-core';

// checkConnectivitySync gives the same result without a Promise, for synchronous command handlers.
const result = await checkConnectivity({
  coreAreas,   // polygons: properties.dfm_id, core_class (old-growth-candidate | old-growth-verified | riparian-core | reserve), evidence_id?
  retained,    // polygons: retained habitat (riparian buffers, corridors, retained stands, habitat patches)
  roads,       // centerlines (with params.roadWidthM) or road-surface polygons
  water,       // open-water polygons
  crossings,   // points: properties.passage = verified | assumed | none
  treatments,  // PROPOSED units: dfm_id, period, intensity, corridor_permitted?, reason?,
               //   and for a restoration planting, species?: [{name, native}] + native_status_source?
  parcels,     // polygons: dfm_id, consent = covered | none
  params: {
    minWidthM: 100, minWidthSource: 'Co-op charter 2026, section 4', roadWidthM: 6,
    // Optional stepping stones (see "Any biome"):
    // gapCrossingM: 100, gapCrossingSource: 'Dispersal study for the species the corridors serve',
    // Optional native-planting policy (see "Native old growth, not plantations"):
    // nativeStatusSource: 'USDA PLANTS native status for Vermont',
  },
});
// result.status: 'pass' | 'fail' | 'incomplete'
```

The check compares the current state (no proposed treatments) with the proposed plan:

1. Habitat = retained polygons and core areas, unioned; seams under 0.8 m between adjacent polygons are closed.
2. Road surfaces and open water are removed. A crossing recorded as `verified` or `assumed` (assumed warns) restores the road strip only on the road part it sits on, and only where habitat lies straight across the road on both sides.
3. Proposed treatment units are removed, except permitted light treatments: `corridor_permitted: true`, an `intensity` from `LIGHT_INTENSITIES` and a `reason`. The list holds forest treatments (`single-tree-selection`, `light-thinning`, `invasive-removal`, `restoration-planting`) and the upkeep that fire- and grazing-dependent habitat needs (`prescribed-burn`, `prescribed-grazing`, `late-season-mowing`, `brush-management`). A restoration planting that lists its species stays permitted only when every species is native, with a recorded source for native status (N1).
4. Two cores are linked when a disk `minWidthM` across can travel between them inside the habitat (morphological erosion by half the width). A core counts only while at least half of it remains. With stepping stones (`params.gapCrossingM`), pieces of habitat at least the minimum width across also link when the gap between them is at most that distance and crosses no road.
5. The check fails when a link is lost, naming each unit that alone breaks it (`causes`) and the units overlapping the lost corridor (`contributing`). Any unpermitted overlap with retained habitat or a core also fails, even when every link holds.
6. Links that hold at the minimum but not at the minimum + 10% are reported as `pinchedLinks`.

Inputs of the wrong geometry type, projected coordinates, extents over 0.5° and unsupported values return `incomplete` with reasons. Polygon operations retry on a 1 mm grid when edges nearly coincide.

All lengths are meters and all areas square meters. Results include the engine version and a SHA-256 checksum of the canonical input (computed in plain JS, identical to Node's `crypto`), so a stored result can be reproduced. Without `gapCrossingM`, results are identical to engine 0.1.0 apart from the version string.

Dependencies are individual Turf.js 7 modules, so a bundle includes only what the check uses (about 100 KB gzipped, mostly the polygon clipping and buffering libraries).

## The old-growth spine

The spine is the dendritic network of retained habitat that follows a landscape's natural corridors. Streams set its branches, and it is widest along the largest rivers; links over ridges, along dry valleys and across saddles close loops. In flat country, links follow swales, escarpments, moraines and shorelines, or land use: rights-of-way, field margins and hedgerows. A watershed is mapped whole, then built holding by holding: each woodlot, farm or ranch that joins commits its stretch, and the spine ages toward old growth while the land around it is worked.

Five functions, all synchronous. Each returns `status: 'ok' | 'incomplete'` with `reasons` and `warnings`, the engine version, an input checksum and its `limitations`; none throws on bad input. Result shapes are in [`../dfm-schema/spine-results.schema.json`](../dfm-schema/spine-results.schema.json).

```js
import {deriveSpine, spineNetwork, projectSpine, climateRoutes, buildOutFrontier} from '@viridis/dfm-core';

const input = {
  streams,     // lines: dfm_id, stream_order (Strahler, 1-12)
  connectors,  // lines: dfm_id, kind (SPINE_LINK_KINDS), width_m + width_source?
  retained,    // optional polygons already mapped: habitat patches (stepping stones), remnants
  exits,       // optional polygons where the spine continues into the next landscape: dfm_id, temp_c, toward?
  coreAreas, roads, crossings, water, parcels,
  params: {
    minWidthM: 100, minWidthSource: 'Co-op charter 2026, section 4',
    spineWidthByOrderM: {1: 115, 2: 140, 3: 180}, spineWidthSource: 'Forester recommendation, 2026',
    connectorWidthM: 115, connectorWidthSource: 'Forester recommendation, 2026',
  },
};
const draft = deriveSpine(input);       // corridor polygons, for a steward to review
const network = spineNetwork(input);    // links, loops, single points of failure
const plan = {...input, retained: [...reviewedFeatures, ...(input.retained ?? [])]}; // reviewed draft with stand_age or remnant recorded
const outlook = projectSpine(plan, {years: [2026, 2036, 2051, 2076, 2126]});
const routes = climateRoutes(plan);
const next = buildOutFrontier(plan);
```

### `deriveSpine(input)`

Drafts retained-habitat polygons from stream and link lines. Each corridor is the line buffered to its width; its properties record `dfm_id` (`spine-<line id>`), `spine: true`, `origin` (`stream` or the link's kind), `source_id`, `stream_order`, `width_m` and `width_source`. Review the draft before saving it as the retained layer.

- **SP1** Every section is at least `minWidthM` wide. Stream widths come from `spineWidthByOrderM` (each at least the minimum, with `spineWidthSource`) or, with no table, the minimum plus the pinch margin rounded up past the next 5 m, recorded as a default. A link uses its own `width_m` (with `width_source`) when recorded, otherwise `connectorWidthM`. A width at the minimum, or within the pinch margin, warns; a width below the minimum is `incomplete`.
- **SP2** Width never decreases with stream order.
- **SP3** Streams need a whole `stream_order` from 1 to 12; links need a `kind` from `SPINE_LINK_KINDS` (`ridge`, `valley`, `saddle`, `swale`, `escarpment`, `moraine`, `shoreline`, `right-of-way`, `field-margin`, `hedgerow`, `planned`); every line is at least 1 m long; IDs are unique. Anything else is `incomplete`, naming the line.
- **SP4** Provenance is recorded on every section, and derivation never assigns a core class. A corridor that overlaps mapped open water warns.

### `spineNetwork(input, {cuts = true, cutBudget = 20000})`

Tests the draft as a network, on the habitat the connectivity check would see: the derived corridors, any retained habitat already mapped (patches and remnants; features marked `spine: true` are left out, so a saved draft is not counted twice) and the cores.

- **SP5** Links come only from the check's rules: roads sever unless a crossing on that road carries the corridor, open water is removed, the minimum width holds, and stepping stones join habitat only within `gapCrossingM`. The line graph never adds a link.
- **SP6** Single points of failure. A disturbance is a disk `disturbanceWidthM` across, centered anywhere; it may overlap a core but does not remove core habitat. The default is the widest corridor plus 2 m; with stepping stones it is wide enough that a cut across the widest corridor leaves a gap longer than the crossing distance. Centers are tested on a grid whose disks are enlarged to cover every position in between (see `src/robust.mjs` for the argument), so **a pair reported robust cannot be separated by any such disturbance under the check's rules**, stepping stones included. Each place where a tested cut separates linked cores is reported as a zone with the pairs it separates, then re-tested exactly at nominal width and marked `verified` or not. An unverified zone may still be real; it was not confirmed at nominal width.
- **SP7** `rho2` = robust pairs / all core pairs (the HDFM paper's 2-edge connectivity, over cores, tested on the habitat); `pFail1` = share of sections some separating disturbance reaches; `exposedShare` = that length / total length. Robustness is reported only from a completed search: `cuts: false`, or a search that needs more than `cutBudget` cuts, reports it as `null`, never as true. Exposure is `null` when there are no sections to measure.
- **SP8** Loops are counted on the line graph of sections between junctions (lines join where one ends within `junctionSnapM`, default 5 m, at most 10 m). Routes that close only through a core are counted separately as `loopsThroughCores`. Lines that cross mid-way are reported, not joined.

`minWidthM` must be at least 5 m for the network analysis. The disturbance raster spacing is at most 2.5 m, at most a quarter of the minimum width, and at most a sixth of the margin by which the narrowest corridor exceeds the minimum (never below 0.5 m). A pair the raster cannot confirm, because a corridor is close to the minimum width or a stepping-stone gap is close to the crossing distance or a road, is never reported robust.

### `projectSpine(input, {years})`

At each milestone year, the core pairs linked through committed habitat and through habitat at old-growth age. Uses `params.ageAsOfYear`, `oldGrowthAgeYears` with `oldGrowthAgeSource`, and `milestoneYears` (or `years`, up to 12, from `ageAsOfYear` on); `stand_age` or `remnant` on retained features and cores; `consent_year` or `planned_year` on parcels.

- **PJ1** A covered parcel counts from its `consent_year` (no later than `ageAsOfYear`); a parcel without consent counts from its `planned_year`, which must be after `ageAsOfYear`. Consent never lapses in a projection.
- **PJ2** The proposed treatment units apply at every milestone.
- **PJ3** Habitat is at old-growth age in year Y when `stand_age + (Y - ageAsOfYear) >= oldGrowthAgeYears`. A remnant (`remnant: true` with `remnant_source`: never plowed or clear-cut, as recorded) is at old-growth age in every year. Where features overlap, the youngest recorded age applies; a feature with neither an age nor remnant status never counts as old. A core is a link's endpoint whatever its age. Age is reported, never condition.
- **PJ4** Old-age links ⊆ committed links ⊆ links after the plan, in every year.
- **PJ5** Committed and old-age links never decrease from one milestone to the next: no fire, storm or new road is predicted.

### `climateRoutes(input, check?)`

For each core, the coolest core or exit it can reach through links that hold after the plan. Cores and exits carry `temp_c` (with `params.coreTempSource`); an optional warming target `params.climateWarmingC` needs `params.climateSource`.

- **CL1** Routes use only the links the check finds after the plan, chained through linked cores. A supplied check result must match the input's checksum; otherwise it is recomputed, with a warning.
- **CL2** A core without `temp_c` is `unknown`, never `coolest`.
- **CL3** `route`: a reachable core or exit is cooler by at least the target (by any amount with no target); `short`: cooler, by less; `none`: nothing reachable is cooler, although a cooler core or exit exists; `coolest`: every core has a temperature and nothing is cooler. `none`, `short` and `unknown` are flagged.
- **CL4** An exit (`input.exits`: a polygon with `dfm_id`, `temp_c` and an optional `toward`) is where the spine continues into the next landscape. A core reaches an exit by the same rule as another core, on the habitat after the plan. Exits end routes and never pass one on; the result's `coolest` records `exit: true` and `toward` when the coolest place reached is an exit, and `exits` lists the cores each exit links. An exit that links to no core warns.

### `buildOutFrontier(input)`

The parcels without consent whose spine touches the committed spine, with the core pairs each would link if it alone joined now, its spine area and its direction (8-point compass and bearing) from the committed piece it adjoins.

- **FR1** Only features marked `spine: true` count as spine when any are marked. With no committed spine yet, every parcel holding spine is listed, without a direction.
- **FR2** Links each parcel would complete are computed through committed habitat, with the proposed treatment units applied.
- **FR3** Ordered by links completed, then spine area, then ID. Parcel IDs must be unique.

### Performance

The polygon operations dominate. Measured on one core of a cloud build machine:

| Call | Watershed (`fixtures/watershed.mjs`) | Prairie (`fixtures/prairie.mjs`, stepping stones at 100 m) |
|---|---|---|
| Extent | 14 lines, 4 cores, about 2.5 km across | 3 lines, 4 patches, 4 cores, about 2.6 km across |
| `deriveSpine` | under 0.1 s | under 0.1 s |
| `spineNetwork` | about 3 s (about 2,200 cuts); under 1 s with `cuts: false` | about 6 s (about 5,900 cuts); 1.3 s without stepping stones |
| `checkConnectivitySync` on the derived spine | about 2 s | about 1 s; 0.25 s without stepping stones |
| `projectSpine` (5 milestones) | about 4 s | about 1 s |
| `climateRoutes` | about 2 s | about 1.5 s |
| `buildOutFrontier` | about 2.5 s | 0.3 s |

All calls block the thread they run on. A server should run them in a worker thread or a job queue, not in a request handler. Larger landscapes raise the cut count roughly with area; past `cutBudget` the network reports robustness as `null` with a warning. Analyze one watershed at a time.

## Any biome: forest, prairie and flat country

The rules are the same in every biome: a corridor is retained habitat at least the minimum width across, roads sever it unless a crossing is recorded, and nothing reads a biome label. Old growth is not only forest: never-plowed prairie and savanna are old-growth grassland, with species and soils that do not come back on a human timescale once plowed (Veldman et al. 2015). What changes from one biome to another is recorded data, each value with its source.

| | Invariant | Where |
|---|---|---|
| **B1** | No rule reads a biome label. Minimum width, crossing distance, old-growth age and remnant status are recorded parameters or properties, each with its source; naming the biome changes nothing but the checksum. | Every function |
| **B2** | Stepping stones. With `params.gapCrossingM` (0 to 1,000 m, with `gapCrossingSource`), pieces of habitat at least the minimum width across link when the gap between them is at most that distance. A gap may cross cropland or open water but never a road; only a recorded crossing carries a link over a road. `0` or unset gives exactly the results without it, and a longer distance never removes a link. | Check, network, projections, climate, frontier |
| **B3** | Robustness with stepping stones stays sound: a pair reported robust cannot be separated by any single disturbance under the stepping-stone rule (tested against the exact rule at sampled positions, at the equator and at 45° N). | `spineNetwork` |
| **B4** | Flat-land links: `swale`, `escarpment`, `moraine`, `shoreline`, `right-of-way`, `field-margin`, `hedgerow` and `planned`, besides `ridge`, `valley` and `saddle`. A link may carry its own `width_m` with `width_source`, because land use often fixes it (a railroad right-of-way is as wide as the deed). | `deriveSpine` |
| **B5** | Remnants. `remnant: true` with `remnant_source` (never plowed or clear-cut, as recorded) is at old-growth age in every year; restored habitat waits for its recorded age. A remnant that also records `stand_age` warns that remnant status applies. | `projectSpine` |
| **B6** | Upkeep. `prescribed-burn`, `prescribed-grazing`, `late-season-mowing` and `brush-management` are permitted light treatments, on the same terms as forest ones: `corridor_permitted` and a recorded reason. Anything else, such as plowing, removes habitat and is named when it breaks a link. | Check |
| **B7** | Exits. Climate velocity is highest where the land is flat (Loarie et al. 2009), so cooler ground usually lies beyond the landscape. Exits let routes reach it (CL4), and each landscape checks its own stretch of the route. | `climateRoutes` |

Choose `gapCrossingM` from what the species the corridors serve will cross, from a dispersal study, not from the map. Stepping stones are how patch networks hold together where continuous corridors are rare (Saura, Bodin & Fortin 2014; in grassland, Herrera et al. 2017). The prairie fixture shows the effect: four wet-meadow patches, with gaps of up to 90 m between them, link a remnant north of the creek to the rest of the landscape at a crossing distance of 100 m, and not at 80 m; plowing one patch breaks the chain and the check names the unit, while a permitted burn over the same patches passes with a warning.

**References**

- Herrera, L. P., Sabatino, M. C., Jaimes, F. R., & Saura, S. (2017). Landscape connectivity and the role of small habitat patches as stepping stones: an assessment of the grassland biome in South America. *Biodiversity and Conservation* 26(14), 3465–3479.
- Loarie, S. R., et al. (2009). The velocity of climate change. *Nature* 462, 1052–1055.
- Saura, S., Bodin, Ö., & Fortin, M.-J. (2014). Stepping stones are crucial for species' long-distance dispersal and range expansion through habitat networks. *Journal of Applied Ecology* 51(1), 171–182.
- Veldman, J. W., et al. (2015). Toward an old-growth concept for grasslands, savannas, and woodlands. *Frontiers in Ecology and the Environment* 13(3), 154–162.

## Native old growth, not plantations

Only about a third of the world's forest is still primary forest (FAO 2025), and nearly half of planted forest is plantation (FAO 2020): one or two species, even-aged and regularly spaced. A plantation does not become old growth by standing long enough, and neither does a stand made up mostly of species that do not belong to the place. The spine's aim is old growth of the place's own species, so the engine records how habitat was established and what it is made of, and uses that in the old-growth projection. None of it changes whether a link holds today.

| | Invariant | Where |
|---|---|---|
| **O1** | Stand origin. `stand_origin` is `natural` (naturally regenerating), `planted` (planted or seeded, not a plantation) or `plantation` (planted, intensively managed, one or two species, even-aged, regularly spaced), as defined in FAO's Global Forest Resources Assessment, with `origin_source`. Any other value, or no source, is `incomplete`. | `projectSpine` |
| **O2** | A plantation never reaches old-growth age, whatever its `stand_age`; where it overlaps other habitat it carves the old-age area, as a younger stand does (PJ3). A remnant cannot also be a plantation. Planted habitat counts by its recorded age, as restoration plantings of native species can grow old. | `projectSpine` |
| **O3** | Native composition. `native_share` (0 to 1, with `composition_source`) is the share of a feature's cover or basal area in species native to the place. With `params.nativeShareMin` (and `nativeShareSource`), habitat counts at old-growth age only where it records a share at or above the minimum; habitat without a recorded share never counts, like habitat without an age. A never-plowed remnant overrun by introduced grasses does not count either. | `projectSpine` |
| **O4** | Structure is unchanged: origin and composition are never read by the check, the network, climate routes or the frontier. A plantation still carries today's structural link. | Every other function |
| **O5** | Each milestone reports `plantationM2` when any feature records `stand_origin`, and `belowNativeShareM2` when `nativeShareMin` is set: committed habitat outside cores that is plantation, or below the native share. Old-age links still nest and never decrease (PJ4, PJ5). | `projectSpine` |
| **N1** | Native planting. A corridor-permitted `restoration-planting` that lists `species: [{name, native}]` stays permitted only when every species is native and a source for native status is recorded (`native_status_source` on the unit, or `params.nativeStatusSource`). An introduced species, or a list without a source, makes the unit not permitted: it removes habitat like any other unit, and the check names it and says why. | Check, projections, climate routes |
| **N2** | With `params.nativeStatusSource`, every corridor-permitted restoration planting must list its species. | Check, projections, climate routes |
| **N3** | A species list is 1 to 200 entries of `{name, native: true \| false}`; anything else is `incomplete`, naming the unit. | Every function |
| **O6, N4** | Additive. Without these records and parameters, results are identical to 0.2.0 apart from the engine version strings, so checks VergeCommon stored with 0.2.0 still reproduce (`test/vergecommon-contract.test.mjs`). | Every function |

What these records establish is what was recorded, with its source: the engine does not identify species, survey composition or judge whether a planting is suitable for the site. Native status comes from the source you record, such as a state flora or the [USDA PLANTS database](https://plants.usda.gov/), which gives native status by state.

**References**

- FAO (2025). *Global Forest Resources Assessment 2025.* Food and Agriculture Organization of the United Nations, Rome. 4.14 billion hectares of forest, of which at least 1.18 billion are primary forest.
- FAO (2020). *Global Forest Resources Assessment 2020.* Plantation forests cover about 131 million hectares, 45% of planted forests.
- FAO. *Global Forest Resources Assessment 2020: Terms and definitions* (naturally regenerating forest, planted forest, plantation forest, other planted forest). Food and Agriculture Organization of the United Nations, Rome. https://fra-data.fao.org/definitions/fra/2020/en/tad

## Landscape Package

`toLandscapePackage(input, meta)` and `fromLandscapePackage(pkg)` read and write the exchange file defined in [`../dfm-schema`](../dfm-schema/README.md). The optional `streams`, `connectors` and `exits` layers are written only when they hold features and read back only when present, so a package without them, and its checksum, are unchanged from 0.1.0.

## Known limits

- A single road polygon that doubles back on itself (a U-shaped surface) is treated as one piece at a crossing; supply such roads as centerlines or split parts.
- Road-polygon width at a crossing is estimated from the local surface when not recorded; landings and junctions inflate it. Record `width_m` where it matters.
- Turf.js measures on a sphere (within about 0.5% of ellipsoidal distances).
- The spine network tests one disturbance at a time, shaped as a disk. Two disturbances at once, or one long disturbance such as a new road across two corridors, are not tested.
- Stepping stones never cross a road, however narrow, unless a crossing is recorded on it. Where the species do cross a road, record the crossing.
- Around the end of a road, a stepping-stone gap is measured as it bends around the end, in steps that each cut the corner slightly: there a join can be made up to about a quarter of the minimum width beyond the crossing distance. Elsewhere a join can be missed within about 0.1% of the distance, never made beyond it.
- Projections take ages and join years as recorded and predict no loss. Climate routes use the temperatures supplied for cores and exits; they do not model climate or species, and a route through an exit is checked here only as far as the exit.

## Development

```sh
npm ci
npm test
```

Tests use fictional fixtures placed at the equator (`test/origin.test.mjs` adds a two-reserve corridor for stand origin, native share and native planting): a woodlot (`fixtures/woodlot.mjs`), a watershed (`fixtures/watershed.mjs`, the illustration on dendriticforest.com, at 5 m per drawing unit) and a flat prairie (`fixtures/prairie.mjs`: a creek, a railroad right-of-way, a moraine, a chain of wet-meadow stepping stones and an exit north). Each test names the DFM Build Spec invariant it checks. Soundness tests compare every "robust" claim with an independent exact check at sampled disturbance positions, with and without stepping stones. Turf.js computes buffers in a local azimuthal projection around each feature, which is accurate for woodlot and watershed-scale extents (a few kilometers); larger landscapes need a projected-CRS engine.
