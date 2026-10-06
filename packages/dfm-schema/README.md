# @viridis/dfm-schema

JSON Schemas (draft 2020-12) for data exchanged between DFM tools and VergeCommon.

- `landscape-package.schema.json`: **DFM Landscape Package v1**. One JSON object with WGS84 (EPSG:4326) GeoJSON features grouped by layer (`boundary`, `parcels`, `coreAreas`, `retained`, `roads`, `water`, `crossings`, `treatments`, and the optional spine layers `streams`, `connectors` and `exits`), parameters with a recorded source for every width, distance, age and temperature that changes a result, and a source record per layer. Lengths in meters, areas in square meters.
- `connectivity-result.schema.json`: the result of `checkConnectivity` in `@viridis/dfm-core`, including status, lost links with responsible units, pinch points, consent coverage, the engine version and an input checksum.
- `spine-results.schema.json`: the results of the old-growth spine functions in `@viridis/dfm-core`, one `$defs` entry each: `derivation` (`deriveSpine`), `network` (`spineNetwork`), `projection` (`projectSpine`), `climate` (`climateRoutes`) and `frontier` (`buildOutFrontier`). Validate a result against `spine-results.schema.json#/$defs/<name>`.

## Version 0.3.0: additive

Stand origin, native composition and native planting, for old growth of the place's own species. `dfm_package` is still `1.0`, and a package without these records is unchanged, with the same checksum.

- New optional properties on core areas and retained features:
  - `stand_origin`: `natural` (naturally regenerating), `planted` (planted or seeded, not a plantation) or `plantation` (planted, intensively managed, one or two species, even-aged, regularly spaced), as defined in FAO's Global Forest Resources Assessment 2020. Needs `origin_source`. A remnant cannot also be a plantation.
  - `native_share`: the share (0 to 1) of the feature's cover or basal area in species native to the place. Needs `composition_source`.
- New optional properties on treatment units: `species`, a list of 1 to 200 `{name, native}` entries for a restoration planting, and `native_status_source`.
- New optional parameters: `nativeShareMin` (above 0, at most 1), which needs `nativeShareSource`, and `nativeStatusSource`.
- `spine-results.schema.json`: projection milestones may report `plantationM2` and `belowNativeShareM2`.

A 0.2.0 engine ignores all of these. It projects a plantation as old growth once its recorded age passes the threshold, and it permits a corridor planting of introduced species; read such a package with dfm-core 0.3.0 or later.

## Version 0.2.0: additive

The spine and biome additions do not change `dfm_package` (still `1.0`):

- New optional layers:
  - `streams`: lines with `dfm_id` and `stream_order`, the Strahler order from 1 to 12.
  - `connectors`: lines with `dfm_id` and `kind`. Landforms: `ridge`, `valley`, `saddle`, `swale`, `escarpment`, `moraine`, `shoreline`. Land use: `right-of-way`, `field-margin`, `hedgerow`. `planned` for a link drawn where neither carries one. A connector may record its own `width_m`, which then needs `width_source`.
  - `exits`: polygons where the spine continues into the next landscape, with `dfm_id`, `temp_c` (the ground it leads to) and an optional `toward`.

  Writers include these layers only when they hold features, so packages without them are byte-for-byte what 0.1.0 wrote, and their checksums are unchanged.
- New optional properties:
  - core areas: `temp_c`, `stand_age`, and `remnant` (never plowed or clear-cut, as recorded), which needs `remnant_source`;
  - retained features: `stand_age`, `remnant` with `remnant_source`, and the spine provenance fields (`spine`, `origin`, `source_id`, `stream_order`, `width_m`, `width_source`);
  - parcels: `consent_year` and `planned_year`.
- New optional parameters: `spineWidthByOrderM`, `spineWidthSource`, `connectorWidthM`, `connectorWidthSource`, `junctionSnapM`, `disturbanceWidthM`, `ageAsOfYear`, `oldGrowthAgeYears`, `oldGrowthAgeSource`, `milestoneYears`, `coreTempSource`, `climateWarmingC`, `climateSource`, and `gapCrossingM` (stepping stones, 0 to 1,000 m), which needs `gapCrossingSource` when above 0.
- Four more permitted light treatments for fire- and grazing-dependent habitat: `prescribed-burn`, `prescribed-grazing`, `late-season-mowing` and `brush-management`.

Readers that take known layers by name, such as `fromLandscapePackage` in dfm-core 0.1.0, ignore the spine layers, and the connectivity check never reads them. A package that has spine layers or one of the new treatments must be validated with this 0.2.0 schema: the 0.1.0 schema rejects unknown layers and treatment intensities. It accepts the new properties and parameters without checking them, so validate with 0.2.0 to catch a remnant without its source or a stepping-stone distance without its source. A 0.1.0 engine ignores `gapCrossingM` and reports fewer links than a 0.2.0 engine for the same package, and it removes habitat under the new upkeep treatments, so read such a package with dfm-core 0.2.0 or later.

Versioning: a breaking change to any schema increments `dfm_package` and the package's major version. Readers must reject a package version they do not support.
