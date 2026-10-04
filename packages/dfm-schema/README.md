# @viridis/dfm-schema

JSON Schemas (draft 2020-12) for data exchanged between DFM tools and VergeCommon.

- `landscape-package.schema.json`: **DFM Landscape Package v1**. One JSON object with WGS84 (EPSG:4326) GeoJSON features grouped by layer (`boundary`, `parcels`, `coreAreas`, `retained`, `roads`, `water`, `crossings`, `treatments`, and the optional spine layers `streams` and `connectors`), parameters with a recorded source for every width, age and temperature that changes a result, and a source record per layer. Lengths in meters, areas in square meters.
- `connectivity-result.schema.json`: the result of `checkConnectivity` in `@viridis/dfm-core`, including status, lost links with responsible units, pinch points, consent coverage, the engine version and an input checksum.
- `spine-results.schema.json`: the results of the old-growth spine functions in `@viridis/dfm-core`, one `$defs` entry each: `derivation` (`deriveSpine`), `network` (`spineNetwork`), `projection` (`projectSpine`), `climate` (`climateRoutes`) and `frontier` (`buildOutFrontier`). Validate a result against `spine-results.schema.json#/$defs/<name>`.

## Version 0.2.0: additive

The spine additions do not change `dfm_package` (still `1.0`):

- New optional layers `streams` (lines with `dfm_id` and `stream_order`, the Strahler order from 1 to 12) and `connectors` (lines with `dfm_id` and `kind`: ridge, valley or saddle). Writers include them only when they hold features, so packages without a spine are byte-for-byte what 0.1.0 wrote, and their checksums are unchanged.
- New optional properties: `temp_c` on core areas; `stand_age` and the spine provenance fields (`spine`, `origin`, `source_id`, `stream_order`, `width_m`, `width_source`) on retained features; `consent_year` and `planned_year` on parcels.
- New optional parameters: `spineWidthByOrderM`, `spineWidthSource`, `connectorWidthM`, `connectorWidthSource`, `junctionSnapM`, `disturbanceWidthM`, `ageAsOfYear`, `oldGrowthAgeYears`, `oldGrowthAgeSource`, `milestoneYears`, `coreTempSource`, `climateWarmingC` and `climateSource`.

Readers that take known layers by name, such as `fromLandscapePackage` in dfm-core 0.1.0, ignore the spine layers, and the connectivity check never reads them. A package that has spine layers must be validated with this 0.2.0 schema: the 0.1.0 schema allows no unknown layers.

Versioning: a breaking change to any schema increments `dfm_package` and the package's major version. Readers must reject a package version they do not support.
