# @viridis/dfm-schema

JSON Schemas (draft 2020-12) for data exchanged between DFM tools and VergeCommon.

- `landscape-package.schema.json`: **DFM Landscape Package v1**. One JSON object with WGS84 (EPSG:4326) GeoJSON features grouped by layer (`boundary`, `parcels`, `coreAreas`, `retained`, `roads`, `water`, `crossings`, `treatments`), connectivity parameters with a recorded minimum-width source, and a source record per layer. Lengths in meters, areas in square meters.
- `connectivity-result.schema.json`: the result of `checkConnectivity` in `@viridis/dfm-core`, including status, lost links with responsible units, pinch points, consent coverage, the engine version and an input checksum.

Versioning: a breaking change to either schema increments `dfm_package` and the package's major version. Readers must reject a package version they do not support.
