# Viridis DFM

Open-source landscape planning workspace, built around existing roads and waterways. This is a geometric screening pilot, not a validated ecological optimizer or regulatory assessment.

## What works

- Draw or import a WGS84 planning boundary, roads, waterways, forest, open-water polygons, management units and proposed retention connections.
- Inspect or remove individual working features; record source, date, reuse permission and accuracy notes per layer.
- Save and reopen private projects. Server ownership checks isolate records. Revision checks prevent overwriting another save.
- Compare named, immutable scenarios from saved inputs. Each contains an input snapshot, source records, parameters, method version, results and limitations.
- Buffer waterways, clip to the planning boundary, union overlapping areas, intersect with existing forest, and exclude modeled road surfaces and supplied open-water polygons.
- Review crossing candidates and retention overlap with management units. These are screening flags, not verified culverts, access rights or ecological passages.
- Download map GeoJSON, derived scenario GeoJSON, a full reproducibility record, or an HTML report that prints to PDF.

## Run locally

Requires Node 22 or newer.

```sh
npm ci
npm test
npm run dev
```

Open http://127.0.0.1:4173/map. Local mode has one local owner, binds to loopback, and stores data in `.data/dfm.sqlite`. It never trusts caller-provided Sites identity headers. Stop the process before copying this file for backup; restore it with the process stopped. Do not run multiple processes against this SQLite file.

Alternatively `docker compose up --build` provides the same loopback-only workspace and a durable named volume. Docker execution has not been verified on this machine.

## Hosted private website

The current Sites deployment serves a bundled Worker with a D1 database. Authentication uses the platform's authenticated user ID, and every API query is scoped to that owner. Mutations require a same-origin JSON request. The original informational website and mapping UI are preserved. Sites applies the committed Drizzle migrations before deployment. Never expose the Worker through a route that bypasses the Sites authenticated gateway or trusts browser-supplied identity headers.

Register your own Sites project and put its identity in `.openai/hosting.json`; the public template intentionally contains no Viridis project ID. Run `npm run db:generate` after a schema change, inspect the SQL, then build with the Sites build script. Do not edit migrations already deployed.

### Schema-tool dependency maintenance

Stable `drizzle-kit` 0.31.11 still declares the retired `@esbuild-kit/esm-loader`, whose `core-utils` dependency pins vulnerable esbuild 0.18.20. The scoped npm override makes only `core-utils` use the application's pinned esbuild 0.28.2. It keeps the existing loader entrypoints and stable Drizzle versions. Remove this override and its legacy-loader compatibility test when a stable Drizzle release removes that dependency; the 1.0 release candidate requires a different migration format.

`npm test` checks the legacy loader's CommonJS and ESM transforms, TypeScript imports through its Node loader, and actual Drizzle generation through an ESM TypeScript config. Generation runs in a temporary fixture: an unchanged schema must leave every migration byte intact, and an added table must produce one executable SQLite migration without rewriting the old SQL or snapshots.

## Independent service

The same Node API supports an optional PostgreSQL database: apply `db/postgres.sql` once, set `DATABASE_URL`, and start the server. It stores validated GeoJSON snapshots in text columns; it does not yet run spatial queries in PostGIS. The PostgreSQL adapter is supplied but has not been integration-tested against a running database here.

For external single-owner hosting, set `PUBLIC_ORIGIN` to the exact HTTPS origin, `DFM_PASSWORD` to a strong secret (at least 12 characters), and `HOST` as required. The server refuses to start if either is missing or malformed. Ten failed sign-ins from one client address lock that address out for 15 minutes; behind a reverse proxy every client shares the proxy's address, so an attacker can also lock out the owner for that window. Put the service behind a TLS reverse proxy. Browser sign-in uses username `dfm`. Never expose unauthenticated local mode. This single-owner option is not a substitute for multi-tenant SaaS authentication.

The frontend uses MapLibre GL JS 6.9.0. Analysis and storage remain separate from rendering. PostGIS spatial processing, GDAL raster ingestion and terrain analysis are future integrations.

## API

All routes require trusted identity. JSON responses are private and not cached.

- `GET /api/account`: service state and limits.
- `GET /api/projects`: owner's saved projects.
- `POST /api/projects`: name, data (layer-keyed GeoJSON features), sources.
- `GET /api/projects/:id`: project and revision.
- `PUT /api/projects/:id`: name, data, sources and expected revision; conflicts return 409.
- `GET /api/projects/:id/validation`: missing inputs/source fields and evidence warnings.
- `GET /api/projects/:id/scenarios`: immutable scenario records.
- `POST /api/projects/:id/scenarios`: name, expected revision, waterWidth (each side, meters), roadWidth (full surface width, meters).
- `DELETE /api/projects/:id`: permanently delete a project and all its scenarios.
- `DELETE /api/projects/:id/scenarios/:scenarioId`: permanently delete one scenario.

Invalid input returns 400 with a message. Storage failures return 503 and the change is not saved; a failed local disk write is rolled back and later saves continue. Unexpected server errors return 500 without internal detail.

Imports support 2,000 features and 20,000 coordinates, under 5 degrees across and within ±85 degrees latitude. Saved payloads are limited to 2 MB. Hosted analysis is limited to 300 features and 6,000 coordinates, plus at most 100 waterway and boundary features. Simplify larger datasets before import. Files can be up to 5 MB for local exploration, but must fit the smaller saving limit to persist.

## Revenue and release status

No customer billing, checkout, subscriptions, team invitations, public signup, external automatic data ingestion, guaranteed backup service or usage-based invoicing is enabled. A private pilot can support preparing and delivering landscape assessments. Enabling commercial service needs a chosen account/payment provider, agreed offer, public access decision and verified customer workflow. No revenue or customer validation is claimed.

Production dependencies passed `npm audit --omit=dev` on 2026-09-13. Automated tests cover ownership isolation, cross-origin rejection, stale-save conflicts, input validation, reproducible scenario storage, area units, clipping and overlap/exclusion calculations. Earlier browser testing covered saving a fictional project and viewing scenario results. Release 0.4.0 adds automated GIS checks and successful USGS/NASA tile delivery checks; the new renderer and physical GPS workflow have not yet been tested interactively on a field device.

## License and data

Application code is MIT licensed; vendored MapLibre, Leaflet and Turf retain their own notices in `public/vendor`. The illustrative geometry is fictional. Dataset permissions are independent of code licensing. Optional OpenStreetMap tiles use the attribution shown in the map; production-scale tile service is not provisioned here. User-supplied project branding is retained separately and is not offered as a trademark license.


## Release 0.3.0 — reproducible assessment handoffs

Comparison tables are grouped by saved revision and calculation method, with explicit waterway and road-width assumptions and CSV export. Spreadsheet formula prefixes in user labels are escaped. New scenarios require a forest layer, preventing missing forest data from being reported as measured zero retention. Existing saved scenarios are preserved.

Run a bounded assessment without the browser or hosted account:

```sh
npm run analyze -- examples/fictional-watershed.geojson assessment.json 50 6
```

Arguments are input path, new output path, waterway buffer distance each side (m), and assumed full road width (m). The command also accepts a downloaded full scenario record and recalculates from its input snapshot. It refuses to overwrite an existing file. Limits and scientific caveats are the same as the hosted calculation. The sample contains fictional geometry.

The assessment-pilot page describes a proposed scoped service and opens an email draft. It does not accept payments or submit an order. The public source repository is https://github.com/jdhart81/hdfm-framework/tree/main/web.


## Release 0.4.0 — connected GIS and field observations

- MapLibre replaces the elementary renderer; start with an empty map and connected imagery, then open a saved project or load explicitly fictional example data.
- USGS aerial imagery and topographic maps provide detailed U.S. context. NASA MODIS true color provides dated regional satellite context with a date selector. These are remotely delivered imagery, not a real-time satellite video feed; acquisition age, resolution, clouds and coverage vary. Overzoom does not add detail. Background imagery never silently becomes an analytical road, waterway or forest layer.
- Coordinate navigation accepts latitude, longitude in WGS84. Device location is off until requested. Locate, follow and stop controls show horizontal accuracy and observation time, reject stale fixes, and require an explicit action to add a field point. Saving the project uploads that point; location tracking alone does not. Browser location requires HTTPS (or localhost) and user permission, and is not survey-grade GPS.
- ECAD evidence export preserves original IDs, layer and source records in notes. It targets ECAD's existing manual GeoJSON import for points, lines and single-ring polygons. Unsupported multipart geometry, holes and oversized files are rejected without silent simplification. ECAD assigns new feature IDs; this is not shared storage or bidirectional synchronization.

Imagery comes from [USGS](https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer) and [NASA GIBS](https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/1.0.0/WMTSCapabilities.xml); provider attribution remains on the map. Imagery and tile terms are separate from the software license. No bulk/offline imagery download is implemented. Road and waterway geometry must still be imported or drawn with recorded provenance.


## Release 0.5.0 — dimensioned corridor deliverables

Each saved scenario now has a **Corridor plan** action. Choose roadside corridor width, background and (for NASA) date, generate the preview, then download the printable HTML plan, corridor GeoJSON and companion plan record. Open the HTML file and choose Print / Save as PDF; its map image is embedded and works offline after export. It includes numbered corridors, approximate scale bar, north arrow, attribution, source records and a schedule of centerline lengths, nominal widths and section areas.

Waterway width comes from the saved scenario and is measured each side of the centerline. Roadside width is a new deliverable parameter measured outward from each edge of the assumed road surface. Roads and mapped open water are subtracted. Centerline lengths are clipped to the boundary and holes; they are horizontal lengths, not slope-adjusted ground distances. Polygon areas are unioned for the total. Per-section areas may overlap. Widths are nominal design inputs and may be narrowed by exclusions. The plan is a separate geometric proposal, not a changed saved retention assessment, measured forest acreage, or an optimized ecological network. Keep its downloaded record to preserve the chosen parameters.

The image registration option accepts north-up EPSG:3857 PNG/JPEG images under 12 MB / 16 million pixels, with known outer bounds supplied in WGS84 west/south/east/north order. The image must cover the property. It is held locally in the browser and embedded in the printable export; it is not uploaded to project storage. The companion record stores its filename, source/registration notes and bounds, not image bytes. Unreferenced screenshots, rotated imagery, other CRSs and GeoTIFF ingestion require preprocessing. A contour map is visual context; no DEM drainage or contour-following routing is calculated.

Sheets support at most 120 clipped centerline sections. Inspect label overlap, image alignment and source rights before delivery. USGS backgrounds are intended for U.S. coverage; NASA imagery is regional/coarse. If a provider fails, export is blocked instead of silently dropping the background. Browser validation confirmed generation of the fictional four-section example; geometric tests cover boundary clipping, holes, exclusions, width interpretation and escaped report content.


## Release 0.6.0 — corridor connectivity check

Three new layers: **core areas** (old-growth candidates, riparian cores, reserves), **road crossings** and **proposed treatment units**. **Check corridors** runs the `@viridis/dfm-core` connectivity check on the latest saved project: do the core areas stay linked by retained habitat at least the minimum width, before and after the proposed treatment units?

- Retained habitat = proposed forest connections plus the riparian buffer at the distance set above (when a boundary and waterways exist). Road surfaces (assumed width) and open-water polygons are removed. A road keeps a corridor only at a recorded crossing.
- The result is **pass**, **fail** (a link is lost, a core is cleared, or a treatment overlaps retained habitat without a recorded light-treatment permission) or **incomplete** (missing inputs, or no source recorded for the minimum width). Lost corridor is drawn in red, with the units that caused it named.
- Drawn cores start as `old-growth-candidate`, drawn crossings as `assumed` passage (with a warning) and drawn treatments with `unrecorded` intensity (treated as removing habitat). Import GeoJSON with `core_class`, `passage`, `intensity`, `corridor_permitted` and `reason` properties to record them.
- **Download Landscape Package** exports the same inputs as a DFM Landscape Package v1 file. A VergeCommon steward can upload it as woodland corridor layers, and `checkConnectivity(fromLandscapePackage(pkg))` reproduces the result and its input checksum.
- API: `POST /api/projects/:id/connectivity` with `revision`, `waterWidth`, `roadWidth`, `minWidthM`, `minWidthSource`. Returns `{result, package, revision}`; nothing is stored.

The check is structural connectivity only. It does not establish species movement, genetic viability or old-growth condition. `@viridis/dfm-core` is vendored as `vendor/viridis-dfm-core-0.1.0.tgz` from `packages/dfm-core`; CI fails if the two differ.
