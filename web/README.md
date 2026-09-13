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

Requires Node 20 or newer.

```sh
npm ci
npm test
npm run dev
```

Open http://127.0.0.1:4173/map. Local mode has one local owner, binds to loopback, and stores data in `.data/dfm.sqlite`. It never trusts caller-provided Sites identity headers. Stop the process before copying this file for backup; restore it with the process stopped. Do not run multiple processes against this SQLite file.

Alternatively `docker compose up --build` provides the same loopback-only workspace and a durable named volume. Docker execution has not been verified on this machine.

## Hosted private website

The current Sites deployment serves a bundled Worker with a D1 database. Authentication uses the platform's authenticated user ID, and every API query is scoped to that owner. Mutations require a same-origin JSON request. The original informational website and mapping UI are preserved. Sites applies the committed Drizzle migrations before deployment. Never expose the Worker through a route that bypasses the Sites authenticated gateway or trusts browser-supplied identity headers.

The public `.openai/hosting.json` contains logical bindings only. Register your own Site before publishing; the Viridis deployment identity is kept outside this public source. Run `npm run db:generate` after a schema change, inspect the SQL, then build with the Sites build script. Do not edit migrations already deployed.

## Independent service

The same Node API supports an optional PostgreSQL database: apply `db/postgres.sql` once, set `DATABASE_URL`, and start the server. It stores validated GeoJSON snapshots in text columns; it does not yet run spatial queries in PostGIS. The PostgreSQL adapter is supplied but has not been integration-tested against a running database here.

For external single-owner hosting, set `PUBLIC_ORIGIN` to the exact HTTPS origin, `DFM_PASSWORD` to a strong secret, and `HOST` as required. Put the service behind a TLS reverse proxy. Browser sign-in uses username `dfm`. Never expose unauthenticated local mode. This single-owner option is not a substitute for multi-tenant SaaS authentication.

The frontend remains Leaflet for continuity. The analysis module and storage interface are separate from the map renderer. Moving to OpenLayers, Python and PostGIS for large datasets remains a future backend/renderer migration, not a capability claimed by this release.

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

Imports support 2,000 features and 20,000 coordinates, under 5 degrees across and within ±85 degrees latitude. Saved payloads are limited to 2 MB. Hosted analysis is limited to 300 features and 6,000 coordinates, plus at most 100 waterway and boundary features. Simplify larger datasets before import. Files can be up to 5 MB for local exploration, but must fit the smaller saving limit to persist.

## Revenue and release status

No customer billing, checkout, subscriptions, team invitations, public signup, external automatic data ingestion, guaranteed backup service or usage-based invoicing is enabled. A private pilot can support preparing and delivering landscape assessments. Enabling commercial service needs a chosen account/payment provider, agreed offer, public access decision and verified customer workflow. No revenue or customer validation is claimed.

Production dependencies passed `npm audit --omit=dev` on 2026-09-13. Automated tests cover ownership isolation, cross-origin rejection, stale-save conflicts, input validation, reproducible scenario storage, area units, clipping and overlap/exclusion calculations. Browser testing covers saving a fictional project and viewing scenario results.

## License and data

Application code is MIT licensed; vendored Leaflet and Turf retain their own notices in `public/vendor`. The illustrative geometry is fictional. Dataset permissions are independent of code licensing. Optional OpenStreetMap tiles use the attribution shown in the map; production-scale tile service is not provisioned here. User-supplied project branding is retained separately and is not offered as a trademark license.


## Release 0.3.0 — reproducible assessment handoffs

Comparison tables are grouped by saved revision and calculation method, with explicit waterway and road-width assumptions and CSV export. Spreadsheet formula prefixes in user labels are escaped. New scenarios require a forest layer, preventing missing forest data from being reported as measured zero retention. Existing saved scenarios are preserved.

Run a bounded assessment without the browser or hosted account:

```sh
npm run analyze -- examples/fictional-watershed.geojson assessment.json 50 6
```

Arguments are input path, new output path, waterway buffer distance each side (m), and assumed full road width (m). The command also accepts a downloaded full scenario record and recalculates from its input snapshot. It refuses to overwrite an existing file. Limits and scientific caveats are the same as the hosted calculation. The sample contains fictional geometry.

The assessment-pilot page describes a proposed scoped service and opens an email draft. It does not accept payments or submit an order. The public source repository is https://github.com/jdhart81/hdfm-framework/tree/main/web.
