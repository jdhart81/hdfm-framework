# Releases

## dfm-core 0.2.0 — unreleased

Adds the old-growth spine to `packages/dfm-core`: the dendritic network of retained forest that follows a watershed's streams, valleys and ridges, mapped whole and built woodlot by woodlot.

- `deriveSpine`: drafts corridor polygons from stream lines (width by Strahler order, never narrower for a larger river, never below the minimum) and ridge, valley or saddle links, each with its width source.
- `spineNetwork`: tests the draft as a network under the connectivity check's own rules: core links, loops, road severing at each crossing place, and single points of failure. A pair reported robust cannot be separated by one disturbance of the stated width placed anywhere (a covering-grid search, checked against an independent exact test); each failure zone is re-tested at nominal width and marked verified or not. Reports rho2 (robust pairs over all pairs), pFail1 and exposed length. Robustness is `null`, never true, when the search is skipped or exceeds its cut budget.
- `projectSpine`: committed and old-growth-age links at milestone years from consent and planned join years and recorded stand ages. Nested (old-age ⊆ committed ⊆ after the plan) and never decreasing over time; the youngest recorded age applies where features overlap.
- `climateRoutes`: for each core, the coolest core it can reach through links that hold after the plan; flags cores with no route cool enough for a recorded warming target, or no temperature.
- `buildOutFrontier`: the woodlots whose land would extend the committed spine, with the links each would complete and its direction.
- Landscape Package: optional `streams` and `connectors` layers and spine, age, consent-year and temperature properties and parameters. Packages without them, and their checksums, are unchanged. New `spine-results.schema.json` in `packages/dfm-schema` (0.2.0).
- The connectivity check's results are unchanged; malformed layer types now return `incomplete` instead of throwing.
- 53 new tests (83 in all) on fictional fixtures, including the watershed drawn on dendriticforest.com.

## Site — unreleased

dendriticforest.com: the home-page map shows the spine mapped along a watershed's rivers, built out woodlot by woodlot in four directions, aging toward old growth and keeping a route to cooler ground as climate lines move upslope. One orchestrated time-lapse that respects reduced motion, four static stages, and a section on the method. The data format page documents the new layers and parameters, and a test keeps it in step with the schema.

## Web application 0.6.0 — unreleased

Corridor connectivity check in the workspace: core-area, road-crossing and treatment-unit layers; `POST /api/projects/:id/connectivity`; red lost-corridor overlay; DFM Landscape Package download for VergeCommon. Uses the vendored `@viridis/dfm-core`. Includes the phase 0 web hardening (#37).

## dfm-core 0.1.0 — unreleased

Adds `packages/dfm-core`, the shared analysis package for the DFM workspace and VergeCommon woodland projects, and `packages/dfm-schema`.

- Corridor connectivity check: compares the current state with proposed treatment units and fails when a link between core areas is lost or narrowed below the minimum width, naming the responsible units. Unpermitted treatment overlap with retained habitat also fails. Roads sever corridors except at recorded crossings. Reports pinch points, consent coverage (committed vs proposed habitat), engine version and input checksum.
- DFM Landscape Package v1 and connectivity-result JSON Schemas, with round-trip tests.
- 28 tests on fictional fixtures, each tied to a DFM Build Spec invariant or an independent-review finding.

Structural connectivity only; no claims about species movement, genetics or old-growth condition.

## Web application 0.3.0 — 2026-09-13

Adds the MIT-licensed planning workspace under `web/`: persistent owner-scoped projects, source records, GeoJSON import and drawing, reproducible geometric scenarios, map overlays, reports and self-hosting instructions.

Follow-up improvements group comparisons by saved input revision and calculation method, expose width assumptions, provide spreadsheet-safe CSV export, require forest data for new assessments, and add a bounded offline analysis command with a fictional fixture.

Public documentation now distinguishes software behavior, synthetic research, scientific validation and field outcomes. An unseeded test asserting composite-entropy optimality was replaced with a deterministic counterexample that verifies the narrower minimum-length guarantee. Experimental Python algorithms are not connected to the website's planning results; their unresolved defects are listed in `docs/RESEARCH_STATUS.md`.

A proposed landscape-assessment pilot is described without enabling billing, customer accounts, team access or public signup. The website's existing private audience is unchanged. PostgreSQL support is an optional storage adapter; no large-scale PostGIS processing is claimed.

Validation: application tests and offline fixture; Python implementation tests; Worker build. These checks do not establish ecological benefit. Docker and PostgreSQL integration still need runtime validation.
