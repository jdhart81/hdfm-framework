# Releases

## dfm-core 0.2.0 — unreleased

Adds the old-growth spine to `packages/dfm-core`: the dendritic network of retained forest that follows a watershed's streams, valleys and ridges, mapped whole and built woodlot by woodlot.

- `deriveSpine`: drafts corridor polygons from stream lines (width by Strahler order, never narrower for a larger river, never below the minimum) and ridge, valley or saddle links, each with its width source.
- `spineNetwork`: tests the draft as a network under the connectivity check's own rules: core links, loops, road severing at each crossing place, and single points of failure. A pair reported robust cannot be separated by one disturbance of the stated width placed anywhere (a covering-grid search, checked against an independent exact test); each failure zone is re-tested at nominal width and marked verified or not. Reports rho2 (robust pairs over all pairs), pFail1 and exposed length. Robustness is `null`, never true, when the search is skipped or exceeds its cut budget.
- `projectSpine`: committed and old-growth-age links at milestone years from consent and planned join years and recorded stand ages. Nested (old-age ⊆ committed ⊆ after the plan) and never decreasing over time; the youngest recorded age applies where features overlap.
- `climateRoutes`: for each core, the coolest core it can reach through links that hold after the plan; flags cores with no route cool enough for a recorded warming target, or no temperature.
- `buildOutFrontier`: the woodlots whose land would extend the committed spine, with the links each would complete and its direction.
- Landscape Package: optional `streams`, `connectors` and `exits` layers and spine, age, remnant, consent-year and temperature properties and parameters. Packages without them, and their checksums, are unchanged. New `spine-results.schema.json` in `packages/dfm-schema` (0.2.0).
- The connectivity check's results are unchanged without stepping stones, apart from the engine version (`dfm-connectivity-0.2.0`); malformed layer types now return `incomplete` instead of throwing.

Any biome: the same rules run in forest, prairie, savanna, wetland and flat country. No rule reads a biome label; what differs is recorded data, each value with its source.

- Stepping stones: `params.gapCrossingM` (0 to 1,000 m, with `gapCrossingSource`) links pieces of habitat at least the minimum width across when the gap between them is at most that distance. A gap never crosses a road without a recorded crossing. Applies in the check, the network, projections, climate routes and the frontier. Unset or 0 gives exactly the results without it, and a longer distance never removes a link.
- Robustness with stepping stones stays sound: a pair reported robust cannot be separated by one disturbance under the stepping-stone rule, checked against the exact rule at sampled positions near and far from roads, at the equator and at 45° N. The default disturbance is wide enough to open a gap longer than the crossing distance.
- Flat-land links: connector kinds `swale`, `escarpment`, `moraine`, `shoreline`, `right-of-way`, `field-margin`, `hedgerow` and `planned`, besides ridge, valley and saddle. A link may record its own width, with a source.
- Remnants: `remnant: true` with `remnant_source` (never plowed or clear-cut, as recorded) counts at old-growth age in every projection year; restored habitat waits for its recorded age.
- Upkeep: `prescribed-burn`, `prescribed-grazing`, `late-season-mowing` and `brush-management` are permitted light treatments, on the same terms as forest ones.
- Exits: where the spine continues into the next landscape, with the temperature of the ground it leads to. In flat country cooler ground usually lies beyond the landscape, so climate routes can end at an exit; an exit that links no core warns.
- Spine network fixes from independent review: the disturbance raster no longer joins habitat across a pinch between raster rows (a rare case that could report a pair robust when it was not); distances are corrected for the map frame's distortion away from its center latitude; retained habitat already mapped (patches, remnants) counts in the network, and saved spine features are not counted twice. Spine and outlook engine versions are `dfm-spine-0.2.0` and `dfm-outlook-0.2.0`.
- 116 tests in all, on fictional fixtures: the woodlot, the watershed drawn on dendriticforest.com, a flat prairie with a creek, a railroad right-of-way, a moraine, wet-meadow stepping stones and an exit north, and a synthetic VergeCommon plan.
- Used by VergeCommon v0.10.0, which vendors a tarball byte-identical to `npm pack` of this package. A contract test (`test/vergecommon-contract.test.mjs`) reads the check package a steward downloads from a VergeCommon plan and must reproduce the result and input checksum VergeCommon stored, so a change here cannot silently break plans stored there.

## Site — unreleased

dendriticforest.com: the home-page map shows the spine mapped along a watershed's rivers, built out woodlot by woodlot in four directions, aging toward old growth and keeping a route to cooler ground as climate lines move upslope. One orchestrated time-lapse that respects reduced motion, four static stages, and a section on the method. A section on prairie, savanna and flat country covers stepping stones, links on flat land, never-plowed ground, fire and grazing, and exits to the next landscape. The data format page documents the new layers, parameters, link kinds and treatments, and a test keeps it in step with the schema. The VergeCommon links go to its woodland page, which says whether woodland projects are open there, and the Unbroken Woods pilot offer goes to that page's first-woodlot contact.

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
