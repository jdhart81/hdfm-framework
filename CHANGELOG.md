# Releases

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
