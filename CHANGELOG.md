# Releases

## Web application 0.3.0 — 2026-09-13

Adds the MIT-licensed planning workspace under `web/`: persistent owner-scoped projects, source records, GeoJSON import and drawing, reproducible geometric scenarios, map overlays, reports and self-hosting instructions.

Follow-up improvements group comparisons by saved input revision and calculation method, expose width assumptions, provide spreadsheet-safe CSV export, require forest data for new assessments, and add a bounded offline analysis command with a fictional fixture.

Public documentation now distinguishes software behavior, synthetic research, scientific validation and field outcomes. An unseeded test asserting composite-entropy optimality was replaced with a deterministic counterexample that verifies the narrower minimum-length guarantee. Experimental Python algorithms are not connected to the website's planning results; their unresolved defects are listed in `docs/RESEARCH_STATUS.md`.

A proposed landscape-assessment pilot is described without enabling billing, customer accounts, team access or public signup. The website's existing private audience is unchanged. PostgreSQL support is an optional storage adapter; no large-scale PostGIS processing is claimed.

Validation: application tests and offline fixture; Python implementation tests; Worker build. These checks do not establish ecological benefit. Docker and PostgreSQL integration still need runtime validation.
