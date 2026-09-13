# Contributing

Use a focused branch and pull request. Describe the concrete input, observed behavior, expected behavior and reproducible example. Do not upload private landowner datasets or credentials.

## Working application

From `web/`: run `npm ci`, `npm test`, and `npm run build`. Geometry changes need a hand-checkable fixture covering units, overlaps and exclusions. Preserve owner-scoped queries, revision conflict protection, old scenario records and immutable database migrations. Do not place a deployment-specific Site ID or secrets in the public checkout.

## Research toolkit

From `hdfm-framework/`: install the package and run `python -m pytest -q`. Read `docs/RESEARCH_STATUS.md` at the repository root before changing scientific claims. A passing test is not field validation. Use deterministic cases and distinguish minimum length, objective score and ecological outcome.

## Releases

The public repository contains the reusable application. The Viridis deployment checkout has its own hosting identity. Promote reviewed `web/` source to that checkout while preserving its `.openai/hosting.json`, then build and deploy. Never copy a developer's `.data`, `.env`, credentials, Git metadata or `node_modules` into a release.
