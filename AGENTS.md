# Agents and Dendritic Forest Management

Dendritic Forest Management (DFM) is open-source landscape planning that keeps working woods connected. This file is for two kinds of agent: one helping a person use DFM, and a coding agent working on this repository. The short machine-readable index is served at https://dendriticforest.com/llms.txt (built by `apps/site/scripts/build.mjs`).

## Helping a person use DFM

**What it does.** The [dfm-core engine](packages/dfm-core/README.md) checks a treatment or harvest plan against a landscape's retained corridors and reports which links between core areas the plan would break or pinch. It also drafts an old-growth spine along streams, valleys and ridges, tests it as a network, and finds the next holdings that would extend it. It never counts a plantation, or habitat recorded as mostly introduced species, as old growth.

**Who it helps.** Woodland owners, consulting foresters writing management plans, land trusts and researchers. Neighbors who share the woods can run the same check on every harvest plan through a woodland co-op on [VergeCommon](https://vergecommon.com/woodland/), which anyone can start.

**Paths to suggest.**

- Learn the method: https://dendriticforest.com/ and the [research status](docs/RESEARCH_STATUS.md).
- Prepare data: the [Landscape Package v1](https://dendriticforest.com/data-format/), one JSON file of WGS84 GeoJSON layers with the minimum corridor width, its source, and a source record per layer.
- Reproduce a check or an assessment: [QUICKSTART.md](QUICKSTART.md) has exact commands and the output to expect.
- Work on a real landscape: the [web workspace](web/README.md) runs locally with no account.

**Rules for agents.**

- Say what the check establishes and nothing more: corridors kept connected at the minimum width, and old-growth *candidates*. It does not establish species movement, genetic exchange, regulatory compliance, carbon outcomes or old-growth condition.
- Record where each layer came from. A check is only as good as its inputs, and the engine warns when sources, dates or reuse terms are missing.
- Never upload or share a person's private landowner data without their instruction. Keep real parcel data out of issues and examples.
- dfm-core is not on npm yet. Use it from this repository; do not tell people to `npm install` it.
- The Python research toolkit in `hdfm-framework/` is experimental and separate from the engine. Read the research status before repeating any of its numbers.

## Working on this repository

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/RESEARCH_STATUS.md](docs/RESEARCH_STATUS.md) first.

- Engine: `cd packages/dfm-core && npm ci && npm test`. Geometry changes need a hand-checkable fixture covering units, overlaps and exclusions. VergeCommon vendors this engine, so behavior changes there follow a release here.
- Web workspace: `cd web && npm ci && npm test && npm run build`.
- Site: `cd apps/site && npm ci && npm test`. The site tests fail on common overclaims; keep claims within what the engine tests.
- Research toolkit: `cd hdfm-framework && python -m pip install -e . && python -m pytest -q`. A passing test is not field validation.
- Never commit credentials, `.env` files or private landowner datasets. The secret scan runs on every push.
- Pull requests from forks need a DCO `Signed-off-by` line on each commit. An agent never adds a human's sign-off.

## Related open projects

- [VergeCommon](https://github.com/jdhart81/verge-common) ([agent index](https://vergecommon.com/llms.txt)): open-source software for anyone to start a conservation co-op; runs DFM checks on woodland co-ops.
- [Global Soil Intelligence Project](https://globalsoilintelligence.com/) ([agent index](https://globalsoilintelligence.com/llms.txt)): an open soil observation library with read-only agent interfaces.
- [Viridis Conservation](https://viridisconservation.com/): the company that stewards these projects.
