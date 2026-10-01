<div align="center"><img src="Project logo.png" alt="HDFM project" width="280"/></div>

# Dendritic Forest Management

[![DFM checks](https://github.com/jdhart81/hdfm-framework/actions/workflows/dfm-checks.yml/badge.svg)](https://github.com/jdhart81/hdfm-framework/actions/workflows/dfm-checks.yml)

**Open-source landscape planning, beginning with existing roads and waterways.**

Viridis DFM helps land managers compare forest-retention options with traceable inputs, explicit assumptions, maps and portable reports. It is currently a geometric screening pilot. It does not establish ecological viability, regulatory compliance or an optimal forest plan.

## Start with the working application

The [web application](web/README.md) includes:

- Draw or import roads, waterways, planning boundaries, forest, open-water polygons and management units.
- Record data sources, dates, reuse terms and field notes.
- Save and reopen projects, with protection against conflicting saves.
- Compare waterway buffer and proposed retention scenarios using the same saved input revision and calculation method.
- Calculate forest within retention proposals after modeled road-surface and supplied open-water exclusions.
- Export maps, comparison spreadsheets, printable reports and reproducible assessment records.
- Run the same bounded analysis offline from a command line.

```sh
git clone https://github.com/jdhart81/hdfm-framework.git
cd hdfm-framework/web
npm ci
npm test
npm run dev
```

Open **http://127.0.0.1:4173/map**. Local projects are stored in `.data/dfm.sqlite`; no hosted account is required. A fictional watershed is included for exploration. See [self-hosting and deployment instructions](web/README.md) for Docker, PostgreSQL and authentication requirements.

The [Viridis-hosted pilot](https://dendritic-forest-management.jdhart.chatgpt.site/map) currently requires owner access; it is not a public signup service. Self-hosting remains available independently.

## The foundation

Waterways organize riparian study zones. Existing roads identify access and crossing questions. Forest, terrain, boundaries, management objectives and field evidence determine how these networks can inform a plan. A road surface is not retained forest, and a waterway centerline is not automatically a wildlife corridor.

See the [system mapping](docs/SYSTEM_MAP.md), [release notes](CHANGELOG.md) and [pilot assessment scope](docs/PILOT_ASSESSMENT.md).

## Research toolkit

The existing Python package remains in [`hdfm-framework/`](hdfm-framework/README.md). It explores patch networks, entropy scores, widths, genetics and synthetic climate scenarios. It is separate from the working web application's geometric analysis.

**The research toolkit is experimental.** A minimum spanning tree minimizes summed edge length on its supplied graph, not necessarily the implemented composite entropy score. Synthetic comparisons and passing software tests do not prove ecological superiority. Known issues remain in area budgets, solver-result handling and population interpretation. Consult [research status](docs/RESEARCH_STATUS.md) before using research outputs.

```sh
cd hdfm-framework
python -m pip install -e .
python -m pytest -q
```

The research package and web application have independent versions. Web release 0.5.0 does not certify or change the research package's scientific claims.

## Open source and the Viridis service

The software is MIT licensed. Users can inspect methods, self-host and export their work. Viridis's proposed paid service adds data preparation, managed project operation, assessment delivery and support. Pricing and scope should be agreed for each initial pilot. Billing, subscriptions, teams, public signup and large-scale PostGIS processing are not enabled in this release.

Code adoption and GitHub activity are not evidence of customer demand or ecological benefit. Our next milestone is a real landscape assessment reviewed by its intended user.

## Contribute

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Useful contributions include reproducible bug reports, geometry fixtures, source documentation, GIS interoperability and independent scientific review. Keep tests of implementation separate from claims about conservation outcomes.

## License and attribution

Application code uses the [MIT license](LICENSE); third-party notices remain with vendored libraries. Dataset licenses and project branding are separate. Research materials describe the HDFM agenda of Justin Hart / Viridis and remain available in the repository; this release does not independently validate manuscript claims.
