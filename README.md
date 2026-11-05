<div align="center"><img src="Project logo.png" alt="HDFM project" width="280"/></div>

# Dendritic Forest Management

[![DFM checks](https://github.com/jdhart81/hdfm-framework/actions/workflows/dfm-checks.yml/badge.svg)](https://github.com/jdhart81/hdfm-framework/actions/workflows/dfm-checks.yml)

**Open-source landscape planning, beginning with existing roads and waterways.**

**Status: software pilot and working research, not peer reviewed and not field-validated.**

**Aim: keep the woods working, and reconnect old-growth genetics in a dendritic pattern.** Old growth of each place's own species, regrown along its rivers, valleys and ridges and linked through working woodlots, farms and ranches so that its populations can exchange genes again. The [dfm-core engine](packages/dfm-core/README.md) checks the structure that aim needs: corridors that hold through every harvest plan, and an old-growth spine that never counts a plantation, or habitat recorded as mostly introduced species, as old growth. Whether genes then flow along the spine is an open field question ([research status](docs/RESEARCH_STATUS.md#genetic-connectivity-the-open-question)). Public site: [dendriticforest.com](https://dendriticforest.com).

Viridis DFM helps land managers compare forest-retention options with traceable inputs, explicit assumptions, maps and portable reports. It is currently a geometric screening pilot. It does not establish ecological viability, regulatory compliance or an optimal forest plan.

## Start with the working application

To reproduce the worked examples step by step, with the output you should see, follow [QUICKSTART.md](QUICKSTART.md). To cite the software, use [CITATION.cff](CITATION.cff).

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

## Part of the Viridis open commons

DFM is one of five open projects that build on each other. Each one is useful alone; together they carry a piece of land from a map to a verified result.

| Project | What it does | Where |
| --- | --- | --- |
| Dendritic Forest Management (DFM) | Starts from existing roads and waterways and checks whether retained forest stays connected as harvests are planned. MIT. | [hdfm-framework](https://github.com/jdhart81/hdfm-framework) · [dendriticforest.com](https://dendriticforest.com) |
| VergeCommon | Lets neighbours run a local conservation co-op: private mapping, parcel-by-parcel consent, evidence and shared decisions. Runs DFM inside its woodland projects. AGPL-3.0. | [verge-common](https://github.com/jdhart81/verge-common) · [vergecommon.com](https://vergecommon.com) |
| Global Soil Intelligence Project (GSIP) | Builds an open soil spectral reference library, with phone and web capture. MIT code, ODbL data. | [gsip](https://github.com/global-soil-intelligence/gsip) |
| ViridisOS | Turns verified results into recomputable conservation modules and certificates anyone can check. Apache-2.0. | [viridisos](https://github.com/jdhart81/viridisos) |
| Viridis Research | The public, machine-checked research record behind the methods. CC BY 4.0. | [viridis-research](https://github.com/jdhart81/viridis-research) |

How they connect: DFM finds where a corridor runs. The corridor almost always crosses property lines, so VergeCommon gives the neighbours along it a private place to organize and consent. GSIP adds soil evidence from the same ground. ViridisOS lets anyone recompute what was claimed, and Viridis Research holds the proofs those checks rest on. Viridis Conservation ([viridisconservation.com](https://viridisconservation.com)) offers paid data preparation, assessments and support for teams that want help.

Built something with one of these? Add the badge and tell us in an issue, so we can link back:

```markdown
[![Part of the Viridis open commons](https://img.shields.io/badge/Viridis-open%20commons-2e7d32)](https://github.com/jdhart81/hdfm-framework#part-of-the-viridis-open-commons)
```

## Contribute

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Useful contributions include reproducible bug reports, geometry fixtures, source documentation, GIS interoperability and independent scientific review: foresters testing the check on a woodlot they know, ecologists and geneticists helping design the [field test of gene flow](docs/RESEARCH_STATUS.md#genetic-connectivity-the-open-question), and botanists building per-region native-status sources for planting records. Keep tests of implementation separate from claims about conservation outcomes.

## Part of the Viridis open-source commons

Taking a retention option to the people who share the land? [VergeCommon](https://github.com/jdhart81/verge-common) handles consent-based mapping and group decisions, and [GSIP](https://github.com/global-soil-intelligence/gsip) adds soil observations. The handoff between them is manual today.

This project is one of a connected set of open tools from Viridis. Each stands on its own; together they run from field evidence to checked decisions.

**Conservation commons**

- [VergeCommon](https://github.com/jdhart81/verge-common): community conservation co-ops with consent-based land mapping, field evidence and group decisions
- **Dendritic Forest Management (DFM)** (this repository): compare forest-retention options around roads and waterways, with traceable inputs and portable reports
- [Global Soil Intelligence Project (GSIP)](https://github.com/global-soil-intelligence/gsip): contribute soil observations and explore open soil maps with explicit uncertainty

**Verifiable science**

- [Viridis Canon](https://github.com/jdhart81/viridis-canon): machine-checked Lean 4 mathematics for ecological decisions, with a DOI for every result
- [ViridisOS](https://github.com/jdhart81/viridisos): open reference runtime for DOI-backed decision modules and recomputable results
- [Viridis Research](https://github.com/jdhart81/viridis-research): the nightly public research record, each claim labeled by what a verifier actually checked
- [ProofLens](https://github.com/jdhart81/prooflens): see what a Lean statement means before reading the proof

**Agent tools**

- [Viridis Agent Fleet](https://github.com/jdhart81/viridis-agent-fleet): open-source MCP agents with deterministic, auditable cores
- [EnergyAI MCP](https://github.com/jdhart81/energyai-mcp): energy-planning estimates and incentive sources for AI agents

Browse every project on the [Viridis profile](https://github.com/jdhart81). If this one is useful to you, a star helps other people find it.

## License and attribution

Application code uses the [MIT license](LICENSE); third-party notices remain with vendored libraries. Dataset licenses and project branding are separate. Research materials describe the HDFM agenda of Justin Hart / Viridis and remain available in the repository; this release does not independently validate manuscript claims.

Archived releases are on Zenodo: concept DOI [10.5281/zenodo.23112435](https://doi.org/10.5281/zenodo.23112435), which resolves to the latest release (v0.5.0, git tag `v0.5.0`). Cite with [CITATION.cff](CITATION.cff).

**AI use.** Much of the code and documentation was written with AI coding assistants, mainly Claude (Anthropic) and some OpenAI Codex; those commits carry a `Co-Authored-By` trailer or a `claude/` or `codex/` branch name in the history. Justin D. Hart (Viridis LLC, ORCID [0009-0008-3082-2482](https://orcid.org/0009-0008-3082-2482)) reviews the changes and is the accountable author. No AI tool is listed as an author.
