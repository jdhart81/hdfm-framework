# Forester design partners and the plan appendix (proposal)

Status: proposal for review. Nothing here is built yet.

## Why foresters

Consulting foresters already write most private forest plans. In Vermont, land enrolled in the Use Value Appraisal (Current Use) program needs a forester-written management plan that is renewed every ten years, and the program covers roughly 16,000 forest parcels (Vermont Legislature testimony, 2023). Across the United States, NRCS cost-shares forest management plans written by registered Technical Service Providers. Every one of those plans needs a clear account of streams, riparian buffers, roads and crossings. That is exactly what the DFM web application already screens.

One forester serves many landowners. A tool that saves a forester time on every plan travels with that forester from woodlot to woodlot, and each plan shows the landowner where the corridor runs onto the neighbours' land.

## Design partner programme (no cash)

- Three consulting foresters, one planning season.
- Each runs one real plan through the DFM workspace with the landowner's permission and tells us where it falls short.
- In return: free use, priority fixes, and named credit in the release notes and `CITATION.cff` contributors once they agree.
- Nothing about the landowner, the parcel or its location is published without the landowner's written consent.

## The plan appendix export

A printable section a forester can attach to a plan, built from one saved assessment record:

| Part | Contents | Source in the current app |
| --- | --- | --- |
| Map page | Boundary, waterways, buffers, roads, retained forest, crossing candidates, scale bar, north arrow, data dates | Existing map export |
| Riparian table | Buffer width used, buffer area, forest within the buffer after road-surface and open-water exclusions | Assessment metrics |
| Crossings | Each road and waterway crossing candidate, numbered to match the map | `crossingCandidates` |
| Data sources | Each layer's source, date and reuse terms, and every data warning | Source records |
| Method note | Method identifier (for example `dfm-geometric-0.3.0`), parameters, and the plain statement that this is geometric screening, not a regulatory determination | Assessment record |
| Neighbour note | Whether a retained corridor leaves the boundary, and where, so the landowner can talk to the neighbour. No neighbour data is shown | New: boundary intersection of retained areas |
| Reproduce | A line saying the open-source tool and the exported record reproduce these numbers | Assessment record |

### Invariants

1. The appendix is produced only from a saved assessment record, so the same record always gives the same appendix.
2. It never states compliance with any program's rules; the forester decides what the plan says.
3. It names DFM and its version in the method note, so the method travels with the plan.
4. It shows nothing outside the plan's boundary except the points where a corridor leaves it.

### Acceptance checks

- The fictional watershed in `web/examples/` produces an appendix whose numbers match the QUICKSTART metrics exactly.
- Missing source dates or reuse terms appear as warnings in the appendix, never silently dropped.
- A test fails if the appendix shows any geometry outside the boundary.

## What to measure

Plans exported by foresters other than Viridis, repeat use by the same forester, and landowner requests to involve a neighbour. Stars and downloads are leading signals only.

## Hand-off to VergeCommon

When a corridor leaves the boundary, the landowner can start or join a woodland co-op on [VergeCommon](https://vergecommon.com), which runs the same `dfm-core` engine, so the neighbours along the corridor can organize and consent parcel by parcel.
