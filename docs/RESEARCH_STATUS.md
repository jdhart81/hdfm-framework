# Research status and claim boundaries

Updated 2026-10-06. This document supersedes older completion percentages and claims of global optimality or production readiness. The Python toolkit remains experimental and is not invoked by the web application's screening service.

| Topic | Established implementation behavior | Unresolved interpretation or defect |
|---|---|---|
| MST construction | Finds a tree minimizing summed edge weight on the supplied graph | Does not minimize the composite entropy objective; Euclidean edges are not terrain-routed corridors |
| Entropy comparisons | Computes a weighted score on synthetic networks | The cycle penalty favors trees by construction; a score improvement is not independent evidence of biological benefit |
| Width budgets | Patch hectares are converted to m² before comparison with corridor length × width (fixed 2026-10-03; regression tests) | Synthetic landscapes at default guild widths are often infeasible under a 20–30% budget; that is now reported, not hidden |
| Optimizers | Width solvers check feasibility first, verify returned widths independently, and return them on the result; `success=False` with a message on any failure (fixed 2026-10-03) | Topology search remains a local heuristic; the composite objective is still unvalidated |
| Movement | Relative weights are normalized | Normalization can conceal uniformly poor movement success |
| Population estimates | The current formula returns a numeric proxy | Uniform population scaling cancels; do not use it for population viability decisions |
| Climate | Synthetic scenarios and scheduling interfaces exist | No independently validated regional ecological forecast is supplied |
| Roads and waterways | Explicit typed layers in the web application | The Python patch-graph package does not implement the full road/water foundation |

## Reproducible limitation

For `SyntheticLandscape(n_patches=4, random_seed=0)`, the MST has lower total length than the star with edges `(0,1), (0,2), (0,3)`, yet the star has lower composite entropy. Under the reviewed environment the scores were approximately 1.4002 and 1.0377 respectively.

The former unseeded test asserting that an MST should beat a random tree in composite entropy could pass or fail depending on the random draw. It is replaced by a deterministic regression documenting the actual distinction. This corrects the test's claim; it does not fix or validate the ecological objective.

## Evidence levels

1. Software checks establish specific implementation behavior.
2. Reproducible synthetic experiments establish behavior under stated assumptions.
3. Independent scientific review and empirical comparison are needed for ecological interpretation.
4. Field assessment is needed for operational decisions.

The web application currently operates at the first two levels for geometric screening. Its source checks establish metadata completeness, not the truth of a dataset. Road width is assumed; missing open-water polygons remain an explicit warning. Forest input is required for new assessments.

## Priorities

Rederive the population estimator; distinguish absolute movement from normalized distribution; compare alternatives at equal management objectives and budgets; then seek independent review on a real landscape. Do not connect the experimental algorithms to customer-facing recommendations before these questions are resolved.

## Genetic connectivity: the open question

DFM's aim is to keep working woods working while old growth of each place's own species regrows in a dendritic pattern, linked so that its populations can exchange genes again. What the software establishes today is the structure that aim needs, and nothing more:

| | Established now | Not established |
|---|---|---|
| Corridor check (`packages/dfm-core`) | Mapped retained habitat at least the minimum width links the same cores after a plan as before, under recorded roads, crossings, open water and stepping-stone gaps | That animals or plants move through it, or that genes flow |
| Old-growth projection | Which links run through habitat at or above a recorded old-growth age, never counting plantations, and, with a minimum native share, never counting habitat recorded below it (dfm-core 0.3.0) | Old-growth condition, which needs field evidence; whether recorded origin and composition are correct |
| Research toolkit (`hdfm-framework/hdfm/genetics.py`) | An island-model calculation that returns a numeric proxy | An effective population size usable for decisions; see the population-estimates row above |

Gene flow along a spine can only be shown in the field. This is the test we propose; it has not started.

1. **Species.** Choose, with field partners, at least one forest-floor species of low mobility that depends on older, closed-canopy habitat, and for prairie landscapes a plant or invertebrate of never-plowed remnants. A species that disperses widely shows little genetic structure at woodlot scale and tests nothing.
2. **Sites.** In a pilot landscape, sample sites inside the mapped spine and in comparable habitat outside it, at matched straight-line distances.
3. **Baseline.** Genotype the samples and measure genetic differentiation between sites. Compare models in which differentiation follows straight-line distance, distance through all forest, and distance through the spine. Methods follow landscape genetics (Manel et al. 2003, *Trends in Ecology & Evolution* 18: 189-197).
4. **Repeat.** Genetic structure lags habitat change by generations, so a single survey describes the landscape as it was, not the spine's effect. Repeat the sampling as the spine is committed and ages, on a schedule set by the species' generation time.
5. **Report either way.** If differentiation does not follow the spine, that is a result to publish, and a reason to change the method rather than the claim.

Field data from private land is published only with the owner's consent. Until such results exist, DFM's material describes reconnecting old-growth genetics as its aim, never as an outcome. Contributors with landscape-genetics experience: see the open issues labelled `research`.

