# Research status and claim boundaries

Updated 2026-09-13. This document supersedes older completion percentages and claims of global optimality or production readiness. The Python toolkit remains experimental and is not invoked by the web application's screening service.

| Topic | Established implementation behavior | Unresolved interpretation or defect |
|---|---|---|
| MST construction | Finds a tree minimizing summed edge weight on the supplied graph | Does not minimize the composite entropy objective; Euclidean edges are not terrain-routed corridors |
| Entropy comparisons | Computes a weighted score on synthetic networks | The cycle penalty favors trees by construction; a score improvement is not independent evidence of biological benefit |
| Width budgets | Length and width enter the allocation calculation | Patch areas are documented in hectares but some budgets compare them directly with square meters |
| Optimizers | Solvers return candidate widths | Failed/infeasible solver outputs and preservation of width results require repair |
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

Correct area units and solver failure handling; rederive the population estimator; distinguish absolute movement from normalized distribution; compare alternatives at equal management objectives and budgets; then seek independent review on a real landscape. Do not connect the experimental algorithms to customer-facing recommendations before these questions are resolved.
