# HDFM manuscript: corrections to the topology results

Draft, 2026-10-30. Applies to *Hierarchical Dendritic Forest Management:
Mathematical Framework for Entropy-Minimizing Corridor Networks with Dimensional
Consistency and Robustness Analysis* (J. Hart), file
`HDFM_FINAL_With_Figures.docx` in this repository and
`HDFM_FINAL_With_Figures.pdf` on Zenodo (record 23112436, v0.5.0). Status of
the manuscript: working paper, not peer reviewed. All results in it are from
synthetic landscapes; none is field-validated.

## What was wrong

The section "Network Topology Results" reports

> H_mov(MST) = 2.28 ± 0.04 nats (mean ± SE). [...] Gabriel +27.6%
> (H = 2.91 ± 0.06), Delaunay +37.3% (H = 3.13 ± 0.07), k-NN +29.8%
> (H = 2.96 ± 0.05), RNG +21.5% (H = 2.77 ± 0.04). MST+3Loops [...] +2.6%.

No version of this repository produces these numbers, at the tag linked from
Zenodo (v0.5.0) or at any earlier or later commit. The same figures appeared as
"expected results" in the first upload (commit `27aca06`), before the comparison
code had been run, and `hdfm-framework/REPOSITORY_SUMMARY.md` says they "match
the values in your perspectives paper": they were copied from the manuscript,
not produced by the code.

They also cannot be right as defined. The manuscript describes H_mov as the
average movement uncertainty per patch. A patch with d corridors has at most
ln d nats of choice entropy, so for any 15-patch tree (28 corridor ends) the
average is at most ln(28/15) = 0.62 nats (Jensen's inequality). An MST value of
2.28 nats, and the Delaunay value of 3.13 nats (which would need more than 22
corridors per patch on average, out of a possible 14), are impossible under that
definition. Under the literal formula written in the manuscript (a sum over
patches rather than an average) the code gives 6.35 nats, not 2.28.

The same unreproduced values recur in the abstract and conclusions ("22–37%
lower entropy"; the reported +21.5% to +37.3% would in any case be 18–27%
lower), in Figure 1 to Figure 4 captions, and in Table 3.

## What the code gives

Command, from `hdfm-framework/`:

```sh
MPLBACKEND=Agg python examples/reproduce_topology_table.py --tree-checks --out-dir results
```

Runtime about 100 s (numpy 2.5.4, scipy 1.18.1, networkx 3.7, Python 3.13).
Output: `hdfm-framework/results/topology_table.md`, with per-landscape values in
the CSV files beside it. Landscape i uses seed 42 + i, i = 0..99.

Configuration "manuscript-stated" uses the design in "Experimental Design":
100 km × 100 km, 15 patches, areas U(500, 5000) ha, quality Beta(2, 2),
α = 0.1 km⁻¹, β = 0.15. Local choice entropy is the per-patch Shannon entropy of
the normalized neighbour weights exp(−α d_ij) q_j, averaged over patches. Widths
are uniform; with a common width, φ(w) cancels in the normalization, so these
entropies do not depend on width.

| Network | Local choice entropy, nats (mean ± SE) | vs MST | Entropy rate, nats | Total length, km | Uniform width allowed by β, m |
|---|---|---|---|---|---|
| MST | 0.423 ± 0.004 | reference | 0.424 ± 0.007 | 261 ± 2 | 237 ± 3 |
| Gabriel | 0.712 ± 0.009 | +68.3% | 0.721 ± 0.012 | 517 ± 8 | 122 ± 2 |
| Delaunay | 1.023 ± 0.008 | +141.7% | 1.029 ± 0.010 | 1076 ± 12 | 58 ± 1 |
| k-NN (k = 3) | 0.981 ± 0.007 | +131.9% | 0.984 ± 0.009 | 706 ± 9 | 88 ± 2 |
| RNG | 0.490 ± 0.007 | +15.9% | 0.492 ± 0.009 | 314 ± 5 | 199 ± 4 |
| MST + 3 loops | 0.443 ± 0.005 | +4.7% | 0.443 ± 0.007 | 477 ± 6 | 130 ± 2 |

All differences from MST: paired t-test p < 1e-10. MST is strictly lower than
Gabriel, Delaunay, k-NN and MST + 3 loops in 100 of 100 landscapes, and lower
than the RNG in 88 (in the other 12 the RNG is the MST).

The package's `movement_entropy()` adds a corridor-length term
0.1 × mean edge length / dispersal scale, which is not in the manuscript's
formula; with it, MST = 0.609 ± 0.004 and Gabriel = 0.952 ± 0.010 (+56.3%). The
composite score `calculate_entropy()` (with the loop penalty F, which counts
edges beyond n − 1 and so favours trees by construction) gives MST 1.418 and
Gabriel +525%. The package's default 10 km landscapes (the configuration in
`examples/synthetic_landscape_validation.py`) give the composite table in
`QUICKSTART.md` (MST 1.321, Gabriel +559.8%) and local choice entropy MST 0.316,
Gabriel +62.0%.

Running the package's width solver on each network (`--optimize-widths`,
manuscript-stated configuration, about 4 minutes) does not restore the reported
numbers either: MST 0.609, Gabriel +51.9%, Delaunay +17.7%. The solver lowers
the entropy of dense networks by setting corridors to the minimum width, where
movement success is zero and those corridors drop out; 77 of 600 solves failed
(60 Delaunay, 16 k-NN, 1 Gabriel).

### Is the MST the minimum-entropy tree?

No. The MST beats these denser networks largely because it has fewer corridors
per patch (mean degree 1.87 against 2.87 for Gabriel), and fewer choices mean
less choice entropy. Among spanning trees on the same landscapes, a star has
lower local choice entropy than the MST in 100 of 100 landscapes (best star
0.053 against MST 0.423 nats), and 99.5% of 20,000 random spanning trees
(200 per landscape) have lower entropy than the MST. The MST does minimize the
surrogate Σᵢ Σⱼ pᵢⱼ dᵢⱼ in every comparison made (0 of 20,000 random trees
lower), so the surrogate does not track the entropy, and a quantity in km cannot
bound one in nats as the Proposition states. These checks are the
`--tree-checks` section of the same output; random trees are minimum spanning
trees of uniform random weights, seeded 1000 + landscape index.

### Table 3 (loops)

Same configuration, loops added by `add_strategic_loops(criterion='betweenness')`;
extra area assumes one common width; P_fail(1) is exact (bridges / corridors):

| Loops | Local choice entropy (nats) | ΔH | ρ₂ | Extra area | P_fail(1) |
|---|---|---|---|---|---|
| 0 | 0.423 | 0 | 0.00 | 0 | 1.00 |
| 1 | 0.430 | +1.6% | 0.35 | +27.6% | 0.41 |
| 2 | 0.436 | +3.0% | 0.42 | +56.0% | 0.34 |
| 3 | 0.443 | +4.7% | 0.48 | +83.2% | 0.27 |
| 5 | 0.458 | +8.4% | 0.55 | +137.2% | 0.21 |

The manuscript reports an area overhead of +1.8% to +5.1%. Adding the shortest
available non-tree corridors instead (`criterion='shortest'`) costs +5.6% area
for one loop but gives ρ₂ = 0.03 and ΔH = +14%. Neither criterion reproduces
"ρ₂ = 0.58 at +3.4% area".

## Corrected statements

Replace the passages below in the next manuscript version.

**Abstract**, sentence beginning "Dendritic networks achieve 22–37% lower
entropy":

> In 100 synthetic 15-patch landscapes, minimum spanning trees had lower local
> choice entropy than Gabriel graphs, relative neighbourhood graphs, k-nearest-
> neighbour graphs and Delaunay triangulations (0.42 nats against 0.49–1.02
> nats; reproducible with `examples/reproduce_topology_table.py`). This
> difference largely reflects the smaller number of corridors per patch in a
> tree. The minimum spanning tree is not the minimum-entropy tree: stars and
> most random spanning trees score lower.

**Network Topology Results**, first paragraph:

> Across 100 synthetic landscapes (seeds 42–141), local choice entropy H_mov,
> averaged over patches, was 0.423 ± 0.004 nats (mean ± SE) for the MST. The
> alternatives were higher (paired t-tests, all p < 1e-10): RNG 0.490 ± 0.007
> (+15.9%), Gabriel 0.712 ± 0.009 (+68.3%), k-NN (k = 3) 0.981 ± 0.007
> (+131.9%), Delaunay 1.023 ± 0.008 (+141.7%). Adding three loops to the MST
> raised it to 0.443 ± 0.005 (+4.7%). Entropy rate with π ∝ area × quality gave
> the same ordering (MST 0.424 ± 0.007; Gabriel +70.2%; Delaunay +142.9%). The
> MST had the shortest total length, 261 ± 2 km, against RNG 314 km (+20.6%),
> Gabriel 517 km (+98.5%), k-NN 706 km (+170.6%) and Delaunay 1,076 km
> (+312.9%); at β = 0.15 this allows a uniform width of 237 m for the MST
> against 122 m for Gabriel and 58 m for Delaunay. Because a tree has the fewest
> corridors per patch, a lower H_mov for the MST is expected from its edge count
> alone and is not evidence that it is optimal: among spanning trees, the MST
> scored higher than a star in every landscape and higher than 99.5% of random
> spanning trees.

**Figure 1 caption.** The per-network values are from a "representative"
landscape that cannot be identified. Regenerate the figure from landscape
seed 42 and use its values: MST 14 corridors, H_mov 0.434 nats, uniform width
257 m; Gabriel 21, 0.668; Delaunay 35, 1.057; k-NN 29, 1.016; RNG 17, 0.517;
MST + 3 loops 17, 0.454, ρ₂ = 0.63.

**Figures 2–4 and Table 3.** Every value that rests on H_mov(MST) = 2.28 nats
(Fig. 2A, Fig. 3A, Fig. 4A, Table 3) is withdrawn. Replace Table 3 with the
table above and state the loop-selection criterion. Replace "adding 3–5
strategic loops increases 2-edge-connectivity from 0% to 45–68% while entropy
cost remains under 4.5% and area overhead under 5%" with:

> Adding 1 to 5 loops chosen to reduce maximum edge betweenness raised ρ₂ from
> 0 to 0.35–0.55 and lowered P_fail(1) from 1.00 to 0.41–0.21, at an entropy
> cost of +1.6% to +8.4% and an extra corridor length (area, at a common width)
> of +28% to +137%.

The "Algorithm consistently identifies global optimum" claim (Fig. 4A) and the
3.2% median surrogate gap (Discussion and Appendix A.3) are withdrawn: no code
in the repository computes them, and the MST is not the entropy optimum among
trees.

**Conclusions**, sentence beginning "Computational experiments confirm":

> On synthetic landscapes, minimum spanning trees had 16–142% lower local choice
> entropy than the denser alternatives tested, mainly because trees have fewer
> corridors per patch. Adding a few loops buys redundancy (ρ₂ up to 0.55 with
> five loops) at a substantial cost in corridor length. These are properties of
> the model on synthetic data; they have not been tested in the field.
