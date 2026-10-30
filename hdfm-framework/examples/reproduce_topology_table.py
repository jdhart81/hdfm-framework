#!/usr/bin/env python3
"""
Regenerate the HDFM manuscript's topology comparison from this code.

The manuscript (HDFM_FINAL_With_Figures, Zenodo record 23112436, v0.5.0),
section "Network Topology Results", reports MST H_mov = 2.28 +/- 0.04 nats and
Gabriel +27.6%, Delaunay +37.3%, k-NN +29.8%, RNG +21.5%, MST+3 loops +2.6%.
No version of this repository produces those numbers (see
docs/HDFM_MANUSCRIPT_ERRATUM.md at the repository root). This script prints what the code does produce,
under two configurations, so the corrected table can be regenerated:

  code-default       The landscapes and kernel the package uses by default and
                     that examples/synthetic_landscape_validation.py runs:
                     10 km x 10 km, patch areas U(10, 100) ha, quality
                     N(0.7, 0.15) clipped to [0, 1], alpha = 2 per 1000 m
                     (2 per km).

  manuscript-stated  The design the manuscript states in "Experimental Design":
                     100 km x 100 km, patch areas U(500, 5000) ha, quality
                     Beta(2, 2), alpha = 0.1 per km (alpha = 1 per 10 km
                     dispersal scale), beta = 0.15, gamma = 0.05 per m,
                     w_min = 50 m.

Both use 100 landscapes of 15 patches, landscape i seeded with seed + i.

Columns:
  H_mov        The package's movement_entropy(): per-patch Shannon entropy of
               the normalised neighbour weights, averaged over patches, plus a
               corridor-length term 0.1 * mean edge length / dispersal scale.
  H_shannon    The same average without the corridor-length term.
  H_sum        The manuscript's written formula, -sum_i sum_j p_ij log p_ij
               (a sum over patches, not an average). Equals 15 x H_shannon.
  H_rate       calculate_entropy_rate(): patch entropies weighted by
               pi_i = A_i q_i / sum_k A_k q_k.
  H_total      calculate_entropy(): H_mov + C + F + D with unit weights. F
               counts edges beyond n - 1, so any graph with loops is penalised
               by construction.
  length_km    Total corridor length.
  w_bar_m      The uniform width the area budget allows (manuscript config
               only): beta * total patch area / total length, capped at 500 m.

Widths: with every corridor at the same width, phi(w) is a common factor and
cancels when neighbour weights are normalised, so H_mov, H_shannon, H_sum and
H_rate do not depend on width. --optimize-widths additionally runs the
package's SLSQP width solver on each network (slow; see the audit note).

Usage (from hdfm-framework/):
    MPLBACKEND=Agg python examples/reproduce_topology_table.py
    MPLBACKEND=Agg python examples/reproduce_topology_table.py --tree-checks --out-dir results

Runtime: about 40 s for both configurations on a laptop, about 100 s with
--tree-checks, and about 4 minutes more with --optimize-widths.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hdfm.landscape import Landscape, Patch, SyntheticLandscape  # noqa: E402
from hdfm.network import (  # noqa: E402
    build_dendritic_network,
    build_gabriel_graph,
    build_delaunay_triangulation,
    build_knn_network,
    build_threshold_network,
)
from hdfm.entropy import (  # noqa: E402
    calculate_entropy,
    calculate_entropy_rate,
    movement_entropy,
)
from hdfm.robustness import add_strategic_loops, calculate_two_edge_connectivity  # noqa: E402

M2_PER_HA = 10_000.0
W_MAX_M = 500.0

CONFIGS = {
    "code-default": dict(
        extent_m=10_000.0, dispersal_scale_m=1_000.0, alpha=2.0, beta=None,
        topologies=["MST", "Gabriel", "Delaunay", "k-NN (k=3)", "Threshold (30th pct)",
                    "RNG", "MST + 3 loops"],
    ),
    "manuscript-stated": dict(
        extent_m=100_000.0, dispersal_scale_m=10_000.0, alpha=1.0, beta=0.15,
        gamma=0.05, w_min=50.0,
        topologies=["MST", "Gabriel", "Delaunay", "k-NN (k=3)", "RNG", "MST + 3 loops"],
    ),
}


def build_relative_neighborhood_graph(landscape):
    """RNG: keep (i, j) unless some k is closer to both i and j than they are to each other."""
    P = landscape.patches
    D = landscape.distance_matrix()
    edges = []
    n = len(P)
    for i in range(n):
        for j in range(i + 1, n):
            d = D[i, j]
            if not any(max(D[i, k], D[j, k]) < d for k in range(n) if k not in (i, j)):
                edges.append((P[i].id, P[j].id))
    return edges


def manuscript_landscape(n_patches, extent_m, seed):
    rng = np.random.RandomState(seed)
    patches = []
    for i in range(n_patches):
        x, y = rng.uniform(0, extent_m, size=2)
        area = rng.uniform(500.0, 5000.0)          # hectares
        quality = float(np.clip(rng.beta(2.0, 2.0), 0.0, 1.0))
        patches.append(Patch(id=i, x=float(x), y=float(y), area=float(area), quality=quality))
    return Landscape(patches)


def network_edges(name, landscape):
    if name == "MST":
        return build_dendritic_network(landscape).edges
    if name == "Gabriel":
        return build_gabriel_graph(landscape)
    if name == "Delaunay":
        return build_delaunay_triangulation(landscape)
    if name == "k-NN (k=3)":
        return build_knn_network(landscape, k=3)
    if name == "Threshold (30th pct)":
        D = landscape.distance_matrix()
        return build_threshold_network(landscape, np.percentile(D[D > 0], 30))
    if name == "RNG":
        return build_relative_neighborhood_graph(landscape)
    if name == "MST + 3 loops":
        return add_strategic_loops(landscape, build_dendritic_network(landscape).edges, n_loops=3)
    raise ValueError(name)


def metrics_for(landscape, edges, cfg, guild=None):
    s, a = cfg["dispersal_scale_m"], cfg["alpha"]
    kw = dict(dispersal_scale=s, alpha=a)
    H_mov = movement_entropy(landscape, edges, **kw)
    lengths = [landscape.graph[i][j]["distance"] for (i, j) in edges]
    length_term = 0.1 * float(np.mean(lengths)) / s if lengths else 0.0
    H_shannon = H_mov - length_term
    H_rate, _ = calculate_entropy_rate(landscape, edges, **kw)
    H_total, _ = calculate_entropy(landscape, edges, **kw)
    rho2, rc = calculate_two_edge_connectivity(landscape, edges)
    total_len = float(sum(lengths))
    row = dict(H_mov=H_mov, H_shannon=H_shannon, H_sum=H_shannon * landscape.n_patches,
               H_rate=H_rate, H_total=H_total, n_edges=len(edges),
               length_km=total_len / 1000.0, rho2=rho2,
               # Exact P_fail(1): share of corridors whose single loss disconnects the network.
               p_fail1=rc["n_bridges"] / len(edges))
    if cfg.get("beta"):
        budget = cfg["beta"] * sum(p.area for p in landscape.patches) * M2_PER_HA
        row["w_bar_m"] = min(W_MAX_M, budget / total_len)
    if guild is not None:
        from hdfm.optimization import _solve_widths
        x0 = [max(guild.w_crit, row.get("w_bar_m", guild.w_crit))] * len(edges)
        widths, ok, _ = _solve_widths(landscape, edges, guild, cfg["beta"], x0, 100, kw)
        if ok:
            Hw = movement_entropy(landscape, edges, corridor_widths=widths, species_guild=guild, **kw)
            row["H_mov_optw"] = Hw
        else:
            row["H_mov_optw"] = np.nan
    return row


def run(config_name, n_landscapes, n_patches, seed, optimize_widths):
    cfg = CONFIGS[config_name]
    guild = None
    if optimize_widths and cfg.get("beta"):
        from hdfm.species import SpeciesGuild
        w_crit = cfg["w_min"] - np.log(1 - 0.85) / cfg["gamma"]
        guild = SpeciesGuild(name="manuscript", alpha=0.1, gamma=cfg["gamma"],
                             w_min=cfg["w_min"], w_crit=w_crit, Ne_threshold=500)
    rows = {t: [] for t in cfg["topologies"]}
    for i in range(n_landscapes):
        if config_name == "code-default":
            land = SyntheticLandscape(n_patches=n_patches, extent=cfg["extent_m"], random_seed=seed + i)
        else:
            land = manuscript_landscape(n_patches, cfg["extent_m"], seed + i)
        for t in cfg["topologies"]:
            rows[t].append(metrics_for(land, network_edges(t, land), cfg, guild))
    return {t: {k: np.array([r[k] for r in v]) for k in v[0]} for t, v in rows.items()}


def loop_table(config_name, n_landscapes, n_patches, seed, criterion="betweenness",
               loop_counts=(0, 1, 2, 3, 5)):
    """Manuscript Table 3 analogue: MST plus k loops chosen by add_strategic_loops(criterion)."""
    cfg = CONFIGS[config_name]
    acc = {k: [] for k in loop_counts}
    for i in range(n_landscapes):
        if config_name == "code-default":
            land = SyntheticLandscape(n_patches=n_patches, extent=cfg["extent_m"], random_seed=seed + i)
        else:
            land = manuscript_landscape(n_patches, cfg["extent_m"], seed + i)
        mst = build_dendritic_network(land).edges
        base = metrics_for(land, mst, cfg)
        for k in loop_counts:
            m = base if k == 0 else metrics_for(land, add_strategic_loops(land, mst, n_loops=k, criterion=criterion), cfg)
            acc[k].append(dict(
                H_shannon=m["H_shannon"],
                dH=100.0 * (m["H_shannon"] - base["H_shannon"]) / base["H_shannon"],
                rho2=m["rho2"],
                # Extra corridor area at a common width equals extra length.
                d_area=100.0 * (m["length_km"] / base["length_km"] - 1.0),
                p_fail1=m["p_fail1"]))
    lines = [f"### MST plus loops ({config_name}, criterion '{criterion}'; manuscript Table 3 analogue)", "",
             f"Mean over landscapes. Loops added by robustness.add_strategic_loops(criterion='{criterion}'). "
             "Extra area assumes all corridors share one width. P_fail(1) is exact: bridges / corridors.", "",
             "| Loops | H_shannon (nats) | dH vs MST | rho2 | extra area | P_fail(1) |",
             "|---|---|---|---|---|---|"]
    for k, rows in acc.items():
        mean = {key: np.mean([r[key] for r in rows]) for key in rows[0]}
        lines.append(f"| {k} | {mean['H_shannon']:.3f} | {mean['dH']:+.1f}% | {mean['rho2']:.2f} | "
                     f"{mean['d_area']:+.1f}% | {mean['p_fail1']:.2f} |")
    return "\n".join(lines) + "\n"


def tree_checks(config_name, n_landscapes, n_patches, seed, n_random=200):
    """Is the MST the minimum-entropy spanning tree? Compare it with stars and random trees.

    Random trees are minimum spanning trees of independent uniform random weights
    (seeded 1000 + landscape index). Also checks the manuscript's surrogate
    sum_i sum_j p_ij d_ij (km), which its Proposition says the MST minimises.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    cfg = CONFIGS[config_name]
    s, a = cfg["dispersal_scale_m"], cfg["alpha"]

    def h_shannon(land, edges):
        lengths = [land.graph[i][j]["distance"] for (i, j) in edges]
        return movement_entropy(land, edges, dispersal_scale=s, alpha=a) - 0.1 * np.mean(lengths) / s

    def surrogate_km(land, edges):
        nbrs = {i: [] for i in range(land.n_patches)}
        for (i, j) in edges:
            d = land.graph[i][j]["distance"] / 1000.0
            nbrs[i].append((j, d))
            nbrs[j].append((i, d))
        total = 0.0
        for i, nb in nbrs.items():
            w = np.array([np.exp(-a * d * 1000.0 / s) * land.patches[j].quality for j, d in nb])
            total += float(np.dot(w / w.sum(), [d for _, d in nb]))
        return total

    mst_h, star_h, star_wins, rnd_h_wins, rnd_s_wins, total = [], [], 0, 0, 0, 0
    for i in range(n_landscapes):
        if config_name == "code-default":
            land = SyntheticLandscape(n_patches=n_patches, extent=cfg["extent_m"], random_seed=seed + i)
        else:
            land = manuscript_landscape(n_patches, cfg["extent_m"], seed + i)
        mst = build_dendritic_network(land).edges
        h0, s0 = h_shannon(land, mst), surrogate_km(land, mst)
        hs = min(h_shannon(land, [(c, j) for j in range(n_patches) if j != c]) for c in range(n_patches))
        mst_h.append(h0)
        star_h.append(hs)
        star_wins += hs < h0
        rng = np.random.RandomState(1000 + i)
        for _ in range(n_random):
            T = minimum_spanning_tree(np.triu(rng.rand(n_patches, n_patches), 1) + 1e-9).tocoo()
            edges = list(zip(T.row.tolist(), T.col.tolist()))
            rnd_h_wins += h_shannon(land, edges) < h0
            rnd_s_wins += surrogate_km(land, edges) < s0
            total += 1
    return "\n".join([
        f"### Spanning-tree checks ({config_name})", "",
        f"- MST local choice entropy (H_shannon): {np.mean(mst_h):.3f} nats; best star: {np.mean(star_h):.3f} nats; "
        f"star lower than MST in {star_wins}/{n_landscapes} landscapes.",
        f"- Random spanning trees with lower H_shannon than the MST: {rnd_h_wins}/{total} "
        f"({100.0 * rnd_h_wins / total:.1f}%).",
        f"- Random spanning trees with lower surrogate sum_i sum_j p_ij d_ij than the MST: {rnd_s_wins}/{total}.",
        "",
    ])


def summarise(results, metric):
    ref = results["MST"][metric]
    out = []
    for t, m in results.items():
        x = m[metric]
        ok = ~np.isnan(x) & ~np.isnan(ref)
        mean, se = np.nanmean(x), np.nanstd(x, ddof=1) / np.sqrt(ok.sum())
        if t == "MST":
            out.append((t, mean, se, None, None, None, None))
            continue
        pct = 100.0 * (np.mean(x[ok]) - np.mean(ref[ok])) / np.mean(ref[ok])
        p_t = stats.ttest_rel(x[ok], ref[ok]).pvalue
        frac_mst_lower = float(np.mean(ref[ok] < x[ok]))
        out.append((t, mean, se, pct, p_t, frac_mst_lower, int(ok.sum())))
    return out


def fmt_p(p):
    return "<1e-10" if p < 1e-10 else f"{p:.1e}"


def markdown(config_name, results, metrics, args, seconds):
    cfg = CONFIGS[config_name]
    lines = [f"### {config_name}", "",
             f"{args.n_landscapes} landscapes x {args.n_patches} patches, seeds {args.seed}..{args.seed + args.n_landscapes - 1}, "
             f"extent {cfg['extent_m'] / 1000:g} km, alpha = {cfg['alpha'] / cfg['dispersal_scale_m'] * 1000:g} per km.", ""]
    print(f"[{config_name}: topology table computed in {seconds:.1f} s]", file=sys.stderr)
    for metric in metrics:
        if metric not in results["MST"]:
            continue
        lines += [f"**{metric}** (mean +/- SE; % vs MST; paired t-test p; share of landscapes where MST is strictly lower)", "",
                  "| Network | mean | SE | vs MST | p (paired t) | MST lower |",
                  "|---|---|---|---|---|---|"]
        for t, mean, se, pct, p, frac, n in summarise(results, metric):
            if pct is None:
                lines.append(f"| {t} | {mean:.3f} | {se:.3f} | reference | | |")
            else:
                lines.append(f"| {t} | {mean:.3f} | {se:.3f} | {pct:+.1f}% | {fmt_p(p)} | {frac * 100:.0f}% |")
        lines.append("")
    edges = " ; ".join(f"{t}: {results[t]['n_edges'].mean():.1f}" for t in results)
    lines += [f"Mean corridor count: {edges}", ""]
    rho = " ; ".join(f"{t}: {results[t]['rho2'].mean():.2f}" for t in results)
    lines += [f"Mean 2-edge-connectivity rho2: {rho}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", choices=["both"] + list(CONFIGS), default="both")
    ap.add_argument("--n-landscapes", type=int, default=100)
    ap.add_argument("--n-patches", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--optimize-widths", action="store_true",
                    help="also run the SLSQP width solver (manuscript-stated config only; slow)")
    ap.add_argument("--tree-checks", action="store_true",
                    help="also compare the MST with stars and 200 random spanning trees per landscape")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="write topology_table.md and one CSV per configuration here")
    args = ap.parse_args()

    import numpy, scipy, networkx
    header = (f"Generated by examples/reproduce_topology_table.py with numpy {numpy.__version__}, "
              f"scipy {scipy.__version__}, networkx {networkx.__version__}.")
    metrics = ["H_mov", "H_shannon", "H_sum", "H_rate", "H_total", "length_km", "w_bar_m", "H_mov_optw"]
    blocks = [header, ""]
    names = list(CONFIGS) if args.config == "both" else [args.config]
    all_results = {}
    for name in names:
        t0 = time.time()
        res = run(name, args.n_landscapes, args.n_patches, args.seed, args.optimize_widths)
        all_results[name] = res
        blocks.append(markdown(name, res, metrics, args, time.time() - t0))
        for criterion in ("betweenness", "shortest"):
            blocks.append(loop_table(name, args.n_landscapes, args.n_patches, args.seed, criterion))
        if args.tree_checks:
            blocks.append(tree_checks(name, args.n_landscapes, args.n_patches, args.seed))
    text = "\n".join(blocks)
    print(text)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "topology_table.md").write_text(text + "\n")
        for name, res in all_results.items():
            keys = list(next(iter(res.values())).keys())
            with open(out / f"topology_{name}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["landscape", "network"] + keys)
                for t, m in res.items():
                    for i in range(args.n_landscapes):
                        w.writerow([args.seed + i, t] + [f"{m[k][i]:.6g}" for k in keys])


if __name__ == "__main__":
    main()
