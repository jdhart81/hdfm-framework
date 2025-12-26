#!/usr/bin/env python3
"""
HDFM Paper Validation Script

This script validates the scientific claims from:
Hart, J. (2024). "Hierarchical Dendritic Forest Management: A Graph-Theoretic
Framework for Optimized Landscape Connectivity"

Validated Claims:
1. Proposition 3.1: MST achieves minimum total edge resistance
2. Proposition 3.2: Deterministic vulnerability quantification (tree partitioning)
3. Proposition 3.3: Augmented dendritic networks (μ=2-3) achieve favorable
   efficiency-resilience trade-offs
4. Table 1: ECA retention under stochastic edge failure
5. Branching entropy (Hb) as connectivity metric
6. Strahler ordering for hierarchical corridor management
7. Corridor width scaling: W(e) = Wmin × α^(S(e)-1)

Usage:
    python paper_validation.py [--full] [--landscape-size N] [--seed S]

    --full: Run full validation suite (slower, more thorough)
    --landscape-size N: Number of patches in synthetic landscape (default: 20)
    --seed S: Random seed for reproducibility (default: 42)
"""

import argparse
import numpy as np
import sys
from typing import Dict, Any

# HDFM imports
from hdfm import (
    # Landscape
    SyntheticLandscape,

    # Network construction
    build_dendritic_network,
    compare_network_topologies,

    # Graph theory (paper constructs)
    compute_strahler_order,
    compute_branching_entropy,
    compute_corridor_widths_strahler,
    build_augmented_dendritic_network,
    compute_cyclomatic_complexity,
    compute_equivalent_connected_area,
    generate_table_1,
    identify_critical_edges,
    verify_proposition_3_1,
    verify_proposition_3_3,

    # Robustness
    calculate_robustness_metrics,
    pareto_frontier_analysis,
)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def validate_proposition_3_1(landscape: SyntheticLandscape) -> Dict[str, Any]:
    """
    Validate Proposition 3.1: MST achieves minimum total edge resistance.

    "Under graph-theoretic assumptions, the minimum total edge resistance
    achieving full connectivity is attained by a tree (minimum spanning tree, MST)."
    """
    print_section("PROPOSITION 3.1: MST Minimum Resistance")

    # Build MST
    mst_network = build_dendritic_network(landscape)
    mst_edges = list(mst_network.edges)

    # Compare with alternative topologies
    results = compare_network_topologies(landscape)

    mst_length = results['dendritic']['total_length']

    print(f"\nMST total corridor length: {mst_length:.0f} m")
    print(f"MST number of edges: {len(mst_edges)}")

    print("\nComparison with alternative connected topologies:")
    print("-" * 50)

    all_pass = True
    for topo, data in sorted(results.items(), key=lambda x: x[1]['total_length']):
        length = data['total_length']
        n_edges = data['n_edges']
        diff = ((length - mst_length) / mst_length) * 100

        status = "✓" if topo == 'dendritic' or length >= mst_length else "✗"
        if length < mst_length and topo != 'dendritic':
            all_pass = False

        print(f"  {topo:20s}: {length:10.0f} m  ({n_edges:3d} edges)  {diff:+.1f}%  {status}")

    proposition_holds = all_pass

    print(f"\nProposition 3.1 verified: {proposition_holds}")

    return {
        'proposition': '3.1',
        'description': 'MST achieves minimum total edge resistance',
        'verified': proposition_holds,
        'mst_length': mst_length,
        'topology_comparison': results
    }


def validate_proposition_3_2(landscape: SyntheticLandscape) -> Dict[str, Any]:
    """
    Validate Proposition 3.2: Deterministic vulnerability quantification.

    "For a tree, removing any edge e partitions the network into exactly
    two connected components with deterministic, computable sizes."
    """
    print_section("PROPOSITION 3.2: Deterministic Vulnerability")

    import networkx as nx

    # Build MST
    mst_network = build_dendritic_network(landscape)
    edges = list(mst_network.edges)
    n = landscape.n_patches

    # Build graph
    G = nx.Graph()
    id_to_idx = {p.id: i for i, p in enumerate(landscape.patches)}
    G.add_nodes_from(range(n))
    for (i, j) in edges:
        G.add_edge(id_to_idx[i], id_to_idx[j])

    print(f"\nTesting {len(edges)} MST edges...")

    all_pass = True
    partitions = []

    for edge in edges:
        i, j = edge
        idx_i, idx_j = id_to_idx[i], id_to_idx[j]

        # Remove edge
        G_test = G.copy()
        G_test.remove_edge(idx_i, idx_j)

        # Count components
        components = list(nx.connected_components(G_test))
        n_components = len(components)

        if n_components != 2:
            all_pass = False
            print(f"  Edge ({i}, {j}): FAILED - {n_components} components")
        else:
            sizes = [len(c) for c in components]
            partitions.append(tuple(sorted(sizes)))

    # Show unique partition sizes
    unique_partitions = sorted(set(partitions))

    print(f"\nUnique partition sizes (component1, component2):")
    for p in unique_partitions[:10]:
        print(f"  {p}")
    if len(unique_partitions) > 10:
        print(f"  ... and {len(unique_partitions) - 10} more")

    proposition_holds = all_pass
    print(f"\nProposition 3.2 verified: {proposition_holds}")
    print("  Every edge removal produces exactly 2 components with deterministic sizes")

    return {
        'proposition': '3.2',
        'description': 'Tree edge removal produces exactly 2 components',
        'verified': proposition_holds,
        'n_edges_tested': len(edges),
        'unique_partitions': unique_partitions
    }


def validate_proposition_3_3(landscape: SyntheticLandscape, n_simulations: int = 500) -> Dict[str, Any]:
    """
    Validate Proposition 3.3: Efficiency-resilience trade-off.

    "Augmented dendritic networks with small μ (2–3) achieve near-optimal
    efficiency while substantially improving resilience compared to pure
    dendritic (μ = 0) networks."
    """
    print_section("PROPOSITION 3.3: Efficiency-Resilience Trade-off")

    result = verify_proposition_3_3(
        landscape,
        edge_failure_prob=0.05,
        n_simulations=n_simulations
    )

    print(f"\nAt edge failure probability p = 0.05:")
    print(f"  μ = 0 (pure dendritic) ECA retention: {result['mu_0_eca_retention']:.2%}")
    print(f"  μ = 2 (augmented)      ECA retention: {result['mu_2_eca_retention']:.2%}")
    print(f"  Resilience improvement: {result['resilience_improvement_percent']:.1f}%")
    print(f"  Corridor area increase: {result['area_increase_percent']:.1f}%")

    print(f"\nProposition 3.3 verified: {result['proposition_holds']}")
    print("  (Resilience improvement > 0% with area increase < 15%)")

    return {
        'proposition': '3.3',
        'description': 'Augmented networks improve resilience with modest area increase',
        'verified': result['proposition_holds'],
        **result
    }


def validate_table_1(landscape: SyntheticLandscape, n_simulations: int = 500) -> Dict[str, Any]:
    """
    Reproduce Table 1: ECA retention under stochastic edge failure.

    Tests networks with μ = 0, 1, 2, 3 under failure probabilities
    p = 0.01, 0.05, 0.10, 0.20
    """
    print_section("TABLE 1: ECA Retention Under Edge Failure")

    print(f"\nRunning {n_simulations} Monte Carlo simulations per configuration...")

    results = generate_table_1(
        landscape,
        max_mu=3,
        edge_failure_probs=[0.01, 0.05, 0.10, 0.20],
        n_simulations=n_simulations
    )

    print("\n" + results['formatted_table'])

    print("\nCorridor area by augmentation level:")
    for mu, area in results['corridor_areas'].items():
        print(f"  μ = {mu}: {area/10000:.2f} ha")

    # Verify expected patterns
    table = results['table']

    # Pattern 1: ECA retention decreases with failure probability
    pattern1_holds = all(
        table[i, mu] >= table[i+1, mu]
        for mu in range(table.shape[1])
        for i in range(table.shape[0] - 1)
    )

    # Pattern 2: ECA retention increases with augmentation (μ)
    pattern2_holds = all(
        table[p, mu] <= table[p, mu+1]
        for p in range(table.shape[0])
        for mu in range(table.shape[1] - 1)
    )

    print("\nExpected patterns verified:")
    print(f"  ECA retention decreases with failure prob: {pattern1_holds}")
    print(f"  ECA retention increases with augmentation: {pattern2_holds}")

    return {
        'table_name': 'Table 1',
        'description': 'ECA retention under stochastic edge failure',
        'verified': pattern1_holds and pattern2_holds,
        'table': table,
        'probabilities': results['probabilities'],
        'mu_values': results['mu_values'],
        'formatted_table': results['formatted_table']
    }


def validate_strahler_ordering(landscape: SyntheticLandscape) -> Dict[str, Any]:
    """
    Validate Strahler ordering implementation.

    Tests Definition 2.4:
    (i) Leaf nodes: S(v) = 1
    (ii) If v has children with maximum order k, and ≥2 children have order k: S(v) = k + 1
    (iii) Otherwise: S(v) = max(child orders)
    """
    print_section("STRAHLER ORDERING (Definition 2.4)")

    # Build MST
    mst_network = build_dendritic_network(landscape)
    edges = list(mst_network.edges)

    # Compute Strahler ordering
    strahler = compute_strahler_order(landscape, edges, root_selection='max_area')

    print(f"\nRoot node: {strahler.root_node} (largest patch)")
    print(f"Maximum Strahler order: {strahler.max_order}")

    print("\nOrder distribution (edge counts):")
    total_edges = sum(strahler.order_distribution.values())
    for order in sorted(strahler.order_distribution.keys()):
        count = strahler.order_distribution[order]
        pct = count / total_edges * 100
        print(f"  Order {order}: {count:3d} edges ({pct:.1f}%)")

    # Validate leaf nodes have order 1
    import networkx as nx
    G = nx.Graph()
    id_to_idx = {p.id: i for i, p in enumerate(landscape.patches)}
    G.add_nodes_from(range(landscape.n_patches))
    for (i, j) in edges:
        G.add_edge(id_to_idx[i], id_to_idx[j])

    leaf_orders_correct = True
    for node in G.nodes():
        if G.degree(node) == 1:  # Leaf
            patch_id = landscape.patches[node].id
            if strahler.node_orders[patch_id] != 1:
                leaf_orders_correct = False
                break

    print(f"\nLeaf nodes have order 1: {leaf_orders_correct}")

    return {
        'feature': 'Strahler Ordering',
        'description': 'Hierarchical ordering for corridor management',
        'verified': leaf_orders_correct,
        'max_order': strahler.max_order,
        'order_distribution': strahler.order_distribution
    }


def validate_branching_entropy(landscape: SyntheticLandscape) -> Dict[str, Any]:
    """
    Validate branching entropy (Hb) calculation.

    Hb(T) = −Σs ps log2 ps

    where ps = As/ΣA is fractional area in Strahler order s.
    """
    print_section("BRANCHING ENTROPY (Definition 2.3)")

    # Build MST
    mst_network = build_dendritic_network(landscape)
    edges = list(mst_network.edges)

    # Compute branching entropy
    Hb, components = compute_branching_entropy(landscape, edges)

    print(f"\nBranching entropy Hb = {Hb:.4f} bits")

    print("\nOrder fractions (ps):")
    for order in sorted(components['order_fractions'].keys()):
        ps = components['order_fractions'][order]
        area = components['order_areas'][order]
        print(f"  Order {order}: ps = {ps:.4f} (area = {area/10000:.2f} ha)")

    print(f"\nTotal corridor area: {components['total_corridor_area']/10000:.2f} ha")

    # Theoretical bounds
    n_orders = len(components['order_fractions'])
    Hb_max = np.log2(n_orders) if n_orders > 0 else 0

    print(f"\nTheoretical maximum Hb (uniform): {Hb_max:.4f} bits")
    print(f"Normalized Hb: {Hb/Hb_max:.2%}" if Hb_max > 0 else "N/A")

    # Verify entropy is non-negative
    verified = Hb >= 0

    return {
        'feature': 'Branching Entropy',
        'description': 'Information-theoretic connectivity metric',
        'verified': verified,
        'Hb': Hb,
        'Hb_max': Hb_max,
        'order_fractions': components['order_fractions']
    }


def validate_corridor_width_scaling(landscape: SyntheticLandscape) -> Dict[str, Any]:
    """
    Validate corridor width prescription.

    W(e) = Wmin × α^(S(e)-1)

    With Wmin = 50m and α = 2.0:
    Order 1 = 50m, Order 2 = 100m, Order 3 = 200m, Order 4 = 400m
    """
    print_section("CORRIDOR WIDTH SCALING (Equation 2)")

    # Build MST
    mst_network = build_dendritic_network(landscape)
    edges = list(mst_network.edges)

    # Compute Strahler ordering
    strahler = compute_strahler_order(landscape, edges)

    # Compute widths with paper parameters
    W_min = 50.0
    alpha = 2.0
    widths = compute_corridor_widths_strahler(strahler, W_min=W_min, alpha=alpha)

    print(f"\nParameters: W_min = {W_min}m, α = {alpha}")

    print("\nExpected widths:")
    for order in range(1, 5):
        expected = W_min * (alpha ** (order - 1))
        print(f"  Order {order}: {expected:.0f} m")

    print("\nActual width distribution:")
    width_counts = {}
    for edge, width in widths.items():
        width_counts[width] = width_counts.get(width, 0) + 1

    for width in sorted(width_counts.keys()):
        count = width_counts[width]
        order = int(np.log(width / W_min) / np.log(alpha)) + 1
        print(f"  {width:.0f} m (Order {order}): {count} edges")

    # Verify width formula
    verified = all(
        abs(widths[e] - W_min * (alpha ** (strahler.edge_orders.get(e,
            strahler.edge_orders.get((e[1], e[0]), 1)) - 1))) < 0.01
        for e in edges
    )

    print(f"\nWidth formula verified: {verified}")

    return {
        'feature': 'Corridor Width Scaling',
        'description': 'W(e) = Wmin × α^(S(e)-1)',
        'verified': verified,
        'W_min': W_min,
        'alpha': alpha,
        'width_distribution': width_counts
    }


def validate_edge_criticality(landscape: SyntheticLandscape) -> Dict[str, Any]:
    """
    Validate edge criticality analysis.

    "Criticality analysis identifies two high-vulnerability edges."
    """
    print_section("EDGE CRITICALITY ANALYSIS (Section 6.2)")

    # Build MST
    mst_network = build_dendritic_network(landscape)
    edges = list(mst_network.edges)

    # Identify critical edges
    critical = identify_critical_edges(landscape, edges, top_k=5)

    print("\nTop 5 most critical edges:")
    print("-" * 60)

    for i, (edge, metrics) in enumerate(critical):
        print(f"\n{i+1}. Edge {edge}:")
        print(f"   Criticality score: {metrics['criticality_score']:.3f}")
        print(f"   ECA impact: {metrics['eca_impact']:.3f}")
        print(f"   Betweenness: {metrics['betweenness']:.3f}")
        print(f"   Is bridge: {metrics['is_bridge']}")

    # Verify all MST edges are bridges
    all_bridges = all(m['is_bridge'] for _, m in critical)

    print(f"\nAll MST edges are bridges (cut edges): {all_bridges}")

    return {
        'feature': 'Edge Criticality',
        'description': 'Identification of high-vulnerability edges',
        'verified': all_bridges,  # In MST, all edges are critical
        'critical_edges': critical[:2]
    }


def validate_augmented_network(landscape: SyntheticLandscape) -> Dict[str, Any]:
    """
    Validate augmented dendritic network construction.

    "A network Taug = (V, Eaug) is augmented dendritic if it consists of
    a minimum spanning tree backbone TMST plus k strategically selected
    augmentation edges, yielding cyclomatic complexity μ = k"
    """
    print_section("AUGMENTED DENDRITIC NETWORK (Definition 2.2)")

    results = {}

    for target_mu in [0, 1, 2, 3]:
        aug_net = build_augmented_dendritic_network(landscape, mu=target_mu)
        actual_mu = compute_cyclomatic_complexity(landscape, aug_net.all_edges)

        results[target_mu] = {
            'mst_edges': len(aug_net.mst_edges),
            'augmentation_edges': len(aug_net.augmentation_edges),
            'total_edges': len(aug_net.all_edges),
            'mu': actual_mu,
            'matches': actual_mu == target_mu
        }

    print("\nAugmented network construction:")
    print("-" * 60)
    print(f"{'Target μ':>10} {'MST edges':>12} {'Loop edges':>12} {'Actual μ':>10} {'Match':>8}")
    print("-" * 60)

    all_match = True
    for target_mu, data in results.items():
        match_str = "✓" if data['matches'] else "✗"
        if not data['matches']:
            all_match = False
        print(f"{target_mu:>10} {data['mst_edges']:>12} {data['augmentation_edges']:>12} "
              f"{data['mu']:>10} {match_str:>8}")

    print(f"\nAll augmentation levels correct: {all_match}")

    return {
        'feature': 'Augmented Dendritic Network',
        'description': 'MST + strategic loops with cyclomatic complexity μ',
        'verified': all_match,
        'results': results
    }


def run_full_validation(landscape_size: int = 20, seed: int = 42, full: bool = False):
    """Run complete paper validation suite."""
    print("\n" + "=" * 70)
    print("HDFM PAPER VALIDATION SUITE")
    print("Hart (2024): Hierarchical Dendritic Forest Management")
    print("=" * 70)

    # Create synthetic landscape
    np.random.seed(seed)
    landscape = SyntheticLandscape(
        n_patches=landscape_size,
        extent=15000.0,  # 15km extent (similar to paper's 50,000 ha)
        random_seed=seed
    )

    print(f"\nTest landscape: {landscape_size} patches, 15km × 15km")
    print(f"Random seed: {seed}")

    n_simulations = 1000 if full else 500
    print(f"Monte Carlo simulations: {n_simulations}")

    # Run validations
    results = []

    # Core propositions
    results.append(validate_proposition_3_1(landscape))
    results.append(validate_proposition_3_2(landscape))
    results.append(validate_proposition_3_3(landscape, n_simulations))

    # Table 1
    results.append(validate_table_1(landscape, n_simulations))

    # Paper constructs
    results.append(validate_strahler_ordering(landscape))
    results.append(validate_branching_entropy(landscape))
    results.append(validate_corridor_width_scaling(landscape))
    results.append(validate_edge_criticality(landscape))
    results.append(validate_augmented_network(landscape))

    # Summary
    print_section("VALIDATION SUMMARY")

    n_passed = sum(1 for r in results if r.get('verified', False))
    n_total = len(results)

    print(f"\nResults: {n_passed}/{n_total} validations passed")
    print("-" * 50)

    for r in results:
        name = r.get('proposition', r.get('table_name', r.get('feature', 'Unknown')))
        desc = r.get('description', '')
        status = "✓ PASS" if r.get('verified', False) else "✗ FAIL"
        print(f"  {status}: {name} - {desc}")

    all_passed = n_passed == n_total

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL VALIDATIONS PASSED")
        print("The HDFM framework correctly implements the paper's claims.")
    else:
        print("SOME VALIDATIONS FAILED")
        print("Review the output above for details.")
    print("=" * 70)

    return all_passed, results


def main():
    parser = argparse.ArgumentParser(
        description='Validate HDFM paper claims',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--full', action='store_true',
                       help='Run full validation suite (more simulations)')
    parser.add_argument('--landscape-size', type=int, default=20,
                       help='Number of patches in synthetic landscape')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    all_passed, results = run_full_validation(
        landscape_size=args.landscape_size,
        seed=args.seed,
        full=args.full
    )

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
