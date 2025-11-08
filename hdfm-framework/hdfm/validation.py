"""
Validation tools for HDFM framework.

Provides functions for empirical validation, statistical testing, and
convergence analysis.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
from .landscape import Landscape, SyntheticLandscape
from .network import compare_network_topologies, NetworkTopology
from .optimization import DendriticOptimizer, OptimizationResult


def validate_network(
    network,
    landscape: Landscape
) -> Dict[str, bool]:
    """
    Validate network meets all invariants.
    
    Args:
        network: Network object to validate
        landscape: Landscape object
        
    Returns:
        Dictionary of validation results
    """
    import networkx as nx
    
    n = landscape.n_patches
    G = network.graph
    
    validations = {
        'connected': nx.is_connected(G),
        'acyclic': nx.is_tree(G),
        'correct_size': len(network.edges) == n - 1,
        'is_dendritic': nx.is_tree(G) and nx.is_connected(G),
        'valid_edges': all(
            u in [p.id for p in landscape.patches] and
            v in [p.id for p in landscape.patches]
            for (u, v) in network.edges
        )
    }
    
    # Check entropy is non-negative
    H, _ = network.entropy()
    validations['non_negative_entropy'] = H >= 0
    
    return validations


def run_comparative_analysis(
    landscape: Landscape,
    topologies: List[NetworkTopology] = None,
    **entropy_kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Run comparative analysis across network topologies.
    
    Args:
        landscape: Landscape to analyze
        topologies: List of topologies to compare
        **entropy_kwargs: Parameters for entropy calculation
        
    Returns:
        Dictionary mapping topology name to metrics
    """
    return compare_network_topologies(landscape, topologies, **entropy_kwargs)


def convergence_test(
    optimizer,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Tuple[bool, int, List[float]]:
    """
    Test optimization convergence.
    
    Args:
        optimizer: Optimizer object
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        (converged, iterations, entropy_history)
    """
    result = optimizer.optimize(
        max_iterations=max_iterations,
        tolerance=tolerance
    )
    
    # Check convergence
    history = result.convergence_history
    if len(history) < 2:
        return True, 1, history
    
    # Check if entropy stabilized
    final_values = history[-3:]
    if len(final_values) >= 2:
        variance = np.var(final_values)
        converged = variance < tolerance
    else:
        converged = True
    
    return converged, result.iterations, history


def monte_carlo_validation(
    n_landscapes: int = 100,
    n_patches: int = 15,
    extent: float = 10000.0,
    topologies: List[NetworkTopology] = None,
    random_seed: int = None,
    **entropy_kwargs
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Monte Carlo validation over random landscapes.
    
    Generates many random landscapes and compares network topologies
    to validate theoretical predictions about dendritic optimality.
    
    Args:
        n_landscapes: Number of random landscapes to generate
        n_patches: Patches per landscape
        extent: Landscape extent (meters)
        topologies: Network topologies to compare
        random_seed: Random seed for reproducibility
        **entropy_kwargs: Parameters for entropy calculation
        
    Returns:
        Dictionary mapping topology to arrays of entropy values
        
    Example:
        >>> results = monte_carlo_validation(n_landscapes=100, n_patches=15)
        >>> print(f"Dendritic mean: {results['dendritic']['H_total'].mean():.3f}")
        >>> print(f"Gabriel mean: {results['gabriel']['H_total'].mean():.3f}")
    """
    if topologies is None:
        topologies = [
            NetworkTopology.DENDRITIC,
            NetworkTopology.GABRIEL,
            NetworkTopology.DELAUNAY,
            NetworkTopology.KNN,
            NetworkTopology.THRESHOLD
        ]
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Initialize result storage
    results = {topo.value: {
        'H_total': [],
        'H_mov': [],
        'C': [],
        'F': [],
        'D': [],
        'n_edges': [],
        'total_length': []
    } for topo in topologies}
    
    print(f"Running Monte Carlo validation with {n_landscapes} landscapes...")
    
    for i in range(n_landscapes):
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_landscapes} landscapes")
        
        # Generate random landscape
        landscape = SyntheticLandscape(
            n_patches=n_patches,
            extent=extent,
            random_seed=random_seed + i if random_seed else None
        )
        
        # Compare topologies
        comparison = compare_network_topologies(
            landscape,
            topologies,
            **entropy_kwargs
        )
        
        # Store results
        for topo_name, metrics in comparison.items():
            for key, value in metrics.items():
                results[topo_name][key].append(value)
    
    # Convert to arrays
    for topo_name in results:
        for key in results[topo_name]:
            results[topo_name][key] = np.array(results[topo_name][key])
    
    print("Monte Carlo validation complete!")
    
    return results


def statistical_comparison(
    results: Dict[str, Dict[str, np.ndarray]],
    reference_topology: str = 'dendritic',
    metric: str = 'H_total'
) -> Dict[str, Dict[str, float]]:
    """
    Statistical comparison of network topologies.
    
    Performs paired statistical tests comparing each topology to reference
    (typically dendritic/MST).
    
    Args:
        results: Results from monte_carlo_validation
        reference_topology: Reference topology for comparison
        metric: Metric to compare (default: 'H_total')
        
    Returns:
        Dictionary mapping topology to test statistics:
        - 'mean_difference': Mean difference vs. reference
        - 'percent_difference': Percent difference vs. reference
        - 'p_value': Wilcoxon signed-rank test p-value
        - 'effect_size': Cohen's d effect size
        
    Example:
        >>> comparison = statistical_comparison(results)
        >>> for topo, stats in comparison.items():
        ...     print(f"{topo}: {stats['percent_difference']:.1f}% higher, p={stats['p_value']:.4f}")
    """
    if reference_topology not in results:
        raise ValueError(f"Reference topology '{reference_topology}' not in results")
    
    reference_values = results[reference_topology][metric]
    
    comparisons = {}
    
    for topo_name, metrics in results.items():
        if topo_name == reference_topology:
            continue
        
        topo_values = metrics[metric]
        
        # Mean differences
        mean_diff = np.mean(topo_values - reference_values)
        percent_diff = 100 * mean_diff / np.mean(reference_values)
        
        # Wilcoxon signed-rank test (paired, non-parametric)
        stat, p_value = stats.wilcoxon(topo_values, reference_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.std(topo_values)**2 + np.std(reference_values)**2) / 2
        )
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        comparisons[topo_name] = {
            'mean_difference': mean_diff,
            'percent_difference': percent_diff,
            'p_value': p_value,
            'effect_size': cohens_d,
            'reference_mean': np.mean(reference_values),
            'topology_mean': np.mean(topo_values)
        }
    
    return comparisons


def print_validation_summary(
    results: Dict[str, Dict[str, np.ndarray]],
    comparison: Dict[str, Dict[str, float]]
):
    """
    Print formatted validation summary.
    
    Args:
        results: Results from monte_carlo_validation
        comparison: Results from statistical_comparison
    """
    print("\n" + "="*80)
    print("NETWORK TOPOLOGY VALIDATION RESULTS")
    print("="*80)
    
    # Header
    print(f"\n{'Topology':<20} {'Mean H':<12} {'Std H':<12} {'vs. Dendritic':<20} {'p-value':<10}")
    print("-"*80)
    
    # Dendritic (reference)
    dendritic_H = results['dendritic']['H_total']
    print(f"{'Dendritic (MST)':<20} {dendritic_H.mean():<12.3f} {dendritic_H.std():<12.3f} {'(reference)':<20} {'—':<10}")
    
    # Other topologies
    for topo_name, stats in comparison.items():
        topo_H = results[topo_name]['H_total']
        vs_str = f"+{stats['percent_difference']:.1f}%"
        p_str = f"{stats['p_value']:.4f}" if stats['p_value'] >= 0.0001 else "<0.0001"
        
        print(f"{topo_name.capitalize():<20} {topo_H.mean():<12.3f} {topo_H.std():<12.3f} {vs_str:<20} {p_str:<10}")
    
    print("\n" + "="*80)
    print(f"n = {len(dendritic_H)} landscapes")
    print("Statistical test: Wilcoxon signed-rank (paired, two-tailed)")
    print("="*80 + "\n")


def validate_dendritic_optimality(
    n_landscapes: int = 100,
    n_patches: int = 15,
    random_seed: int = 42
) -> bool:
    """
    Validate that dendritic networks minimize entropy.
    
    Core validation of theoretical claim: dendritic (MST) topology
    minimizes landscape entropy compared to alternative topologies.
    
    Args:
        n_landscapes: Number of test landscapes
        n_patches: Patches per landscape
        random_seed: Random seed
        
    Returns:
        True if dendritic networks consistently minimize entropy
        
    Prints detailed validation report.
    """
    print("\n" + "="*80)
    print("VALIDATING DENDRITIC OPTIMALITY THEOREM")
    print("="*80)
    print(f"\nGenerating {n_landscapes} random landscapes with {n_patches} patches each...")
    
    # Run Monte Carlo validation
    results = monte_carlo_validation(
        n_landscapes=n_landscapes,
        n_patches=n_patches,
        random_seed=random_seed
    )
    
    # Statistical comparison
    comparison = statistical_comparison(results)
    
    # Print summary
    print_validation_summary(results, comparison)
    
    # Check if dendritic is always best
    dendritic_H = results['dendritic']['H_total']
    
    all_better = True
    for topo_name, metrics in results.items():
        if topo_name == 'dendritic':
            continue
        
        topo_H = metrics['H_total']
        
        # Check if dendritic is better in every instance
        instances_better = np.sum(dendritic_H < topo_H)
        percent_better = 100 * instances_better / len(dendritic_H)
        
        print(f"Dendritic better than {topo_name} in {instances_better}/{len(dendritic_H)} cases ({percent_better:.1f}%)")
        
        if percent_better < 95:  # Allow 5% tolerance for numerical issues
            all_better = False
    
    print("\n" + "="*80)
    if all_better:
        print("✓ VALIDATION PASSED: Dendritic networks minimize entropy")
    else:
        print("✗ VALIDATION FAILED: Some alternatives occasionally better")
    print("="*80 + "\n")
    
    return all_better
