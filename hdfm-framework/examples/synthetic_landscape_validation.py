#!/usr/bin/env python3
"""
Synthetic Landscape Validation for HDFM Framework

Reproduces the validation experiments from:
Hart, J. (2024). "Hierarchical Dendritic Forest Management: A Vision for 
Technology-Enabled Landscape Conservation." EcoEvoRxiv.

This script:
1. Generates 100 random 15-patch landscapes
2. Compares 5 network topologies (Dendritic, Gabriel, Delaunay, k-NN, Threshold)
3. Validates that dendritic networks minimize entropy
4. Produces statistical analysis and visualizations

Expected results:
- Dendritic (MST): H = 2.28 ± 0.04
- Gabriel: H = 2.87 ± 0.06 (+25.9%)
- Delaunay: H = 3.12 ± 0.08 (+36.8%)
- k-NN: H = 2.94 ± 0.07 (+28.9%)
- Threshold: H = 3.05 ± 0.09 (+33.8%)

Runtime: ~2-5 minutes depending on hardware
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from hdfm import (
    SyntheticLandscape,
    NetworkTopology,
    monte_carlo_validation,
    statistical_comparison,
    print_validation_summary,
    validate_dendritic_optimality,
    plot_topology_comparison,
    create_validation_figure,
    compare_network_topologies
)


def run_full_validation(
    n_landscapes: int = 100,
    n_patches: int = 15,
    extent: float = 10000.0,
    random_seed: int = 42,
    save_figures: bool = True,
    output_dir: str = "validation_results"
):
    """
    Run complete validation suite reproducing paper results.
    
    Args:
        n_landscapes: Number of random landscapes (100 in paper)
        n_patches: Patches per landscape (15 in paper)
        extent: Landscape extent in meters (10km in paper)
        random_seed: Random seed for reproducibility
        save_figures: Whether to save figures to disk
        output_dir: Directory for output files
        
    Returns:
        Dictionary with validation results
    """
    print("\n" + "="*80)
    print("HDFM SYNTHETIC LANDSCAPE VALIDATION")
    print("Reproducing results from Hart (2024)")
    print("="*80)
    
    # Create output directory
    if save_figures:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        print(f"\nOutput directory: {output_path.absolute()}")
    
    # Define topologies to compare
    topologies = [
        NetworkTopology.DENDRITIC,
        NetworkTopology.GABRIEL,
        NetworkTopology.DELAUNAY,
        NetworkTopology.KNN,
        NetworkTopology.THRESHOLD
    ]
    
    print(f"\nValidation parameters:")
    print(f"  - Landscapes: {n_landscapes}")
    print(f"  - Patches per landscape: {n_patches}")
    print(f"  - Extent: {extent/1000:.1f} km")
    print(f"  - Topologies: {[t.value for t in topologies]}")
    print(f"  - Random seed: {random_seed}")
    
    # Run Monte Carlo validation
    print("\n" + "-"*80)
    print("STEP 1: Monte Carlo Validation")
    print("-"*80)
    
    results = monte_carlo_validation(
        n_landscapes=n_landscapes,
        n_patches=n_patches,
        extent=extent,
        topologies=topologies,
        random_seed=random_seed
    )
    
    # Statistical comparison
    print("\n" + "-"*80)
    print("STEP 2: Statistical Analysis")
    print("-"*80)
    
    comparison = statistical_comparison(results, reference_topology='dendritic')
    print_validation_summary(results, comparison)
    
    # Validate dendritic optimality
    print("\n" + "-"*80)
    print("STEP 3: Dendritic Optimality Test")
    print("-"*80)
    
    dendritic_is_optimal = validate_dendritic_optimality(
        n_landscapes=n_landscapes,
        n_patches=n_patches,
        random_seed=random_seed
    )
    
    # Create visualizations
    print("\n" + "-"*80)
    print("STEP 4: Generating Visualizations")
    print("-"*80)
    
    # Figure 1: Topology comparison
    print("  Creating topology comparison plot...")
    fig1 = plot_topology_comparison(results, metric='H_total')
    if save_figures:
        fig1.savefig(output_path / "topology_comparison.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path / 'topology_comparison.png'}")
    
    # Figure 2: Example landscape validation
    print("  Creating example landscape visualization...")
    example_landscape = SyntheticLandscape(
        n_patches=n_patches,
        extent=extent,
        random_seed=random_seed
    )
    example_results = compare_network_topologies(example_landscape, topologies)
    
    fig2 = create_validation_figure(example_landscape, example_results)
    if save_figures:
        fig2.savefig(output_path / "example_landscape.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path / 'example_landscape.png'}")
    
    # Figure 3: Entropy distribution histograms
    print("  Creating entropy distribution plots...")
    fig3, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (topo_name, metrics) in enumerate(results.items()):
        if idx >= 6:
            break
        
        ax = axes[idx]
        H_values = metrics['H_total']
        
        ax.hist(H_values, bins=20, alpha=0.7, edgecolor='black',
                color='green' if topo_name == 'dendritic' else 'steelblue')
        ax.axvline(H_values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {H_values.mean():.3f}')
        ax.set_xlabel('Entropy H(L)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{topo_name.capitalize()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove unused subplots
    for idx in range(len(results), 6):
        fig3.delaxes(axes[idx])
    
    fig3.suptitle('Distribution of Entropy Across Network Topologies', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_figures:
        fig3.savefig(output_path / "entropy_distributions.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path / 'entropy_distributions.png'}")
    
    # Save numerical results
    if save_figures:
        print("\n  Creating results summary file...")
        with open(output_path / "validation_results.txt", 'w') as f:
            f.write("="*80 + "\n")
            f.write("HDFM SYNTHETIC LANDSCAPE VALIDATION RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Parameters:\n")
            f.write(f"  - Landscapes: {n_landscapes}\n")
            f.write(f"  - Patches per landscape: {n_patches}\n")
            f.write(f"  - Extent: {extent/1000:.1f} km\n")
            f.write(f"  - Random seed: {random_seed}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-"*80 + "\n\n")
            
            f.write(f"{'Topology':<20} {'Mean H':<12} {'Std H':<12} {'Min H':<12} {'Max H':<12}\n")
            f.write("-"*80 + "\n")
            
            for topo_name, metrics in results.items():
                H = metrics['H_total']
                f.write(f"{topo_name.capitalize():<20} {H.mean():<12.4f} {H.std():<12.4f} {H.min():<12.4f} {H.max():<12.4f}\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("STATISTICAL COMPARISONS (vs. Dendritic)\n")
            f.write("-"*80 + "\n\n")
            
            f.write(f"{'Topology':<20} {'Mean Diff':<15} {'% Diff':<15} {'p-value':<15} {'Cohen\'s d':<15}\n")
            f.write("-"*80 + "\n")
            
            for topo_name, stats in comparison.items():
                f.write(f"{topo_name.capitalize():<20} {stats['mean_difference']:<15.4f} "
                       f"{stats['percent_difference']:<15.2f} {stats['p_value']:<15.6f} "
                       f"{stats['effect_size']:<15.3f}\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("VALIDATION OUTCOME\n")
            f.write("-"*80 + "\n\n")
            
            if dendritic_is_optimal:
                f.write("✓ VALIDATION PASSED\n")
                f.write("Dendritic networks consistently minimize landscape entropy.\n")
                f.write("Theoretical predictions confirmed across 100 random landscapes.\n")
            else:
                f.write("✗ VALIDATION FAILED\n")
                f.write("Dendritic networks did not consistently minimize entropy.\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"  ✓ Saved: {output_path / 'validation_results.txt'}")
    
    # Final summary
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    
    if dendritic_is_optimal:
        print("\n✓ Result: Dendritic networks minimize entropy")
        print("  Theoretical predictions CONFIRMED")
    else:
        print("\n✗ Result: Validation inconclusive")
        print("  Further investigation recommended")
    
    if save_figures:
        print(f"\nResults saved to: {output_path.absolute()}")
        print("  - topology_comparison.png")
        print("  - example_landscape.png")
        print("  - entropy_distributions.png")
        print("  - validation_results.txt")
    
    print("\n" + "="*80 + "\n")
    
    return {
        'results': results,
        'comparison': comparison,
        'dendritic_optimal': dendritic_is_optimal,
        'figures': [fig1, fig2, fig3] if not save_figures else None
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run HDFM synthetic landscape validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--n-landscapes', type=int, default=100,
                       help='Number of random landscapes')
    parser.add_argument('--n-patches', type=int, default=15,
                       help='Patches per landscape')
    parser.add_argument('--extent', type=float, default=10000.0,
                       help='Landscape extent (meters)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save figures (display only)')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Run validation
    validation_output = run_full_validation(
        n_landscapes=args.n_landscapes,
        n_patches=args.n_patches,
        extent=args.extent,
        random_seed=args.seed,
        save_figures=not args.no_save,
        output_dir=args.output_dir
    )
    
    # Show plots if not saving
    if args.no_save:
        plt.show()
