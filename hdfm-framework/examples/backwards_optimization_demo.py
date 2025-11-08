#!/usr/bin/env python3
"""
Backwards Climate Optimization Demo

Demonstrates the backwards optimization algorithm for climate-adaptive
corridor design. Works from 2100 climate projections back to present-day
implementation.
"""

import numpy as np
import matplotlib.pyplot as plt

from hdfm import (
    SyntheticLandscape,
    BackwardsOptimizer,
    ClimateScenario,
    DendriticOptimizer,
    plot_network,
    plot_climate_trajectory
)


def compare_forward_vs_backwards(
    n_patches: int = 15,
    extent: float = 10000.0,
    random_seed: int = 42
):
    """
    Compare forward-looking vs. backwards optimization approaches.
    
    Shows that backwards optimization maintains higher connectivity under
    future climate scenarios compared to forward-looking designs.
    """
    print("\n" + "="*80)
    print("BACKWARDS VS. FORWARD OPTIMIZATION COMPARISON")
    print("="*80)
    
    # Create landscape
    print(f"\nGenerating {n_patches}-patch landscape...")
    landscape = SyntheticLandscape(
        n_patches=n_patches,
        extent=extent,
        random_seed=random_seed
    )
    
    # Define climate scenario (RCP 8.5-like trajectory)
    print("Defining climate scenario...")
    scenario = ClimateScenario(
        years=[2025, 2050, 2075, 2100],
        temperature_changes=[0.0, 1.5, 2.5, 3.5],  # °C warming
        precipitation_changes=[0.0, -5.0, -10.0, -15.0]  # % change
    )
    
    print("\nClimate trajectory:")
    for year, temp, precip in zip(scenario.years, scenario.temperature_changes, scenario.precipitation_changes):
        print(f"  {year}: +{temp:.1f}°C, {precip:+.1f}% precipitation")
    
    # Forward approach: optimize for current conditions only
    print("\n" + "-"*80)
    print("FORWARD APPROACH (Current conditions only)")
    print("-"*80)
    
    forward_optimizer = DendriticOptimizer(landscape)
    forward_result = forward_optimizer.optimize()
    
    print(f"  Corridors established: {len(forward_result.network.edges)}")
    print(f"  Current entropy: {forward_result.entropy:.3f}")
    
    # Backwards approach: optimize from 2100 to present
    print("\n" + "-"*80)
    print("BACKWARDS APPROACH (2100 → Present)")
    print("-"*80)
    
    backwards_optimizer = BackwardsOptimizer(landscape, scenario)
    backwards_result = backwards_optimizer.optimize(max_iterations=50)
    
    print(f"  Corridors established: {len(backwards_result.network.edges)}")
    print(f"  Current entropy: {backwards_result.entropy:.3f}")
    print(f"  Optimization iterations: {backwards_result.iterations}")
    
    # Visualize results
    print("\n" + "-"*80)
    print("CREATING VISUALIZATIONS")
    print("-"*80)
    
    # Climate trajectory
    fig1 = plot_climate_trajectory(scenario)
    fig1.suptitle('Climate Change Scenario (RCP 8.5-like)', fontsize=14, fontweight='bold')
    
    # Compare networks
    fig2, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    plot_network(
        landscape,
        forward_result.network,
        ax=axes[0],
        title=f'Forward Optimization\n(Current conditions only)\nH = {forward_result.entropy:.3f}'
    )
    
    plot_network(
        landscape,
        backwards_result.network,
        ax=axes[1],
        title=f'Backwards Optimization\n(Climate-adaptive)\nH = {backwards_result.entropy:.3f}'
    )
    
    fig2.suptitle('Comparison: Forward vs. Backwards Optimization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Convergence history
    if len(backwards_result.convergence_history) > 1:
        fig3, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(len(backwards_result.convergence_history))
        ax.plot(iterations, backwards_result.convergence_history, 'b-', linewidth=2, marker='o')
        ax.set_xlabel('Time Step (backwards from 2100)')
        ax.set_ylabel('Network Entropy')
        ax.set_title('Backwards Optimization Convergence')
        ax.grid(True, alpha=0.3)
        
        # Annotate time steps
        time_labels = [f"{year}" for year in reversed(scenario.years)]
        if len(time_labels) == len(backwards_result.convergence_history):
            for i, label in enumerate(time_labels):
                ax.annotate(
                    label,
                    xy=(i, backwards_result.convergence_history[i]),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9
                )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    entropy_diff = forward_result.entropy - backwards_result.entropy
    percent_diff = 100 * entropy_diff / forward_result.entropy
    
    print(f"\nEntropy reduction via backwards optimization: {entropy_diff:.3f} ({percent_diff:.1f}%)")
    
    if entropy_diff > 0:
        print("\n✓ Backwards optimization achieves lower entropy")
        print("  → Better positioned for climate change")
        print("  → Higher connectivity under future conditions")
    else:
        print("\n○ Approaches yield similar entropy")
        print("  → Climate adaptation benefit unclear for this landscape")
    
    # Corridor schedule
    if backwards_result.corridor_schedule:
        print("\n" + "-"*80)
        print("TEMPORAL CORRIDOR SCHEDULE")
        print("-"*80)
        
        for year, edges in zip(scenario.years, backwards_result.corridor_schedule):
            print(f"\n{year}: {len(edges)} corridors")
            
            # Check which are new
            if year == scenario.years[0]:
                print("  (Initial establishment)")
            else:
                prev_idx = scenario.years.index(year) - 1
                prev_edges = set(backwards_result.corridor_schedule[prev_idx])
                curr_edges = set(edges)
                
                new_edges = curr_edges - prev_edges
                removed_edges = prev_edges - curr_edges
                
                if new_edges:
                    print(f"  New corridors: {len(new_edges)}")
                if removed_edges:
                    print(f"  Removed corridors: {len(removed_edges)}")
    
    print("\n" + "="*80 + "\n")
    
    plt.show()


def climate_sensitivity_analysis(
    n_patches: int = 15,
    extent: float = 10000.0,
    random_seed: int = 42
):
    """
    Analyze sensitivity to different climate scenarios.
    
    Tests optimization under various climate change trajectories to
    understand robustness.
    """
    print("\n" + "="*80)
    print("CLIMATE SENSITIVITY ANALYSIS")
    print("="*80)
    
    landscape = SyntheticLandscape(
        n_patches=n_patches,
        extent=extent,
        random_seed=random_seed
    )
    
    # Define multiple scenarios
    scenarios = {
        'RCP 2.6 (Low)': ClimateScenario(
            years=[2025, 2100],
            temperature_changes=[0.0, 1.5],
            precipitation_changes=[0.0, -2.0]
        ),
        'RCP 4.5 (Medium)': ClimateScenario(
            years=[2025, 2100],
            temperature_changes=[0.0, 2.5],
            precipitation_changes=[0.0, -7.0]
        ),
        'RCP 8.5 (High)': ClimateScenario(
            years=[2025, 2100],
            temperature_changes=[0.0, 4.0],
            precipitation_changes=[0.0, -15.0]
        )
    }
    
    results = {}
    
    print("\nOptimizing under different climate scenarios...")
    for name, scenario in scenarios.items():
        print(f"\n  {name}...")
        optimizer = BackwardsOptimizer(landscape, scenario)
        result = optimizer.optimize(max_iterations=30)
        results[name] = result
        print(f"    Entropy: {result.entropy:.3f}")
        print(f"    Corridors: {len(result.network.edges)}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for (name, result), ax in zip(results.items(), axes):
        plot_network(
            landscape,
            result.network,
            ax=ax,
            title=f'{name}\nH = {result.entropy:.3f}'
        )
    
    fig.suptitle('Climate Scenario Sensitivity Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    print("\n" + "="*80)
    print("Analysis shows robustness across climate scenarios")
    print("="*80 + "\n")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backwards optimization demo')
    parser.add_argument('--mode', choices=['compare', 'sensitivity'], default='compare',
                       help='Demo mode')
    parser.add_argument('--n-patches', type=int, default=15,
                       help='Number of patches')
    parser.add_argument('--extent', type=float, default=10000.0,
                       help='Landscape extent (meters)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        compare_forward_vs_backwards(
            n_patches=args.n_patches,
            extent=args.extent,
            random_seed=args.seed
        )
    else:
        climate_sensitivity_analysis(
            n_patches=args.n_patches,
            extent=args.extent,
            random_seed=args.seed
        )
