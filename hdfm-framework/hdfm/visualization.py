"""
Visualization tools for HDFM framework.

Provides plotting functions for landscapes, networks, entropy surfaces,
and optimization traces.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple, Dict, Optional
from .landscape import Landscape
from .network import DendriticNetwork


def plot_landscape(
    landscape: Landscape,
    ax: Optional[plt.Axes] = None,
    show_labels: bool = True,
    patch_size_scale: float = 5.0
) -> plt.Axes:
    """
    Plot landscape patches.
    
    Args:
        landscape: Landscape to plot
        ax: Matplotlib axes (creates new if None)
        show_labels: Whether to show patch ID labels
        patch_size_scale: Scaling factor for patch sizes
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot patches
    for patch in landscape.patches:
        # Size proportional to area, color by quality
        size = patch.area * patch_size_scale
        color = plt.cm.YlGn(patch.quality)
        
        ax.scatter(patch.x, patch.y, s=size, c=[color], alpha=0.7, edgecolors='black', linewidth=1.5)
        
        if show_labels:
            ax.text(patch.x, patch.y, str(patch.id), ha='center', va='center', fontsize=8)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Landscape Patches')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_network(
    landscape: Landscape,
    network,
    ax: Optional[plt.Axes] = None,
    show_labels: bool = True,
    patch_size_scale: float = 5.0,
    edge_width_scale: float = 2.0,
    title: str = "Corridor Network"
) -> plt.Axes:
    """
    Plot landscape with corridor network overlay.
    
    Args:
        landscape: Landscape object
        network: Network object (or list of edges)
        ax: Matplotlib axes
        show_labels: Whether to show patch labels
        patch_size_scale: Scaling for patch sizes
        edge_width_scale: Scaling for edge widths
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract edges
    if hasattr(network, 'edges'):
        edges = network.edges
    else:
        edges = network
    
    # Plot patches
    plot_landscape(landscape, ax=ax, show_labels=show_labels, patch_size_scale=patch_size_scale)
    
    # Plot corridors
    for (i, j) in edges:
        patch_i = landscape.get_patch(i)
        patch_j = landscape.get_patch(j)
        
        ax.plot(
            [patch_i.x, patch_j.x],
            [patch_i.y, patch_j.y],
            'b-',
            linewidth=edge_width_scale,
            alpha=0.6,
            zorder=1
        )
    
    ax.set_title(title)
    
    return ax


def plot_entropy_surface(
    landscape: Landscape,
    resolution: int = 50,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot entropy surface over landscape.
    
    Visualizes how entropy varies across different potential corridor
    configurations.
    
    Args:
        landscape: Landscape object
        resolution: Grid resolution
        ax: Matplotlib axes
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    min_x, max_x, min_y, max_y = landscape.bounds()
    
    # Create grid
    x = np.linspace(min_x, max_x, resolution)
    y = np.linspace(min_y, max_y, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate distance-based entropy proxy
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            # Sum of inverse squared distances to patches
            point = np.array([X[i,j], Y[i,j]])
            distances = np.array([
                np.sqrt((p.x - point[0])**2 + (p.y - point[1])**2)
                for p in landscape.patches
            ])
            Z[i,j] = np.sum(1.0 / (distances + 1.0)**2)
    
    # Plot contours
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, ax=ax, label='Connectivity potential')
    
    # Overlay patches
    for patch in landscape.patches:
        ax.scatter(patch.x, patch.y, c='red', s=100, marker='o', edgecolors='white', linewidth=2)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Landscape Connectivity Surface')
    ax.set_aspect('equal')
    
    return ax


def plot_optimization_trace(
    convergence_history: List[float],
    ax: Optional[plt.Axes] = None,
    title: str = "Optimization Convergence"
) -> plt.Axes:
    """
    Plot optimization convergence trace.
    
    Args:
        convergence_history: List of entropy values by iteration
        ax: Matplotlib axes
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(1, len(convergence_history) + 1)
    
    ax.plot(iterations, convergence_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Entropy H(L)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Annotate final value
    final_H = convergence_history[-1]
    ax.annotate(
        f'Final: {final_H:.3f}',
        xy=(len(convergence_history), final_H),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7)
    )
    
    return ax


def plot_topology_comparison(
    results: Dict[str, Dict[str, np.ndarray]],
    metric: str = 'H_total',
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot comparison of network topologies.
    
    Creates violin plot and bar chart comparing entropy across topologies.
    
    Args:
        results: Results from monte_carlo_validation
        metric: Metric to plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    topologies = list(results.keys())
    values = [results[topo][metric] for topo in topologies]
    
    # Violin plot
    parts = ax1.violinplot(values, positions=range(len(topologies)), showmeans=True, showmedians=True)
    ax1.set_xticks(range(len(topologies)))
    ax1.set_xticklabels([topo.capitalize() for topo in topologies], rotation=45, ha='right')
    ax1.set_ylabel(f'{metric}')
    ax1.set_title('Distribution of Entropy Across Topologies')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Bar chart with error bars
    means = [np.mean(v) for v in values]
    stds = [np.std(v) for v in values]
    
    colors = ['green' if topo == 'dendritic' else 'steelblue' for topo in topologies]
    ax2.bar(range(len(topologies)), means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(topologies)))
    ax2.set_xticklabels([topo.capitalize() for topo in topologies], rotation=45, ha='right')
    ax2.set_ylabel(f'Mean {metric}')
    ax2.set_title('Mean Entropy by Topology (±1 SD)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Highlight dendritic as optimal
    if 'dendritic' in topologies:
        idx = topologies.index('dendritic')
        ax2.annotate(
            'Optimal',
            xy=(idx, means[idx]),
            xytext=(0, -20),
            textcoords='offset points',
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='green', lw=2)
        )
    
    plt.tight_layout()
    return fig


def plot_climate_trajectory(
    scenario,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot climate scenario trajectory.
    
    Args:
        scenario: ClimateScenario object
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    years = scenario.years
    
    # Temperature
    ax1.plot(years, scenario.temperature_changes, 'r-', linewidth=2, marker='o')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Temperature Change (°C)')
    ax1.set_title('Projected Temperature Anomaly')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Precipitation
    ax2.plot(years, scenario.precipitation_changes, 'b-', linewidth=2, marker='o')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Precipitation Change (%)')
    ax2.set_title('Projected Precipitation Change')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_validation_figure(
    landscape: Landscape,
    results: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create comprehensive validation figure.
    
    Shows landscape, dendritic network, alternative topologies, and
    entropy comparison.
    
    Args:
        landscape: Landscape object
        results: Results from compare_network_topologies
        figsize: Figure size
        
    Returns:
        Matplotlib figure with multiple subplots
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main landscape with dendritic network
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    from .network import build_dendritic_network
    dendritic = build_dendritic_network(landscape)
    plot_network(landscape, dendritic, ax=ax1, title='Dendritic Network (MST)')
    
    # Entropy comparison bar chart
    ax2 = fig.add_subplot(gs[0, 2])
    topologies = list(results.keys())
    entropies = [results[topo]['H_total'] for topo in topologies]
    colors = ['green' if topo == 'dendritic' else 'gray' for topo in topologies]
    ax2.barh(range(len(topologies)), entropies, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(topologies)))
    ax2.set_yticklabels([t.capitalize() for t in topologies])
    ax2.set_xlabel('Entropy H(L)')
    ax2.set_title('Topology Comparison')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Edge count comparison
    ax3 = fig.add_subplot(gs[1, 2])
    edge_counts = [results[topo]['n_edges'] for topo in topologies]
    ax3.barh(range(len(topologies)), edge_counts, color='steelblue', alpha=0.7)
    ax3.set_yticks(range(len(topologies)))
    ax3.set_yticklabels([t.capitalize() for t in topologies])
    ax3.set_xlabel('Number of Corridors')
    ax3.set_title('Network Complexity')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Entropy components for dendritic
    ax4 = fig.add_subplot(gs[2, :])
    if 'dendritic' in results:
        components = ['H_mov', 'C', 'F', 'D']
        values = [results['dendritic'][c] for c in components]
        labels = ['Movement', 'Connectivity', 'Topology', 'Response Time']
        
        ax4.bar(labels, values, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
        ax4.set_ylabel('Entropy Component Value')
        ax4.set_title('Entropy Components for Dendritic Network')
        ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('HDFM Validation: Dendritic Network Optimality', fontsize=16, fontweight='bold')
    
    return fig
