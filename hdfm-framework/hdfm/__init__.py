"""
Hierarchical Dendritic Forest Management (HDFM) Framework

A computational toolkit for entropy-minimizing corridor network optimization
with backwards climate adaptation and width-dependent movement analysis.

Core components:
- Landscape representation and graph construction
- Width-dependent information-theoretic entropy calculation
- Dendritic network optimization (MST-based)
- Corridor width optimization under allocation constraints
- Effective population size calculation (island model)
- Backwards temporal optimization for climate adaptation
- Species-specific movement parameters
- Synthetic landscape validation
- Visualization tools
"""

__version__ = "1.0.0"
__author__ = "Justin Hart"
__email__ = "viridisnorthllc@gmail.com"

from .landscape import Landscape, SyntheticLandscape, Patch
from .species import (
    SpeciesGuild,
    SPECIES_GUILDS,
    DEFAULT_GUILD,
    get_guild,
    list_guilds,
    print_guild_summary
)
from .entropy import (
    calculate_entropy,
    movement_entropy,
    entropy_rate,
    stationary_distribution,
    connectivity_constraint,
    effective_population_size,
    forest_topology_penalty,
    disturbance_response_penalty,
    landscape_allocation_constraint
)
from .network import (
    DendriticNetwork,
    build_dendritic_network,
    compare_network_topologies,
    NetworkTopology
)
from .optimization import (
    DendriticOptimizer,
    BackwardsOptimizer,
    ClimateScenario,
    OptimizationResult,
    optimize_corridor_widths,
    optimize_variable_width_network
)
from .validation import (
    validate_network,
    run_comparative_analysis,
    convergence_test,
    monte_carlo_validation,
    statistical_comparison,
    print_validation_summary,
    validate_dendritic_optimality
)
from .visualization import (
    plot_landscape,
    plot_network,
    plot_entropy_surface,
    plot_optimization_trace
)

__all__ = [
    # Core classes
    'Landscape',
    'SyntheticLandscape',
    'Patch',
    'DendriticNetwork',
    'DendriticOptimizer',
    'BackwardsOptimizer',
    'ClimateScenario',
    'OptimizationResult',
    'NetworkTopology',

    # Species parameters
    'SpeciesGuild',
    'SPECIES_GUILDS',
    'DEFAULT_GUILD',
    'get_guild',
    'list_guilds',
    'print_guild_summary',

    # Entropy functions
    'calculate_entropy',
    'movement_entropy',
    'entropy_rate',
    'stationary_distribution',
    'connectivity_constraint',
    'effective_population_size',
    'forest_topology_penalty',
    'disturbance_response_penalty',
    'landscape_allocation_constraint',

    # Network and optimization functions
    'build_dendritic_network',
    'compare_network_topologies',
    'optimize_corridor_widths',
    'optimize_variable_width_network',

    # Validation functions
    'validate_network',
    'run_comparative_analysis',
    'convergence_test',
    'monte_carlo_validation',
    'statistical_comparison',
    'print_validation_summary',
    'validate_dendritic_optimality',

    # Visualization
    'plot_landscape',
    'plot_network',
    'plot_entropy_surface',
    'plot_optimization_trace',
]
