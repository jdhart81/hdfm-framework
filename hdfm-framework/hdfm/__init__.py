"""
Hierarchical Dendritic Forest Management (HDFM) Framework

A computational toolkit for entropy-minimizing corridor network optimization
with backwards climate adaptation.

Core components:
- Landscape representation and graph construction
- Information-theoretic entropy calculation
- Dendritic network optimization (MST-based)
- Backwards temporal optimization
- Synthetic landscape validation
- Visualization tools
"""

__version__ = "0.1.0"
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
    connectivity_constraint,
    forest_topology_penalty,
    disturbance_response_penalty
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
    OptimizationResult
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

    # Functions
    'calculate_entropy',
    'movement_entropy',
    'connectivity_constraint',
    'forest_topology_penalty',
    'disturbance_response_penalty',
    'build_dendritic_network',
    'compare_network_topologies',
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
