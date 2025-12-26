"""
Hierarchical Dendritic Forest Management (HDFM) Framework

A computational toolkit for entropy-minimizing corridor network optimization
with backwards climate adaptation.

Core components:
- Landscape representation and graph construction
- Information-theoretic entropy calculation
- Dendritic network optimization (MST-based)
- Backwards temporal optimization
- Genetic population dynamics and effective population size (Nₑ) tracking
- Robustness analysis with looped topologies and redundancy scoring
- Synthetic landscape validation
- Visualization tools
"""

__version__ = "0.2.1"
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
    calculate_entropy_rate,
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
    WidthOptimizer,
    ClimateScenario,
    OptimizationResult,
    check_allocation_constraint
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
from .genetics import (
    calculate_effective_population_size,
    calculate_migration_rate,
    calculate_coancestry_coefficient,
    check_genetic_viability,
    calculate_inbreeding_coefficient,
    calculate_genetic_diversity_loss
)
from .robustness import (
    RobustnessMetrics,
    calculate_two_edge_connectivity,
    calculate_edge_redundancy_scores,
    calculate_failure_probability,
    add_strategic_loops,
    calculate_robustness_metrics,
    pareto_frontier_analysis
)

__all__ = [
    # Core classes
    'Landscape',
    'SyntheticLandscape',
    'Patch',
    'DendriticNetwork',
    'DendriticOptimizer',
    'BackwardsOptimizer',
    'WidthOptimizer',
    'ClimateScenario',
    'OptimizationResult',
    'NetworkTopology',
    'RobustnessMetrics',

    # Species parameters
    'SpeciesGuild',
    'SPECIES_GUILDS',
    'DEFAULT_GUILD',
    'get_guild',
    'list_guilds',
    'print_guild_summary',

    # Entropy functions
    'calculate_entropy',
    'calculate_entropy_rate',
    'movement_entropy',
    'connectivity_constraint',
    'forest_topology_penalty',
    'disturbance_response_penalty',

    # Genetic functions
    'calculate_effective_population_size',
    'calculate_migration_rate',
    'calculate_coancestry_coefficient',
    'check_genetic_viability',
    'calculate_inbreeding_coefficient',
    'calculate_genetic_diversity_loss',

    # Robustness functions
    'calculate_two_edge_connectivity',
    'calculate_edge_redundancy_scores',
    'calculate_failure_probability',
    'add_strategic_loops',
    'calculate_robustness_metrics',
    'pareto_frontier_analysis',

    # Network and optimization functions
    'check_allocation_constraint',
    'build_dendritic_network',
    'compare_network_topologies',

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
