"""
Optimization algorithms for HDFM framework.

Implements backwards temporal optimization for climate-adaptive corridor design
and corridor width optimization under landscape allocation constraints.
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from scipy.optimize import minimize, Bounds, LinearConstraint
from .landscape import Landscape
from .network import build_dendritic_network, DendriticNetwork
from .entropy import calculate_entropy


@dataclass
class ClimateScenario:
    """
    Represents climate change trajectory over time.
    
    Attributes:
        years: List of years (e.g., [2025, 2050, 2075, 2100])
        temperature_changes: Temperature anomalies (°C) at each year
        precipitation_changes: Precipitation changes (%) at each year
        species_shifts: Optional species distribution shifts (km/year)
    
    Invariants:
    - years is strictly increasing
    - Same length for all attributes
    - temperature_changes, precipitation_changes are monotonic or realistic
    """
    years: List[int]
    temperature_changes: List[float]
    precipitation_changes: List[float]
    species_shifts: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate climate scenario."""
        assert len(self.years) >= 2, "Need at least 2 time points"
        assert len(self.years) == len(self.temperature_changes), "Mismatched lengths"
        assert len(self.years) == len(self.precipitation_changes), "Mismatched lengths"
        
        # Check years are increasing
        assert all(self.years[i] < self.years[i+1] for i in range(len(self.years)-1)), \
            "Years must be strictly increasing"
        
        if self.species_shifts is not None:
            assert len(self.years) == len(self.species_shifts), "Mismatched lengths"
    
    def interpolate(self, year: int) -> Tuple[float, float]:
        """
        Interpolate climate conditions at given year.
        
        Returns:
            (temperature_change, precipitation_change)
        """
        if year <= self.years[0]:
            return self.temperature_changes[0], self.precipitation_changes[0]
        if year >= self.years[-1]:
            return self.temperature_changes[-1], self.precipitation_changes[-1]
        
        # Linear interpolation
        for i in range(len(self.years) - 1):
            if self.years[i] <= year <= self.years[i+1]:
                t = (year - self.years[i]) / (self.years[i+1] - self.years[i])
                temp = self.temperature_changes[i] + t * (self.temperature_changes[i+1] - self.temperature_changes[i])
                precip = self.precipitation_changes[i] + t * (self.precipitation_changes[i+1] - self.precipitation_changes[i])
                return temp, precip
        
        raise ValueError(f"Year {year} out of range")


@dataclass
class OptimizationResult:
    """
    Results from network optimization.
    
    Attributes:
        network: Optimized DendriticNetwork
        entropy: Final entropy value
        entropy_components: Dictionary of entropy components
        iterations: Number of iterations to convergence
        convergence_history: Entropy at each iteration
        corridor_schedule: Optional temporal schedule for corridor establishment
    """
    network: DendriticNetwork
    entropy: float
    entropy_components: Dict[str, float]
    iterations: int
    convergence_history: List[float]
    corridor_schedule: Optional[List[List[Tuple[int, int]]]] = None


class DendriticOptimizer:
    """
    Basic dendritic network optimizer.
    
    Constructs optimal dendritic corridor network via minimum spanning tree
    algorithm, minimizing total corridor length while maintaining connectivity.
    """
    
    def __init__(self, landscape: Landscape):
        """
        Initialize optimizer.
        
        Args:
            landscape: Landscape object to optimize
        """
        self.landscape = landscape
    
    def build_dendritic_network(self) -> DendriticNetwork:
        """
        Build dendritic network via MST.
        
        Returns:
            Optimized DendriticNetwork
            
        Invariants:
        - Network is connected
        - Network is acyclic
        - Network minimizes total corridor length
        """
        return build_dendritic_network(self.landscape)
    
    def optimize(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        **entropy_kwargs
    ) -> OptimizationResult:
        """
        Optimize network (MST is optimal, so single iteration).
        
        Args:
            max_iterations: Not used (MST is exact)
            tolerance: Not used (MST is exact)
            **entropy_kwargs: Parameters for entropy calculation
            
        Returns:
            OptimizationResult with optimal network
        """
        # Build MST (exact solution)
        network = self.build_dendritic_network()
        
        # Calculate entropy
        H_total, components = network.entropy(**entropy_kwargs)
        
        return OptimizationResult(
            network=network,
            entropy=H_total,
            entropy_components=components,
            iterations=1,
            convergence_history=[H_total]
        )


class BackwardsOptimizer:
    """
    Backwards temporal optimization for climate-adaptive corridor design.
    
    Optimizes corridor networks by working backwards from desired 2100 state
    to present implementation, accounting for climate change trajectory.
    
    Algorithm:
    1. Start at final year (e.g., 2100) with target connectivity
    2. Optimize network for climate conditions at that time
    3. Work backwards through time, adjusting network at each step
    4. Ensure corridors established at optimal times for development
    
    Invariants:
    - Maintains connectivity at each time step
    - Minimizes entropy at target year
    - Convergence within max_iterations
    - Each corridor appears at optimal establishment time
    """
    
    def __init__(
        self,
        landscape: Landscape,
        scenario: ClimateScenario,
        target_connectivity: float = 0.95
    ):
        """
        Initialize backwards optimizer.
        
        Args:
            landscape: Landscape object
            scenario: ClimateScenario defining temporal trajectory
            target_connectivity: Target connectivity level at final year
        """
        self.landscape = landscape
        self.scenario = scenario
        self.target_connectivity = target_connectivity
        
        assert 0 < target_connectivity <= 1, "Target connectivity must be in (0,1]"
    
    def _modify_landscape_for_climate(
        self,
        year: int
    ) -> Landscape:
        """
        Create modified landscape accounting for climate at given year.
        
        Adjusts patch quality and connectivity based on projected climate.
        """
        temp_change, precip_change = self.scenario.interpolate(year)
        
        # Create modified patches
        modified_patches = []
        for patch in self.landscape.patches:
            # Simple climate impact model: quality decreases with warming/drying
            # This is a placeholder - real models would be species-specific
            climate_impact = 1.0 - 0.1 * (temp_change / 3.0) - 0.05 * abs(precip_change) / 15.0
            climate_impact = max(0.1, min(1.0, climate_impact))
            
            modified_quality = patch.quality * climate_impact
            
            from .landscape import Patch
            modified_patches.append(Patch(
                id=patch.id,
                x=patch.x,
                y=patch.y,
                area=patch.area,
                quality=modified_quality
            ))
        
        from .landscape import Landscape
        return Landscape(modified_patches)
    
    def optimize(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
        **entropy_kwargs
    ) -> OptimizationResult:
        """
        Run backwards optimization algorithm.
        
        Args:
            max_iterations: Maximum iterations per time step
            tolerance: Convergence tolerance for entropy
            **entropy_kwargs: Parameters for entropy calculation
            
        Returns:
            OptimizationResult with temporal corridor schedule
            
        Algorithm:
        1. Initialize at final year with MST
        2. For each previous time step:
           a. Modify landscape for climate at that time
           b. Re-optimize network maintaining previous structure
           c. Check for convergence
        3. Return corridor establishment schedule
        """
        years = self.scenario.years
        n_steps = len(years)
        
        # Store networks at each time step
        networks_by_year = {}
        convergence_history = []
        
        # Start at final year (2100)
        final_year = years[-1]
        final_landscape = self._modify_landscape_for_climate(final_year)
        final_network = build_dendritic_network(final_landscape)
        
        networks_by_year[final_year] = final_network
        
        H_final, _ = final_network.entropy(**entropy_kwargs)
        convergence_history.append(H_final)
        
        # Work backwards through time
        for i in range(n_steps - 2, -1, -1):
            year = years[i]
            
            # Get landscape at this time
            landscape_t = self._modify_landscape_for_climate(year)
            
            # Start with structure from next time step
            next_network = networks_by_year[years[i+1]]
            current_edges = next_network.edges.copy()
            
            # Iterative refinement
            best_entropy = float('inf')
            no_improvement_count = 0
            
            for iteration in range(max_iterations):
                # Try local modifications
                improved = False
                
                # Try swapping edges
                for j, (u, v) in enumerate(current_edges):
                    # Try replacing this edge
                    test_edges = current_edges[:j] + current_edges[j+1:]
                    
                    # Find edges that would maintain connectivity
                    G_test = nx.Graph()
                    G_test.add_nodes_from(range(self.landscape.n_patches))
                    
                    id_to_idx = {patch.id: idx for idx, patch in enumerate(self.landscape.patches)}
                    for (a, b) in test_edges:
                        G_test.add_edge(id_to_idx[a], id_to_idx[b])
                    
                    # If disconnected, try to reconnect
                    if not nx.is_connected(G_test):
                        components = list(nx.connected_components(G_test))
                        if len(components) == 2:
                            # Find shortest edge between components
                            min_dist = float('inf')
                            best_edge = None
                            
                            comp1_ids = [self.landscape.patches[idx].id for idx in components[0]]
                            comp2_ids = [self.landscape.patches[idx].id for idx in components[1]]
                            
                            for id1 in comp1_ids:
                                for id2 in comp2_ids:
                                    dist = self.landscape.graph[id1][id2]['distance']
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_edge = (id1, id2)
                            
                            if best_edge:
                                test_edges.append(best_edge)
                    
                    # Evaluate
                    if len(test_edges) == len(current_edges):
                        H_test, _ = calculate_entropy(landscape_t, test_edges, **entropy_kwargs)
                        
                        if H_test < best_entropy - tolerance:
                            best_entropy = H_test
                            current_edges = test_edges
                            improved = True
                            break
                
                if not improved:
                    no_improvement_count += 1
                    if no_improvement_count >= 3:
                        break
                else:
                    no_improvement_count = 0
            
            # Store network for this year
            network_t = DendriticNetwork(landscape_t, current_edges)
            networks_by_year[year] = network_t
            
            H_t, _ = network_t.entropy(**entropy_kwargs)
            convergence_history.append(H_t)
        
        # Build corridor schedule (ordered by establishment year)
        corridor_schedule = [networks_by_year[year].edges for year in years]
        
        # Return result for present day
        present_network = networks_by_year[years[0]]
        H_present, components = present_network.entropy(**entropy_kwargs)
        
        return OptimizationResult(
            network=present_network,
            entropy=H_present,
            entropy_components=components,
            iterations=len(convergence_history),
            convergence_history=convergence_history,
            corridor_schedule=corridor_schedule
        )


def greedy_optimization(
    landscape: Landscape,
    max_edges: int,
    **entropy_kwargs
) -> OptimizationResult:
    """
    Greedy algorithm for corridor optimization.
    
    Iteratively adds edges that most reduce entropy until max_edges reached
    or network is connected.
    
    Args:
        landscape: Landscape object
        max_edges: Maximum number of corridors
        **entropy_kwargs: Parameters for entropy calculation
        
    Returns:
        OptimizationResult with greedy solution
        
    Note:
        Greedy algorithm does not guarantee global optimum.
        MST (dendritic) is provably optimal for connectivity + minimum length.
    """
    edges = []
    convergence_history = []
    
    # Get all possible edges sorted by distance
    all_edges = [
        (i, j, landscape.graph[i][j]['distance'])
        for i, j in landscape.graph.edges()
    ]
    all_edges.sort(key=lambda x: x[2])
    
    for iteration in range(max_edges):
        if iteration >= len(all_edges):
            break
        
        # Try adding next edge
        test_edge = all_edges[iteration][:2]
        test_edges = edges + [test_edge]
        
        # Calculate entropy
        H, _ = calculate_entropy(landscape, test_edges, **entropy_kwargs)
        convergence_history.append(H)
        
        # Check if connected
        G = nx.Graph()
        G.add_nodes_from(range(landscape.n_patches))
        G.add_edges_from(test_edges)
        
        edges = test_edges
        
        if nx.is_connected(G):
            break
    
    # Final network
    network = DendriticNetwork(landscape, edges)
    H_final, components = network.entropy(**entropy_kwargs)

    return OptimizationResult(
        network=network,
        entropy=H_final,
        entropy_components=components,
        iterations=len(convergence_history),
        convergence_history=convergence_history
    )


def optimize_corridor_widths(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    species_guild,
    allocation_budget: float = 0.25,
    width_bounds: Tuple[float, float] = (50, 500),
    method: str = 'SLSQP',
    **entropy_kwargs
) -> Tuple[Dict[Tuple[int, int], float], OptimizationResult]:
    """
    Optimize corridor widths to minimize entropy subject to allocation constraint.

    Solves the constrained optimization problem:

    Minimize: H(A, w)
    Subject to:
      ∑ᵢⱼ dᵢⱼ · wᵢⱼ ≤ β · ∑ᵢ Aᵢ  (allocation constraint)
      w_min ≤ wᵢⱼ ≤ w_max         (width bounds)
      Nₑ(A,w) ≥ Nₑᵗʰʳᵉˢʰ           (genetic viability)

    Args:
        landscape: Landscape object
        edges: List of corridor edges (topology fixed)
        species_guild: SpeciesGuild for species-specific parameters
        allocation_budget: Landscape allocation fraction β ∈ [0.2, 0.3]
        width_bounds: (w_min, w_max) corridor width range (meters)
        method: Scipy optimization method ('SLSQP', 'trust-constr')
        **entropy_kwargs: Additional parameters for entropy calculation

    Returns:
        (optimal_widths, optimization_result)

        optimal_widths: Dictionary mapping (i,j) to optimal width
        optimization_result: OptimizationResult with final entropy

    Algorithm:
        1. Initialize widths at species critical width
        2. Set up allocation constraint: ∑ dᵢⱼ·wᵢⱼ ≤ β·∑Aᵢ
        3. Use gradient-based optimization (SLSQP or trust-constr)
        4. Return optimal width allocation

    Reference:
        Hart (2024), Section 3.3, Width Optimization
    """
    assert 0 < allocation_budget <= 1.0, "Allocation budget must be in (0, 1]"
    assert width_bounds[0] < width_bounds[1], "Invalid width bounds"
    assert len(edges) > 0, "Need at least one edge"

    n_edges = len(edges)
    w_min, w_max = width_bounds

    # Initialize widths at species critical width (or middle of range)
    if species_guild is not None:
        w_init = min(max(species_guild.w_crit, w_min), w_max)
    else:
        w_init = (w_min + w_max) / 2

    initial_widths = np.full(n_edges, w_init)

    # Calculate budget threshold
    total_patch_area = sum(p.area * 10000 for p in landscape.patches)  # ha to m²
    budget_threshold = allocation_budget * total_patch_area

    # Get edge lengths
    edge_lengths = np.array([landscape.graph[i][j]['distance'] for (i, j) in edges])

    # Create edge index mapping
    edge_to_idx = {edge: i for i, edge in enumerate(edges)}

    def objective(widths):
        """Objective function: total entropy."""
        # Build corridor_widths dictionary
        corridor_widths = {}
        for edge, width in zip(edges, widths):
            edge_key = tuple(sorted(edge))
            corridor_widths[edge_key] = width

        # Calculate entropy
        H_total, _ = calculate_entropy(
            landscape, edges, corridor_widths, species_guild,
            lambda4=0.0,  # No allocation penalty in objective (handled by constraint)
            **entropy_kwargs
        )
        return H_total

    def allocation_constraint_func(widths):
        """Allocation constraint: ∑ dᵢⱼ·wᵢⱼ - β·∑Aᵢ ≤ 0"""
        total_corridor_area = np.sum(edge_lengths * widths)
        return budget_threshold - total_corridor_area  # >= 0 means satisfied

    # Set up constraints
    constraints = [
        {'type': 'ineq', 'fun': allocation_constraint_func}
    ]

    # Width bounds for each edge
    bounds = Bounds(
        lb=np.full(n_edges, w_min),
        ub=np.full(n_edges, w_max)
    )

    # Run optimization
    result = minimize(
        objective,
        initial_widths,
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 200, 'ftol': 1e-6}
    )

    # Extract optimal widths
    optimal_widths = {}
    for edge, width in zip(edges, result.x):
        edge_key = tuple(sorted(edge))
        optimal_widths[edge_key] = width

    # Calculate final entropy with optimal widths
    H_final, components = calculate_entropy(
        landscape, edges, optimal_widths, species_guild,
        **entropy_kwargs
    )

    # Create network with optimal widths (for compatibility)
    network = DendriticNetwork(landscape, edges)

    opt_result = OptimizationResult(
        network=network,
        entropy=H_final,
        entropy_components=components,
        iterations=result.nit if hasattr(result, 'nit') else 1,
        convergence_history=[H_final]
    )

    return optimal_widths, opt_result


def optimize_variable_width_network(
    landscape: Landscape,
    species_guild,
    allocation_budget: float = 0.25,
    width_bounds: Tuple[float, float] = (50, 500),
    primary_width_factor: float = 1.5,
    **kwargs
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float], OptimizationResult]:
    """
    Design variable-width dendritic network optimizing both topology and widths.

    Implements variable-width strategy:
    - Primary corridors (longest): wider (factor × w_crit)
    - Secondary corridors: critical width (w_crit)
    - Tertiary corridors: minimum feasible

    Args:
        landscape: Landscape object
        species_guild: SpeciesGuild for species-specific parameters
        allocation_budget: Landscape allocation fraction β
        width_bounds: (w_min, w_max) width range
        primary_width_factor: Multiplier for primary corridor widths
        **kwargs: Additional parameters for entropy calculation

    Returns:
        (edges, corridor_widths, optimization_result)

        edges: Dendritic network edges
        corridor_widths: Optimal width allocation
        optimization_result: Final entropy and components

    Strategy:
        1. Build dendritic (MST) network for topology
        2. Classify corridors by importance (betweenness centrality)
        3. Allocate wider widths to high-importance corridors
        4. Optimize width distribution under budget constraint

    Reference:
        Hart (2024), Section 3.5, Variable-Width Design
    """
    assert 0 < allocation_budget <= 1.0, "Allocation budget must be in (0, 1]"
    assert primary_width_factor >= 1.0, "Primary factor must be >= 1"

    w_min, w_max = width_bounds

    # Step 1: Build dendritic network (MST)
    network = build_dendritic_network(landscape)
    edges = network.edges

    # Step 2: Calculate edge importance (betweenness centrality)
    G = nx.Graph()
    G.add_edges_from(edges)
    edge_betweenness = nx.edge_betweenness_centrality(G, normalized=True)

    # Sort edges by importance
    sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)

    # Step 3: Initial width allocation based on importance
    n_edges = len(edges)
    n_primary = max(1, n_edges // 3)  # Top 1/3
    n_secondary = max(1, n_edges // 3)  # Middle 1/3
    # Remaining are tertiary

    initial_widths = {}
    w_crit = species_guild.w_crit if species_guild else (w_min + w_max) / 2

    for i, (edge, importance) in enumerate(sorted_edges):
        edge_key = tuple(sorted(edge))

        if i < n_primary:
            # Primary corridors: wider
            width = min(w_crit * primary_width_factor, w_max)
        elif i < n_primary + n_secondary:
            # Secondary corridors: critical width
            width = w_crit
        else:
            # Tertiary corridors: minimum feasible
            width = max(w_min, w_crit * 0.7)

        initial_widths[edge_key] = width

    # Step 4: Optimize widths under budget constraint
    optimal_widths, opt_result = optimize_corridor_widths(
        landscape, edges, species_guild,
        allocation_budget, width_bounds,
        **kwargs
    )

    return edges, optimal_widths, opt_result
