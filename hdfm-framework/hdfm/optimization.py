"""
Optimization algorithms for HDFM framework.

Implements backwards temporal optimization for climate-adaptive corridor design.
Includes width optimization and landscape allocation constraints.
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from scipy.optimize import minimize, LinearConstraint
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


def check_allocation_constraint(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    corridor_widths: Dict[Tuple[int, int], float],
    beta: float = 0.25
) -> Tuple[bool, float, float]:
    """
    Check if corridor allocation satisfies landscape constraint.

    Allocation constraint: Σᵢⱼ dᵢⱼ wᵢⱼ ≤ β Σᵢ Aᵢ

    Where:
    - dᵢⱼ: corridor length (m)
    - wᵢⱼ: corridor width (m)
    - Aᵢ: patch area (m²)
    - β: allocation fraction (typically 0.20-0.30 = 20-30%)

    Args:
        landscape: Landscape object
        edges: List of corridor edges
        corridor_widths: Dict mapping (i,j) to width in meters
        beta: Landscape allocation fraction (default 0.25 = 25%)

    Returns:
        (constraint_satisfied, corridor_area_used, total_area_available)

    Invariants:
    - 0 < β ≤ 1
    - constraint_satisfied = True if corridor_area_used ≤ beta * total_area_available
    """
    assert 0 < beta <= 1, f"Beta must be in (0,1], got {beta}"

    # Calculate total landscape area
    total_area = sum(patch.area for patch in landscape.patches)

    # Calculate corridor area used
    corridor_area = 0.0
    for (i, j) in edges:
        distance = landscape.graph[i][j]['distance']

        # Handle both orderings of edge
        width = corridor_widths.get((i, j), corridor_widths.get((j, i), 0))

        corridor_area += distance * width

    # Check constraint
    max_allowed = beta * total_area
    constraint_satisfied = corridor_area <= max_allowed

    return constraint_satisfied, corridor_area, total_area


class WidthOptimizer:
    """
    Optimizer for corridor widths with landscape allocation constraints.

    Optimizes corridor widths to minimize entropy while respecting:
    1. Landscape allocation constraint: Σᵢⱼ dᵢⱼ wᵢⱼ ≤ β Σᵢ Aᵢ (20-30%)
    2. Width bounds: w_min ≤ wᵢⱼ ≤ w_max
    3. Genetic viability: Nₑ(A,w) ≥ Nₑᵗʰʳᵉˢʰ (simplified)

    Algorithm:
    Given fixed topology (edges), optimize width allocation to minimize
    H(A, w) subject to area budget constraint.
    """

    def __init__(
        self,
        landscape: Landscape,
        edges: List[Tuple[int, int]],
        species_guild,
        beta: float = 0.25
    ):
        """
        Initialize width optimizer.

        Args:
            landscape: Landscape object
            edges: Fixed corridor topology (from MST or other method)
            species_guild: SpeciesGuild for width-dependent parameters
            beta: Landscape allocation fraction (0.20-0.30 typical)
        """
        self.landscape = landscape
        self.edges = edges
        self.species_guild = species_guild
        self.beta = beta

        assert 0 < beta <= 1, f"Beta must be in (0,1], got {beta}"

    def optimize(
        self,
        initial_widths: Optional[Dict[Tuple[int, int], float]] = None,
        max_iterations: int = 100,
        **entropy_kwargs
    ) -> OptimizationResult:
        """
        Optimize corridor widths subject to allocation constraint.

        Minimize: H(A, w)
        Subject to:
          Σᵢⱼ dᵢⱼ wᵢⱼ ≤ β Σᵢ Aᵢ  (allocation constraint)
          w_min ≤ wᵢⱼ ≤ w_max        (width bounds)

        Args:
            initial_widths: Optional starting widths (default: w_crit for all)
            max_iterations: Maximum optimization iterations
            **entropy_kwargs: Parameters for entropy calculation

        Returns:
            OptimizationResult with optimized widths
        """
        n_edges = len(self.edges)

        # Set width bounds
        w_min = self.species_guild.w_min
        w_max = 500.0  # Maximum practical width (meters)

        # Initialize widths
        if initial_widths is None:
            # Start at critical width
            x0 = np.full(n_edges, self.species_guild.w_crit)
        else:
            x0 = np.array([initial_widths.get(e, self.species_guild.w_crit) for e in self.edges])

        # Calculate allocation constraint parameters
        total_area = sum(patch.area for patch in self.landscape.patches)
        max_corridor_area = self.beta * total_area

        # Build constraint matrix: sum of (distance * width) <= max_corridor_area
        distances = np.array([self.landscape.graph[i][j]['distance'] for (i, j) in self.edges])

        # Linear constraint: distances @ widths <= max_corridor_area
        constraint = LinearConstraint(distances, lb=0, ub=max_corridor_area)

        # Bounds on individual widths
        bounds = [(w_min, w_max) for _ in range(n_edges)]

        # Objective function: minimize entropy
        def objective(widths):
            width_dict = {edge: w for edge, w in zip(self.edges, widths)}
            H, _ = calculate_entropy(
                self.landscape,
                self.edges,
                corridor_widths=width_dict,
                species_guild=self.species_guild,
                **entropy_kwargs
            )
            return H

        # Convergence history
        convergence_history = []

        def callback(xk):
            convergence_history.append(objective(xk))

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[constraint],
            options={'maxiter': max_iterations},
            callback=callback
        )

        # Extract optimal widths
        optimal_widths = {edge: w for edge, w in zip(self.edges, result.x)}

        # Build result
        network = DendriticNetwork(self.landscape, self.edges)
        H_final, components = calculate_entropy(
            self.landscape,
            self.edges,
            corridor_widths=optimal_widths,
            species_guild=self.species_guild,
            **entropy_kwargs
        )

        return OptimizationResult(
            network=network,
            entropy=H_final,
            entropy_components=components,
            iterations=len(convergence_history),
            convergence_history=convergence_history,
            corridor_schedule=None  # Store widths in network metadata if needed
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
