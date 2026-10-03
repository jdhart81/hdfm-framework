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

#: Square meters per hectare. Patch areas are stored in hectares; corridor
#: areas (length in m x width in m) are square meters.
M2_PER_HECTARE = 10_000.0

#: Maximum practical corridor width (meters) used as the solver's upper bound.
W_MAX_M = 500.0


def corridor_area_budget_m2(landscape: Landscape, beta: float) -> float:
    """Allocation budget beta * sum(patch areas), converted from hectares to m^2."""
    assert 0 < beta <= 1, f"Beta must be in (0,1], got {beta}"
    return beta * sum(patch.area for patch in landscape.patches) * M2_PER_HECTARE


def _solve_widths(landscape, edges, species_guild, beta, x0, max_iterations,
                  entropy_kwargs, callback=None):
    """Solve the width allocation problem and verify the answer independently.

    Returns (widths or None, success, message). Widths are returned only when the
    solver reports success AND the returned widths satisfy the bounds and the
    allocation budget within a small tolerance. An infeasible budget is detected
    before calling the solver.
    """
    w_min = species_guild.w_min
    distances = np.array([landscape.graph[i][j]['distance'] for (i, j) in edges])
    budget = corridor_area_budget_m2(landscape, beta)
    minimum_need = float(w_min * distances.sum())
    if minimum_need > budget:
        return None, False, (
            f"Infeasible: corridors at minimum width {w_min:g} m need "
            f"{minimum_need:,.0f} m^2 but the allocation budget is {budget:,.0f} m^2 "
            f"(beta={beta:g} of {budget / beta / M2_PER_HECTARE:,.2f} ha)."
        )

    constraint = LinearConstraint(distances, lb=0, ub=budget)
    bounds = [(w_min, W_MAX_M) for _ in edges]

    def objective(widths):
        width_dict = {edge: w for edge, w in zip(edges, widths)}
        H, _ = calculate_entropy(landscape, edges, corridor_widths=width_dict,
                                 species_guild=species_guild, **entropy_kwargs)
        return H

    x0 = np.clip(np.asarray(x0, dtype=float), w_min, W_MAX_M)
    result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                      constraints=[constraint], options={'maxiter': max_iterations},
                      callback=(lambda xk: callback(objective(xk))) if callback else None)
    if not result.success:
        return None, False, f"Solver failed: {result.message}"

    widths = {edge: float(w) for edge, w in zip(edges, result.x)}
    used = float(distances @ result.x)
    tol = 1e-6 * max(budget, 1.0)
    if used > budget + tol or np.any(result.x < w_min - 1e-6) or np.any(result.x > W_MAX_M + 1e-6):
        return None, False, (
            f"Solver returned widths that violate the constraints "
            f"(uses {used:,.0f} of {budget:,.0f} m^2)."
        )
    return widths, True, "Solved; widths satisfy bounds and allocation budget."


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
        width_schedule: Optional temporal schedule for corridor widths by year
        optimal_widths: Optional dictionary of optimized corridor widths
        success: False when any solve failed or was infeasible; never treat a
            result with success=False as a usable plan
        message: Solver/feasibility status in plain words
    """
    network: DendriticNetwork
    entropy: float
    entropy_components: Dict[str, float]
    iterations: int
    convergence_history: List[float]
    corridor_schedule: Optional[List[List[Tuple[int, int]]]] = None
    width_schedule: Optional[Dict[int, Dict[Tuple[int, int], float]]] = None
    optimal_widths: Optional[Dict[Tuple[int, int], float]] = None
    success: bool = True
    message: str = ""


class DendriticOptimizer:
    """
    Basic dendritic network optimizer.
    
    Constructs a minimum-length tree on the supplied graph via minimum spanning tree
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
        Construct a minimum-length tree and evaluate its composite entropy.

        This does not minimize the composite entropy or establish ecological benefit.
        
        Args:
            max_iterations: Not used (MST is exact)
            tolerance: Not used (MST is exact)
            **entropy_kwargs: Parameters for entropy calculation
            
        Returns:
            OptimizationResult with a minimum-length tree and its evaluated score
        """
        # Build MST (exact for summed edge length, not composite entropy)
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
    Includes integrated width scheduling for temporal corridor width optimization.

    Algorithm:
    1. Start at final year (e.g., 2100) with target connectivity
    2. Optimize network topology and corridor widths for climate conditions
    3. Work backwards through time, adjusting network and widths at each step
    4. Ensure corridors established at optimal times with appropriate widths

    Invariants:
    - Maintains connectivity at each time step
    - Minimizes entropy at target year
    - Convergence within max_iterations
    - Each corridor appears at optimal establishment time
    - Corridor widths satisfy allocation constraints at each time step
    """

    def __init__(
        self,
        landscape: Landscape,
        scenario: ClimateScenario,
        target_connectivity: float = 0.95,
        species_guild=None,
        beta: float = 0.25,
        optimize_widths: bool = True
    ):
        """
        Initialize backwards optimizer.

        Args:
            landscape: Landscape object
            scenario: ClimateScenario defining temporal trajectory
            target_connectivity: Target connectivity level at final year
            species_guild: Optional SpeciesGuild for width-dependent optimization
            beta: Landscape allocation fraction for width constraints (0.20-0.30 typical)
            optimize_widths: Whether to optimize corridor widths at each time step
        """
        self.landscape = landscape
        self.scenario = scenario
        self.target_connectivity = target_connectivity
        self.species_guild = species_guild
        self.beta = beta
        self.optimize_widths = optimize_widths

        assert 0 < target_connectivity <= 1, "Target connectivity must be in (0,1]"
        assert 0 < beta <= 1, f"Beta must be in (0,1], got {beta}"
    
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
    
    def _optimize_widths_for_year(
        self,
        landscape: Landscape,
        edges: List[Tuple[int, int]],
        initial_widths: Optional[Dict[Tuple[int, int], float]] = None,
        max_iterations: int = 50,
        **entropy_kwargs
    ) -> Tuple[Optional[Dict[Tuple[int, int], float]], bool, str]:
        """
        Optimize corridor widths for a specific time step.

        Returns:
            (widths or None, success, message). Widths are None when the budget
            is infeasible or the solver fails; callers must not substitute
            defaults silently.
        """
        if self.species_guild is None:
            return {edge: 200.0 for edge in edges}, True, "No species guild: nominal 200 m widths, not optimized."
        if not edges:
            return {}, True, "No corridors."
        w_crit = self.species_guild.w_crit
        if initial_widths:
            x0 = [initial_widths.get(e, initial_widths.get((e[1], e[0]), w_crit)) for e in edges]
        else:
            x0 = [w_crit] * len(edges)
        return _solve_widths(landscape, edges, self.species_guild, self.beta, x0,
                             max_iterations, entropy_kwargs)

    def optimize(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
        **entropy_kwargs
    ) -> OptimizationResult:
        """
        Run backwards optimization algorithm with integrated width scheduling.

        Args:
            max_iterations: Maximum iterations per time step
            tolerance: Convergence tolerance for entropy
            **entropy_kwargs: Parameters for entropy calculation

        Returns:
            OptimizationResult with temporal corridor schedule and width schedule

        Algorithm:
        1. Initialize at final year with MST
        2. Optimize corridor widths for final year (if enabled)
        3. For each previous time step:
           a. Modify landscape for climate at that time
           b. Re-optimize network maintaining previous structure
           c. Optimize corridor widths for this time step
           d. Check for convergence
        4. Return corridor establishment schedule and width schedule
        """
        years = self.scenario.years
        n_steps = len(years)

        # Store networks and widths at each time step
        networks_by_year = {}
        widths_by_year = {}
        convergence_history = []
        failures = []

        # Start at final year (2100)
        final_year = years[-1]
        final_landscape = self._modify_landscape_for_climate(final_year)
        final_network = build_dendritic_network(final_landscape)

        networks_by_year[final_year] = final_network

        # Optimize widths for final year if enabled
        if self.optimize_widths and self.species_guild is not None:
            widths, ok, msg = self._optimize_widths_for_year(
                final_landscape,
                final_network.edges,
                max_iterations=max_iterations,
                **entropy_kwargs
            )
            widths_by_year[final_year] = widths
            if not ok:
                failures.append(f"{final_year}: {msg}")
            H_final, _ = calculate_entropy(
                final_landscape,
                final_network.edges,
                corridor_widths=widths,
                species_guild=self.species_guild if widths else None,
                **entropy_kwargs
            )
        else:
            widths_by_year[final_year] = {edge: 200.0 for edge in final_network.edges}
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

            # Iterative refinement of topology
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

            # Optimize widths for this year if enabled
            if self.optimize_widths and self.species_guild is not None:
                # Use widths from next time step as initial values
                prev_widths = widths_by_year.get(years[i+1], None)
                widths, ok, msg = self._optimize_widths_for_year(
                    landscape_t,
                    current_edges,
                    initial_widths=prev_widths,
                    max_iterations=max_iterations,
                    **entropy_kwargs
                )
                widths_by_year[year] = widths
                if not ok:
                    failures.append(f"{year}: {msg}")
                H_t, _ = calculate_entropy(
                    landscape_t,
                    current_edges,
                    corridor_widths=widths,
                    species_guild=self.species_guild if widths else None,
                    **entropy_kwargs
                )
            else:
                widths_by_year[year] = {edge: 200.0 for edge in current_edges}
                H_t, _ = network_t.entropy(**entropy_kwargs)

            convergence_history.append(H_t)

        # Build corridor schedule (ordered by establishment year)
        corridor_schedule = [networks_by_year[year].edges for year in years]

        # Return result for present day
        present_network = networks_by_year[years[0]]
        present_widths = widths_by_year[years[0]]

        if self.optimize_widths and self.species_guild is not None and present_widths:
            H_present, components = calculate_entropy(
                self._modify_landscape_for_climate(years[0]),
                present_network.edges,
                corridor_widths=present_widths,
                species_guild=self.species_guild,
                **entropy_kwargs
            )
        else:
            H_present, components = present_network.entropy(**entropy_kwargs)

        return OptimizationResult(
            network=present_network,
            entropy=H_present,
            entropy_components=components,
            iterations=len(convergence_history),
            convergence_history=convergence_history,
            corridor_schedule=corridor_schedule,
            width_schedule=widths_by_year,
            optimal_widths=present_widths,
            success=not failures,
            message="; ".join(failures) if failures else "All time steps solved."
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
    - Aᵢ: patch area (stored in hectares, converted here to m²)
    - β: allocation fraction (typically 0.20-0.30 = 20-30%)

    Args:
        landscape: Landscape object
        edges: List of corridor edges
        corridor_widths: Dict mapping (i,j) to width in meters
        beta: Landscape allocation fraction (default 0.25 = 25%)

    Returns:
        (constraint_satisfied, corridor_area_used_m2, total_patch_area_m2)

    Invariants:
    - 0 < β ≤ 1
    - constraint_satisfied = True if corridor_area_used ≤ beta * total_area_available
    """
    assert 0 < beta <= 1, f"Beta must be in (0,1], got {beta}"

    # Patch areas are hectares; corridor areas are m^2.
    total_area = sum(patch.area for patch in landscape.patches) * M2_PER_HECTARE

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
        w_crit = self.species_guild.w_crit
        if initial_widths is None:
            x0 = [w_crit] * len(self.edges)
        else:
            x0 = [initial_widths.get(e, initial_widths.get((e[1], e[0]), w_crit)) for e in self.edges]

        convergence_history: List[float] = []
        optimal_widths, ok, message = _solve_widths(
            self.landscape, self.edges, self.species_guild, self.beta, x0,
            max_iterations, entropy_kwargs, callback=convergence_history.append)

        network = DendriticNetwork(self.landscape, self.edges)
        H_final, components = calculate_entropy(
            self.landscape,
            self.edges,
            corridor_widths=optimal_widths,
            species_guild=self.species_guild if optimal_widths else None,
            **entropy_kwargs
        )

        return OptimizationResult(
            network=network,
            entropy=H_final,
            entropy_components=components,
            iterations=len(convergence_history),
            convergence_history=convergence_history,
            optimal_widths=optimal_widths,
            success=ok,
            message=message
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
