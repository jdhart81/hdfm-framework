"""
Information-theoretic entropy calculations for HDFM framework.

Implements the core entropy formulation:
H(L) = H_mov + λ₁·C(L) + λ₂·F(L) + λ₃·D(L)

Where:
- H_mov: Movement entropy (dispersal-based)
- C(L): Connectivity constraint (genetic viability)
- F(L): Forest topology penalty (dendritic structure)
- D(L): Disturbance response time penalty
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
from .landscape import Landscape


def movement_entropy(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    corridor_widths: Dict[Tuple[int, int], float] = None,
    species_guild = None,
    dispersal_scale: float = 1000.0,
    alpha: float = 2.0
) -> float:
    """
    Calculate movement entropy based on dispersal kernel with width-dependent success.

    Movement entropy quantifies uncertainty in organism movement across
    the landscape. Lower entropy indicates more predictable movement patterns
    (i.e., fewer, more constrained pathways).

    Uses width-dependent dispersal kernel:
    P(i→j) ∝ exp(-α · d_ij / s) · φ(w_ij) · q_j

    Where:
    - d_ij is corridor distance
    - s is dispersal scale
    - φ(w) = 1 − exp(−γ(w − w_min)) is width-dependent movement success
    - q_j is patch quality

    Args:
        landscape: Landscape object
        edges: List of corridor edges
        corridor_widths: Dictionary mapping (i,j) to corridor width (meters)
                        If None, assumes infinite width (φ=1)
        species_guild: SpeciesGuild object for width sensitivity parameters
                      If None, uses default parameters
        dispersal_scale: Characteristic dispersal distance (meters)
        alpha: Dispersal cost parameter (overridden if species_guild provided)

    Returns:
        Movement entropy H_mov (nats)

    Invariants:
    - H_mov >= 0
    - H_mov decreases with fewer/shorter corridors
    - H_mov decreases with narrower corridors (lower φ(w))
    - H_mov = 0 only if movement is deterministic (single path)

    Reference:
        Shannon entropy: H = -Σ p_i log(p_i)
        Hart (2024), Equations 8-10
    """
    assert dispersal_scale > 0, "Dispersal scale must be positive"
    assert alpha > 0, "Alpha must be positive"

    # Use species guild parameters if provided
    if species_guild is not None:
        alpha = species_guild.alpha

    n = landscape.n_patches

    # Build corridor network graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Map patch IDs to indices
    id_to_idx = {patch.id: i for i, patch in enumerate(landscape.patches)}

    for (i, j) in edges:
        idx_i = id_to_idx[i]
        idx_j = id_to_idx[j]
        distance = landscape.graph[i][j]['distance']

        # Get corridor width (default to infinite if not specified)
        edge_key = tuple(sorted([i, j]))
        if corridor_widths is not None and edge_key in corridor_widths:
            width = corridor_widths[edge_key]
        else:
            width = float('inf')

        G.add_edge(idx_i, idx_j, distance=distance, width=width)

    # Calculate movement probabilities for each patch
    H_total = 0.0

    for i in range(n):
        # Get neighbors in corridor network
        neighbors = list(G.neighbors(i))

        if len(neighbors) == 0:
            # Isolated patch - contributes zero entropy (deterministic: stay)
            continue

        # Calculate transition probabilities
        probs = []
        for j in neighbors:
            d_ij = G[i][j]['distance']
            w_ij = G[i][j]['width']
            q_j = landscape.patches[j].quality

            # Distance-based dispersal probability
            distance_effect = np.exp(-alpha * d_ij / dispersal_scale)

            # Width-dependent movement success φ(w)
            if species_guild is not None and np.isfinite(w_ij):
                width_effect = species_guild.movement_success(w_ij)
            elif np.isfinite(w_ij) and w_ij > 0:
                # Default width effect if no guild specified
                # Using moderate sensitivity: γ ≈ 0.05, w_min ≈ 50
                width_effect = 1.0 - np.exp(-0.05 * max(0, w_ij - 50))
            else:
                # Infinite width or no width specified
                width_effect = 1.0

            # Combined movement probability
            p_ij = distance_effect * width_effect * q_j
            probs.append(p_ij)

        # Normalize
        probs = np.array(probs)
        if probs.sum() > 0:
            probs = probs / probs.sum()

            # Shannon entropy for this patch's movements
            H_i = -np.sum(probs * np.log(probs + 1e-10))  # Add epsilon for numerical stability
            H_total += H_i

    # Average over patches
    H_mov = H_total / n

    # Verify invariant
    assert H_mov >= 0, f"Movement entropy must be non-negative, got {H_mov}"

    return H_mov


def stationary_distribution(landscape: Landscape) -> np.ndarray:
    """
    Calculate quality-weighted stationary distribution.

    For heterogeneous landscapes, the stationary distribution weights
    patches by their area and habitat quality, representing long-term
    occupancy probabilities.

    πᵢ = (Aᵢ · qᵢ) / (∑ₖ Aₖ · qₖ)

    Args:
        landscape: Landscape object

    Returns:
        Array of stationary probabilities [π₁, π₂, ..., πₙ]

    Invariants:
    - All πᵢ >= 0
    - ∑ᵢ πᵢ = 1
    - Larger/higher-quality patches have higher πᵢ

    Reference:
        Hart (2024), Equation 12
    """
    areas = np.array([p.area for p in landscape.patches])
    qualities = np.array([p.quality for p in landscape.patches])

    # Quality-weighted area
    weighted = areas * qualities

    # Normalize to probability distribution
    pi = weighted / weighted.sum()

    # Verify invariants
    assert np.all(pi >= 0), "Stationary probabilities must be non-negative"
    assert np.isclose(pi.sum(), 1.0), f"Stationary distribution must sum to 1, got {pi.sum()}"

    return pi


def entropy_rate(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    corridor_widths: Dict[Tuple[int, int], float] = None,
    species_guild = None,
    dispersal_scale: float = 1000.0,
    alpha: float = 2.0
) -> float:
    """
    Calculate entropy rate with stationary distribution weighting.

    Entropy rate accounts for heterogeneous patch importance by weighting
    each patch's movement entropy by its long-term occupancy probability.

    H_rate(A,w,π) = −∑ᵢ πᵢ ∑ⱼ pᵢⱼ(A,w) log[pᵢⱼ(A,w)]

    Where πᵢ is the stationary distribution (quality-weighted area).

    Args:
        landscape: Landscape object
        edges: List of corridor edges
        corridor_widths: Dictionary mapping (i,j) to corridor width (meters)
        species_guild: SpeciesGuild object for width sensitivity
        dispersal_scale: Characteristic dispersal distance (meters)
        alpha: Dispersal cost parameter

    Returns:
        Entropy rate H_rate (nats)

    Invariants:
    - H_rate >= 0
    - H_rate <= H_mov (weighting by π reduces or maintains entropy)
    - H_rate emphasizes important patches

    Reference:
        Hart (2024), Equations 11-13
    """
    assert dispersal_scale > 0, "Dispersal scale must be positive"
    assert alpha > 0, "Alpha must be positive"

    # Use species guild parameters if provided
    if species_guild is not None:
        alpha = species_guild.alpha

    # Calculate stationary distribution
    pi = stationary_distribution(landscape)

    n = landscape.n_patches

    # Build corridor network graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Map patch IDs to indices
    id_to_idx = {patch.id: i for i, patch in enumerate(landscape.patches)}

    for (i, j) in edges:
        idx_i = id_to_idx[i]
        idx_j = id_to_idx[j]
        distance = landscape.graph[i][j]['distance']

        # Get corridor width
        edge_key = tuple(sorted([i, j]))
        if corridor_widths is not None and edge_key in corridor_widths:
            width = corridor_widths[edge_key]
        else:
            width = float('inf')

        G.add_edge(idx_i, idx_j, distance=distance, width=width)

    # Calculate weighted entropy rate
    H_rate_total = 0.0

    for i in range(n):
        # Get neighbors in corridor network
        neighbors = list(G.neighbors(i))

        if len(neighbors) == 0:
            # Isolated patch - contributes zero
            continue

        # Calculate transition probabilities
        probs = []
        for j in neighbors:
            d_ij = G[i][j]['distance']
            w_ij = G[i][j]['width']
            q_j = landscape.patches[j].quality

            # Distance-based dispersal
            distance_effect = np.exp(-alpha * d_ij / dispersal_scale)

            # Width-dependent success
            if species_guild is not None and np.isfinite(w_ij):
                width_effect = species_guild.movement_success(w_ij)
            elif np.isfinite(w_ij) and w_ij > 0:
                width_effect = 1.0 - np.exp(-0.05 * max(0, w_ij - 50))
            else:
                width_effect = 1.0

            # Combined probability
            p_ij = distance_effect * width_effect * q_j
            probs.append(p_ij)

        # Normalize
        probs = np.array(probs)
        if probs.sum() > 0:
            probs = probs / probs.sum()

            # Shannon entropy for this patch
            H_i = -np.sum(probs * np.log(probs + 1e-10))

            # Weight by stationary probability
            H_rate_total += pi[i] * H_i

    # Verify invariant
    assert H_rate_total >= 0, f"Entropy rate must be non-negative, got {H_rate_total}"

    return H_rate_total


def effective_population_size(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    corridor_widths: Dict[Tuple[int, int], float] = None,
    species_guild = None,
    dispersal_scale: float = 1000.0,
    alpha: float = 2.0,
    carrying_capacity_per_ha: float = 10.0
) -> float:
    """
    Calculate effective population size using island model.

    Implements Wright's island model for effective population size:

    Nₑ(A,w) = [∑ᵢ nᵢ]² / [∑ᵢ nᵢ² + ∑ᵢ ∑ⱼ≠ᵢ 2nᵢnⱼFᵢⱼ(A,w)]

    Where:
    - nᵢ: Local population size in patch i
    - Fᵢⱼ(A,w) = mᵢⱼ(A,w) / [2 − mᵢⱼ(A,w)]: Co-ancestry coefficient
    - mᵢⱼ(A,w): Width-dependent migration rate

    Args:
        landscape: Landscape object
        edges: List of corridor edges
        corridor_widths: Dictionary mapping (i,j) to corridor width (meters)
        species_guild: SpeciesGuild object for movement parameters
        dispersal_scale: Characteristic dispersal distance (meters)
        alpha: Dispersal cost parameter
        carrying_capacity_per_ha: Population density (individuals/hectare)

    Returns:
        Effective population size Nₑ

    Invariants:
    - Nₑ > 0
    - Nₑ ≤ total population size
    - Nₑ increases with connectivity

    Reference:
        Hart (2024), Equations 14-17
        Wright (1943), Island model genetics
    """
    assert dispersal_scale > 0, "Dispersal scale must be positive"
    assert alpha > 0, "Alpha must be positive"
    assert carrying_capacity_per_ha > 0, "Carrying capacity must be positive"

    # Use species guild parameters if provided
    if species_guild is not None:
        alpha = species_guild.alpha

    n = landscape.n_patches

    # Calculate local population sizes
    # nᵢ = Aᵢ · K · qᵢ where K is carrying capacity
    populations = np.array([
        p.area * carrying_capacity_per_ha * p.quality
        for p in landscape.patches
    ])

    # Build corridor network
    G = nx.Graph()
    G.add_nodes_from(range(n))
    id_to_idx = {patch.id: i for i, patch in enumerate(landscape.patches)}

    for (i, j) in edges:
        idx_i = id_to_idx[i]
        idx_j = id_to_idx[j]
        distance = landscape.graph[i][j]['distance']

        # Get corridor width
        edge_key = tuple(sorted([i, j]))
        if corridor_widths is not None and edge_key in corridor_widths:
            width = corridor_widths[edge_key]
        else:
            width = float('inf')

        G.add_edge(idx_i, idx_j, distance=distance, width=width)

    # Calculate migration rates mᵢⱼ(A,w)
    # mᵢⱼ(A,w) = σ · pᵢⱼ(A,w) / (1 + σ · pᵢⱼ(A,w))
    # where σ is dispersal propensity (use 0.1 as default)
    sigma = 0.1

    migration_matrix = np.zeros((n, n))

    for i in range(n):
        neighbors = list(G.neighbors(i))

        if len(neighbors) == 0:
            continue

        # Calculate movement probabilities
        probs = {}
        for j in neighbors:
            d_ij = G[i][j]['distance']
            w_ij = G[i][j]['width']

            # Distance effect
            distance_effect = np.exp(-alpha * d_ij / dispersal_scale)

            # Width effect
            if species_guild is not None and np.isfinite(w_ij):
                width_effect = species_guild.movement_success(w_ij)
            elif np.isfinite(w_ij) and w_ij > 0:
                width_effect = 1.0 - np.exp(-0.05 * max(0, w_ij - 50))
            else:
                width_effect = 1.0

            # Movement probability
            p_ij = distance_effect * width_effect
            probs[j] = p_ij

        # Normalize
        total_prob = sum(probs.values())
        if total_prob > 0:
            for j, p_ij in probs.items():
                p_ij_norm = p_ij / total_prob
                # Migration rate with dispersal propensity
                m_ij = sigma * p_ij_norm / (1 + sigma * p_ij_norm)
                migration_matrix[i, j] = m_ij

    # Calculate co-ancestry coefficients Fᵢⱼ
    # Fᵢⱼ = mᵢⱼ / (2 − mᵢⱼ)
    F_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and migration_matrix[i, j] > 0:
                m_ij = migration_matrix[i, j]
                F_matrix[i, j] = m_ij / (2 - m_ij) if m_ij < 2 else 1.0

    # Calculate effective population size
    total_pop = populations.sum()

    if total_pop == 0:
        return 0.0

    # Numerator: (∑ᵢ nᵢ)²
    numerator = total_pop ** 2

    # Denominator: ∑ᵢ nᵢ² + ∑ᵢ ∑ⱼ≠ᵢ 2nᵢnⱼFᵢⱼ
    within_patch_term = np.sum(populations ** 2)

    between_patch_term = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                between_patch_term += 2 * populations[i] * populations[j] * F_matrix[i, j]

    denominator = within_patch_term + between_patch_term

    if denominator == 0:
        return total_pop

    N_e = numerator / denominator

    # Verify invariants
    assert N_e > 0, f"Effective population size must be positive, got {N_e}"
    assert N_e <= total_pop + 1, f"N_e cannot exceed total population ({total_pop}), got {N_e}"

    return N_e


def connectivity_constraint(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    corridor_widths: Dict[Tuple[int, int], float] = None,
    species_guild = None,
    target_ne: float = 500.0,
    penalty_weight: float = 1.0,
    dispersal_scale: float = 1000.0,
    alpha: float = 2.0
) -> float:
    """
    Connectivity constraint ensuring genetic viability via effective population size.

    Penalizes networks that fail to maintain minimum effective population
    size (N_e) for genetic viability. Now uses full island model calculation.

    Args:
        landscape: Landscape object
        edges: List of corridor edges
        corridor_widths: Dictionary mapping (i,j) to corridor width (meters)
        species_guild: SpeciesGuild object (provides N_e threshold if None uses target_ne)
        target_ne: Target effective population size (used if species_guild is None)
        penalty_weight: Weight for constraint violation
        dispersal_scale: Characteristic dispersal distance (meters)
        alpha: Dispersal cost parameter

    Returns:
        Connectivity penalty C(L)

    Invariants:
    - C(L) >= 0
    - C(L) = 0 when N_e >= target
    - C(L) increases as N_e falls below target
    """
    assert target_ne > 0, "Target N_e must be positive"
    assert penalty_weight >= 0, "Penalty weight must be non-negative"

    # Use species guild threshold if provided
    if species_guild is not None:
        target_ne = species_guild.Ne_threshold

    # Calculate effective population size
    N_e = effective_population_size(
        landscape, edges, corridor_widths, species_guild,
        dispersal_scale, alpha
    )

    # Penalty based on deficit from target
    if N_e >= target_ne:
        # Meets genetic viability threshold
        penalty = 0.0
    else:
        # Penalty proportional to deficit
        deficit_ratio = (target_ne - N_e) / target_ne
        penalty = penalty_weight * deficit_ratio

    # Verify invariant
    assert penalty >= 0, f"Connectivity penalty must be non-negative, got {penalty}"

    return penalty


def forest_topology_penalty(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    penalty_weight: float = 1.0
) -> float:
    """
    Forest topology penalty favoring dendritic (tree) structures.
    
    Penalizes networks that deviate from tree topology by including
    cycles/loops. Dendritic networks (n-1 edges for n nodes) minimize
    this penalty.
    
    Args:
        landscape: Landscape object
        edges: List of corridor edges
        penalty_weight: Weight for topology penalty
        
    Returns:
        Forest penalty F(L)
        
    Invariants:
    - F(L) >= 0
    - F(L) = 0 for trees (acyclic, connected)
    - F(L) increases with number of cycles
    """
    assert penalty_weight >= 0, "Penalty weight must be non-negative"
    
    n = landscape.n_patches
    n_edges = len(edges)
    
    # Build network
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    id_to_idx = {patch.id: i for i, patch in enumerate(landscape.patches)}
    
    for (i, j) in edges:
        idx_i = id_to_idx[i]
        idx_j = id_to_idx[j]
        G.add_edge(idx_i, idx_j)
    
    # Tree has n-1 edges for n nodes
    # Excess edges create cycles
    if nx.is_connected(G):
        n_cycles = n_edges - (n - 1)
    else:
        # Penalize disconnection
        n_components = nx.number_connected_components(G)
        # Disconnected: missing edges plus any extra
        min_edges_needed = n - n_components
        n_cycles = abs(n_edges - min_edges_needed)
    
    F = penalty_weight * max(0, n_cycles)
    
    # Verify invariant
    assert F >= 0, f"Forest penalty must be non-negative, got {F}"
    
    return F


def disturbance_response_penalty(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    penalty_weight: float = 1.0
) -> float:
    """
    Disturbance response time penalty.
    
    Penalizes network configurations with slow disturbance response
    (long paths to reach affected areas). Shorter maximum path lengths
    enable faster response to fires, diseases, logging, etc.
    
    Args:
        landscape: Landscape object
        edges: List of corridor edges
        penalty_weight: Weight for response time penalty
        
    Returns:
        Disturbance response penalty D(L)
        
    Invariants:
    - D(L) >= 0
    - D(L) decreases with shorter maximum paths
    """
    assert penalty_weight >= 0, "Penalty weight must be non-negative"
    
    n = landscape.n_patches
    
    if len(edges) == 0:
        # No connections - maximum penalty
        return penalty_weight * n
    
    # Build network
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    id_to_idx = {patch.id: i for i, patch in enumerate(landscape.patches)}
    
    for (i, j) in edges:
        idx_i = id_to_idx[i]
        idx_j = id_to_idx[j]
        G.add_edge(idx_i, idx_j)
    
    # Calculate diameter (longest shortest path)
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        # Normalize by network size
        D = penalty_weight * (diameter / n)
    else:
        # Disconnected - use maximum possible diameter
        D = penalty_weight * 1.0
    
    # Verify invariant
    assert D >= 0, f"Disturbance penalty must be non-negative, got {D}"
    
    return D


def landscape_allocation_constraint(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    corridor_widths: Dict[Tuple[int, int], float],
    allocation_budget: float = 0.25,
    penalty_weight: float = 10.0
) -> float:
    """
    Landscape allocation constraint enforcing corridor area budget.

    Penalizes networks that exceed the available landscape allocation
    for corridor establishment (typically 20-30% of total area).

    Total corridor area constraint:
    ∑ᵢⱼ dᵢⱼ · wᵢⱼ ≤ β · ∑ᵢ Aᵢ

    Where:
    - dᵢⱼ: Corridor length (meters)
    - wᵢⱼ: Corridor width (meters)
    - β: Landscape allocation fraction [0.20, 0.30]

    Args:
        landscape: Landscape object
        edges: List of corridor edges
        corridor_widths: Dictionary mapping (i,j) to corridor width (meters)
        allocation_budget: Fraction of total landscape area for corridors (β)
        penalty_weight: Weight for budget violation penalty

    Returns:
        Allocation penalty A(L)

    Invariants:
    - A(L) >= 0
    - A(L) = 0 when total corridor area <= budget
    - A(L) increases with budget excess

    Reference:
        Hart (2024), Section 3.4, Design Parameters
    """
    assert 0 < allocation_budget <= 1.0, "Allocation budget must be in (0, 1]"
    assert penalty_weight >= 0, "Penalty weight must be non-negative"

    # Calculate total patch area (convert hectares to m²)
    total_patch_area = sum(p.area * 10000 for p in landscape.patches)  # ha to m²

    # Calculate total corridor area
    total_corridor_area = 0.0
    for (i, j) in edges:
        distance = landscape.graph[i][j]['distance']  # meters
        edge_key = tuple(sorted([i, j]))

        if edge_key in corridor_widths:
            width = corridor_widths[edge_key]
            corridor_area = distance * width  # m²
            total_corridor_area += corridor_area

    # Budget threshold
    budget_threshold = allocation_budget * total_patch_area

    # Penalty if over budget
    if total_corridor_area <= budget_threshold:
        penalty = 0.0
    else:
        # Penalty proportional to excess
        excess_ratio = (total_corridor_area - budget_threshold) / budget_threshold
        penalty = penalty_weight * excess_ratio

    # Verify invariant
    assert penalty >= 0, f"Allocation penalty must be non-negative, got {penalty}"

    return penalty


def calculate_entropy(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    corridor_widths: Dict[Tuple[int, int], float] = None,
    species_guild = None,
    lambda1: float = 1.0,  # Connectivity weight
    lambda2: float = 1.0,  # Forest topology weight
    lambda3: float = 1.0,  # Disturbance response weight
    lambda4: float = 0.0,  # Allocation constraint weight (0 = no constraint)
    allocation_budget: float = 0.25,  # 25% default budget
    dispersal_scale: float = 1000.0,
    alpha: float = 2.0,
    use_entropy_rate: bool = False  # Use H_rate instead of H_mov if True
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate total landscape entropy with width-dependent effects.

    Implements extended HDFM objective function:
    H(L) = H_mov(A,w) + λ₁·C(L) + λ₂·F(L) + λ₃·D(L) + λ₄·A(L)

    Or with entropy rate:
    H(L) = H_rate(A,w,π) + λ₁·C(L) + λ₂·F(L) + λ₃·D(L) + λ₄·A(L)

    Args:
        landscape: Landscape object
        edges: List of corridor edges
        corridor_widths: Dictionary mapping (i,j) to width (meters)
        species_guild: SpeciesGuild for species-specific parameters
        lambda1: Weight for connectivity constraint (genetic viability)
        lambda2: Weight for forest topology penalty (dendritic structure)
        lambda3: Weight for disturbance response penalty (path length)
        lambda4: Weight for allocation constraint (budget enforcement)
        allocation_budget: Landscape allocation fraction (β ∈ [0.2, 0.3])
        dispersal_scale: Characteristic dispersal distance (meters)
        alpha: Dispersal cost parameter
        use_entropy_rate: If True, use H_rate instead of H_mov

    Returns:
        (total_entropy, component_dict)

        where component_dict contains:
        - 'H_mov' or 'H_rate': Movement entropy or entropy rate
        - 'C': Connectivity penalty (genetic viability)
        - 'F': Forest topology penalty
        - 'D': Disturbance response penalty
        - 'A': Allocation penalty (if λ₄ > 0)
        - 'N_e': Effective population size (diagnostic)
        - 'H_total': Total entropy

    Invariants:
    - H_total >= 0
    - All components >= 0
    - Width-dependent: narrower corridors → higher entropy
    - Dendritic networks minimize H_total
    """
    # Calculate movement entropy (basic or rate-weighted)
    if use_entropy_rate:
        H_mov = entropy_rate(
            landscape, edges, corridor_widths, species_guild,
            dispersal_scale, alpha
        )
        entropy_type = 'H_rate'
    else:
        H_mov = movement_entropy(
            landscape, edges, corridor_widths, species_guild,
            dispersal_scale, alpha
        )
        entropy_type = 'H_mov'

    # Calculate connectivity constraint (with N_e)
    C = connectivity_constraint(
        landscape, edges, corridor_widths, species_guild,
        dispersal_scale=dispersal_scale, alpha=alpha
    )

    # Calculate topological penalties
    F = forest_topology_penalty(landscape, edges)
    D = disturbance_response_penalty(landscape, edges)

    # Calculate allocation constraint (if widths specified and λ₄ > 0)
    if corridor_widths is not None and lambda4 > 0:
        A = landscape_allocation_constraint(
            landscape, edges, corridor_widths, allocation_budget
        )
    else:
        A = 0.0

    # Calculate diagnostic effective population size
    N_e = effective_population_size(
        landscape, edges, corridor_widths, species_guild,
        dispersal_scale, alpha
    )

    # Total entropy
    H_total = H_mov + lambda1 * C + lambda2 * F + lambda3 * D + lambda4 * A

    # Verify invariants
    assert H_total >= 0, f"Total entropy must be non-negative, got {H_total}"
    assert H_mov >= 0 and C >= 0 and F >= 0 and D >= 0 and A >= 0, \
        "All components must be non-negative"

    components = {
        entropy_type: H_mov,
        'C': C,
        'F': F,
        'D': D,
        'A': A,
        'N_e': N_e,
        'H_total': H_total
    }

    return H_total, components


def entropy_gradient(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    **kwargs
) -> Dict[Tuple[int, int], float]:
    """
    Calculate entropy gradient for each potential edge.
    
    Estimates ∂H/∂e_ij for each edge, indicating how adding/removing
    that edge would change total entropy.
    
    Args:
        landscape: Landscape object
        edges: Current corridor edges
        **kwargs: Additional parameters for calculate_entropy
        
    Returns:
        Dictionary mapping (i,j) edge to entropy gradient
        
    Usage:
        Used for greedy optimization and sensitivity analysis
    """
    current_entropy, _ = calculate_entropy(landscape, edges, **kwargs)
    
    gradients = {}
    
    # Check adding each missing edge
    current_edge_set = set(edges)
    
    for i, patch_i in enumerate(landscape.patches):
        for patch_j in landscape.patches[i+1:]:
            edge = (patch_i.id, patch_j.id)
            
            if edge not in current_edge_set:
                # Test adding this edge
                new_edges = edges + [edge]
                new_entropy, _ = calculate_entropy(landscape, new_edges, **kwargs)
                gradients[edge] = new_entropy - current_entropy
    
    return gradients
