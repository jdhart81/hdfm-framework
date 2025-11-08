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
    dispersal_scale: float = 1000.0,
    alpha: float = 2.0
) -> float:
    """
    Calculate movement entropy based on dispersal kernel.
    
    Movement entropy quantifies uncertainty in organism movement across
    the landscape. Lower entropy indicates more predictable movement patterns
    (i.e., fewer, more constrained pathways).
    
    Uses cost-based dispersal kernel:
    P(i→j) ∝ exp(-α · d_ij / s) · q_j
    
    Where d_ij is corridor distance, s is dispersal scale, q_j is patch quality.
    
    Args:
        landscape: Landscape object
        edges: List of corridor edges
        dispersal_scale: Characteristic dispersal distance (meters)
        alpha: Dispersal cost parameter
        
    Returns:
        Movement entropy H_mov (nats)
        
    Invariants:
    - H_mov >= 0
    - H_mov decreases with fewer/shorter corridors
    - H_mov = 0 only if movement is deterministic (single path)
    
    Reference:
        Shannon entropy: H = -Σ p_i log(p_i)
    """
    assert dispersal_scale > 0, "Dispersal scale must be positive"
    assert alpha > 0, "Alpha must be positive"
    
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
        G.add_edge(idx_i, idx_j, distance=distance)
    
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
            q_j = landscape.patches[j].quality
            
            # Dispersal probability
            p_ij = np.exp(-alpha * d_ij / dispersal_scale) * q_j
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


def connectivity_constraint(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    target_ne: float = 500.0,
    penalty_weight: float = 1.0
) -> float:
    """
    Connectivity constraint ensuring genetic viability.
    
    Penalizes networks that fail to maintain minimum effective population
    size (N_e) across the landscape. Uses graph connectivity metrics as
    proxy for genetic connectivity.
    
    Args:
        landscape: Landscape object
        edges: List of corridor edges
        target_ne: Target effective population size
        penalty_weight: Weight for constraint violation
        
    Returns:
        Connectivity penalty C(L)
        
    Invariants:
    - C(L) >= 0
    - C(L) = 0 when connectivity meets target
    - C(L) increases with disconnection severity
    """
    assert target_ne > 0, "Target N_e must be positive"
    assert penalty_weight >= 0, "Penalty weight must be non-negative"
    
    n = landscape.n_patches
    
    # Build network
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    id_to_idx = {patch.id: i for i, patch in enumerate(landscape.patches)}
    
    for (i, j) in edges:
        idx_i = id_to_idx[i]
        idx_j = id_to_idx[j]
        G.add_edge(idx_i, idx_j)
    
    # Calculate connectivity metrics
    n_components = nx.number_connected_components(G)
    
    # Penalty for disconnection
    if n_components > 1:
        # Severe penalty for disconnected network
        component_penalty = (n_components - 1) * 10.0
    else:
        component_penalty = 0.0
    
    # Calculate average shortest path length (if connected)
    if n_components == 1:
        avg_path_length = nx.average_shortest_path_length(G)
        # Penalty increases with path length
        path_penalty = max(0, (avg_path_length - 2.0) / n)
    else:
        # Maximum penalty for disconnected
        path_penalty = 1.0
    
    C = penalty_weight * (component_penalty + path_penalty)
    
    # Verify invariant
    assert C >= 0, f"Connectivity penalty must be non-negative, got {C}"
    
    return C


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


def calculate_entropy(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    lambda1: float = 1.0,  # Connectivity weight
    lambda2: float = 1.0,  # Forest topology weight
    lambda3: float = 1.0,  # Disturbance response weight
    dispersal_scale: float = 1000.0,
    alpha: float = 2.0
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate total landscape entropy.
    
    Implements core HDFM objective function:
    H(L) = H_mov + λ₁·C(L) + λ₂·F(L) + λ₃·D(L)
    
    Args:
        landscape: Landscape object
        edges: List of corridor edges
        lambda1: Weight for connectivity constraint
        lambda2: Weight for forest topology penalty
        lambda3: Weight for disturbance response penalty
        dispersal_scale: Characteristic dispersal distance (meters)
        alpha: Dispersal cost parameter
        
    Returns:
        (total_entropy, component_dict)
        
        where component_dict contains:
        - 'H_mov': Movement entropy
        - 'C': Connectivity penalty
        - 'F': Forest topology penalty
        - 'D': Disturbance response penalty
        - 'H_total': Total entropy
        
    Invariants:
    - H_total >= 0
    - All components >= 0
    - Dendritic networks minimize H_total
    """
    # Calculate components
    H_mov = movement_entropy(landscape, edges, dispersal_scale, alpha)
    C = connectivity_constraint(landscape, edges)
    F = forest_topology_penalty(landscape, edges)
    D = disturbance_response_penalty(landscape, edges)
    
    # Total entropy
    H_total = H_mov + lambda1 * C + lambda2 * F + lambda3 * D
    
    # Verify invariants
    assert H_total >= 0, f"Total entropy must be non-negative, got {H_total}"
    assert H_mov >= 0 and C >= 0 and F >= 0 and D >= 0, "All components must be non-negative"
    
    components = {
        'H_mov': H_mov,
        'C': C,
        'F': F,
        'D': D,
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
