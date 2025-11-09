"""
Robustness analysis with looped topologies and redundancy scoring.

Implements network robustness metrics including:
- 2-edge-connectivity (ρ₂) for structural redundancy
- Strategic loop addition to MST networks
- Catastrophic failure probability P_fail(k)
- Redundancy scoring for corridors
- Pareto frontier analysis (entropy vs. robustness tradeoffs)

Reference:
    Hart, J. (2024). Section 3.4: Robustness analysis with loop budgets
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from .landscape import Landscape
from .entropy import calculate_entropy
from .network import build_dendritic_network


@dataclass
class RobustnessMetrics:
    """
    Container for network robustness metrics.

    Attributes:
        two_edge_connectivity: ρ₂ - fraction of node pairs with 2 edge-disjoint paths
        redundancy_score: Overall redundancy score [0, 1]
        failure_probability: P_fail(k) - probability of catastrophic failure
        n_loops: Number of cycles in network
        edge_redundancy: Dict mapping edges to redundancy importance scores
    """
    two_edge_connectivity: float
    redundancy_score: float
    failure_probability: float
    n_loops: int
    edge_redundancy: Dict[Tuple[int, int], float]


def calculate_two_edge_connectivity(
    landscape: Landscape,
    edges: List[Tuple[int, int]]
) -> Tuple[float, Dict[str, any]]:
    """
    Calculate 2-edge-connectivity ρ₂ for network robustness.

    ρ₂ measures the fraction of node pairs that remain connected after
    removing any single edge (have at least 2 edge-disjoint paths).

    ρ₂ = (# node pairs with 2 edge-disjoint paths) / (n choose 2)

    Properties:
    - ρ₂ = 0 for trees (MST has no redundancy)
    - ρ₂ > 0 requires loops/cycles
    - ρ₂ = 1 for 2-edge-connected graphs (every edge removal preserves connectivity)

    Args:
        landscape: Landscape object
        edges: List of corridor edges

    Returns:
        (rho_2, components_dict) where components contains:
        - 'rho_2': 2-edge-connectivity metric [0, 1]
        - 'two_connected_pairs': Number of 2-edge-connected pairs
        - 'total_pairs': Total number of node pairs
        - 'vulnerable_edges': Edges whose removal disconnects the network

    Reference:
        Hart (2024), Equation 8: ρ₂(A) robustness metric
    """
    n = landscape.n_patches

    if n < 2:
        return 0.0, {'rho_2': 0.0, 'two_connected_pairs': 0, 'total_pairs': 0}

    # Build network
    G = nx.Graph()
    G.add_nodes_from(range(n))

    id_to_idx = {patch.id: i for i, patch in enumerate(landscape.patches)}

    for (i, j) in edges:
        idx_i = id_to_idx[i]
        idx_j = id_to_idx[j]
        G.add_edge(idx_i, idx_j)

    # Count pairs with 2 edge-disjoint paths
    two_connected_pairs = 0
    total_pairs = n * (n - 1) // 2

    # For each pair of nodes, check if there are 2 edge-disjoint paths
    for i in range(n):
        for j in range(i + 1, n):
            if nx.has_path(G, i, j):
                # Check edge connectivity between i and j
                edge_connectivity = nx.edge_connectivity(G, i, j)
                if edge_connectivity >= 2:
                    two_connected_pairs += 1

    rho_2 = two_connected_pairs / total_pairs if total_pairs > 0 else 0.0

    # Find vulnerable edges (bridges)
    vulnerable_edges = []
    for edge in G.edges():
        G_copy = G.copy()
        G_copy.remove_edge(*edge)
        if not nx.is_connected(G_copy):
            # Convert back to patch IDs
            patch_i = landscape.patches[edge[0]].id
            patch_j = landscape.patches[edge[1]].id
            vulnerable_edges.append((patch_i, patch_j))

    components = {
        'rho_2': rho_2,
        'two_connected_pairs': two_connected_pairs,
        'total_pairs': total_pairs,
        'vulnerable_edges': vulnerable_edges,
        'n_bridges': len(vulnerable_edges)
    }

    assert 0 <= rho_2 <= 1, f"ρ₂ must be in [0,1], got {rho_2}"

    return rho_2, components


def calculate_edge_redundancy_scores(
    landscape: Landscape,
    edges: List[Tuple[int, int]]
) -> Dict[Tuple[int, int], float]:
    """
    Calculate redundancy importance score for each edge.

    Redundancy score measures how critical an edge is for network connectivity:
    - Score = 1.0: Bridge edge (removal disconnects network)
    - Score = 0.0: Highly redundant edge (many alternative paths)
    - 0 < Score < 1: Partial redundancy

    Score is based on:
    1. Betweenness centrality (how many shortest paths use this edge)
    2. Bridge status (is it a cut edge?)
    3. Impact on average path length if removed

    Args:
        landscape: Landscape object
        edges: List of corridor edges

    Returns:
        Dict mapping (i,j) edge to redundancy score [0, 1]

    Properties:
    - Higher score = more critical edge
    - Bridge edges have score = 1.0
    - Redundant edges have score near 0.0
    """
    n = landscape.n_patches

    # Build network
    G = nx.Graph()
    G.add_nodes_from(range(n))

    id_to_idx = {patch.id: i for i, patch in enumerate(landscape.patches)}

    for (i, j) in edges:
        idx_i = id_to_idx[i]
        idx_j = id_to_idx[j]
        G.add_edge(idx_i, idx_j)

    if not nx.is_connected(G):
        # For disconnected graphs, all edges are critical
        return {edge: 1.0 for edge in edges}

    # Calculate edge betweenness centrality
    edge_betweenness = nx.edge_betweenness_centrality(G)

    # Normalize betweenness to [0, 1]
    max_betweenness = max(edge_betweenness.values()) if edge_betweenness else 1.0
    if max_betweenness > 0:
        edge_betweenness = {e: b / max_betweenness for e, b in edge_betweenness.items()}

    redundancy_scores = {}

    # Calculate baseline average path length
    baseline_apl = nx.average_shortest_path_length(G)

    for (patch_i_id, patch_j_id) in edges:
        idx_i = id_to_idx[patch_i_id]
        idx_j = id_to_idx[patch_j_id]

        # Test removing this edge
        G_test = G.copy()
        G_test.remove_edge(idx_i, idx_j)

        if not nx.is_connected(G_test):
            # Bridge edge - maximum redundancy score
            score = 1.0
        else:
            # Calculate impact on average path length
            new_apl = nx.average_shortest_path_length(G_test)
            apl_increase = (new_apl - baseline_apl) / baseline_apl

            # Get betweenness contribution
            edge_tuple = (idx_i, idx_j) if (idx_i, idx_j) in edge_betweenness else (idx_j, idx_i)
            betweenness = edge_betweenness.get(edge_tuple, 0.0)

            # Combine metrics (weighted average)
            score = 0.5 * betweenness + 0.5 * apl_increase

            # Ensure in [0, 1]
            score = max(0.0, min(1.0, score))

        redundancy_scores[(patch_i_id, patch_j_id)] = score

    return redundancy_scores


def calculate_failure_probability(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    k: int = 1,
    edge_failure_prob: float = 0.1
) -> Tuple[float, Dict[str, any]]:
    """
    Calculate catastrophic failure probability P_fail(k).

    P_fail(k) is the probability that removing k randomly failed edges
    disconnects the network into multiple components.

    Uses Monte Carlo simulation to estimate:
    P_fail(k) ≈ (# simulations resulting in disconnection) / (# total simulations)

    Args:
        landscape: Landscape object
        edges: List of corridor edges
        k: Number of edges to fail simultaneously
        edge_failure_prob: Probability of individual edge failure

    Returns:
        (P_fail, components_dict) where components contains:
        - 'P_fail': Failure probability [0, 1]
        - 'n_simulations': Number of Monte Carlo simulations
        - 'n_failures': Number of simulations resulting in disconnection
        - 'mean_components': Average number of components after failure

    Properties:
    - P_fail(k) increases with k
    - P_fail(k) = 0 for highly redundant networks
    - P_fail(k) = 1 for networks with k or fewer bridges

    Reference:
        Hart (2024), Table 3: Robustness metrics
    """
    assert 0 <= edge_failure_prob <= 1, "Failure probability must be in [0,1]"
    assert k >= 1, "k must be at least 1"
    assert k <= len(edges), "k cannot exceed number of edges"

    n = landscape.n_patches

    # Build network
    G = nx.Graph()
    G.add_nodes_from(range(n))

    id_to_idx = {patch.id: i for i, patch in enumerate(landscape.patches)}

    for (i, j) in edges:
        idx_i = id_to_idx[i]
        idx_j = id_to_idx[j]
        G.add_edge(idx_i, idx_j)

    # Monte Carlo simulation
    n_simulations = 1000
    n_failures = 0
    component_counts = []

    for _ in range(n_simulations):
        # Randomly select k edges to fail
        failed_edges = np.random.choice(len(edges), size=k, replace=False)

        # Create network with failed edges removed
        G_test = G.copy()
        for idx in failed_edges:
            edge = edges[idx]
            idx_i = id_to_idx[edge[0]]
            idx_j = id_to_idx[edge[1]]
            if G_test.has_edge(idx_i, idx_j):
                G_test.remove_edge(idx_i, idx_j)

        # Check if network disconnected
        n_components = nx.number_connected_components(G_test)
        component_counts.append(n_components)

        if n_components > 1:
            n_failures += 1

    P_fail = n_failures / n_simulations
    mean_components = np.mean(component_counts)

    components = {
        'P_fail': P_fail,
        'n_simulations': n_simulations,
        'n_failures': n_failures,
        'mean_components': mean_components,
        'k_failures': k
    }

    assert 0 <= P_fail <= 1, f"Failure probability must be in [0,1], got {P_fail}"

    return P_fail, components


def add_strategic_loops(
    landscape: Landscape,
    mst_edges: List[Tuple[int, int]],
    n_loops: int = 1,
    criterion: str = 'betweenness'
) -> List[Tuple[int, int]]:
    """
    Add strategic loops to MST to improve robustness.

    Starting from a dendritic (MST) network, add n_loops additional edges
    to create redundant paths and improve robustness.

    Loop addition strategies:
    - 'betweenness': Add edges that maximize betweenness centrality reduction
    - 'shortest': Add shortest edges not in MST
    - 'bridge_protection': Add edges that protect high-betweenness bridges
    - 'random': Add random edges (baseline)

    Args:
        landscape: Landscape object
        mst_edges: Minimum spanning tree edges (dendritic base)
        n_loops: Number of loops to add
        criterion: Strategy for selecting which edges to add

    Returns:
        List of edges including MST + n_loops additional edges

    Properties:
    - Result has |MST| + n_loops edges
    - Result is connected (inherits from MST)
    - Loop edges chosen to maximize robustness improvement

    Reference:
        Hart (2024), Algorithm 2: Strategic loop addition
    """
    assert n_loops >= 0, "n_loops must be non-negative"
    assert criterion in ['betweenness', 'shortest', 'bridge_protection', 'random'], \
        f"Unknown criterion '{criterion}'"

    n = landscape.n_patches

    # Build MST network
    G_mst = nx.Graph()
    G_mst.add_nodes_from(range(n))

    id_to_idx = {patch.id: i for i, patch in enumerate(landscape.patches)}

    for (i, j) in mst_edges:
        idx_i = id_to_idx[i]
        idx_j = id_to_idx[j]
        G_mst.add_edge(idx_i, idx_j)

    # Get all possible edges not in MST
    mst_edge_set = set()
    for (i, j) in mst_edges:
        idx_i = id_to_idx[i]
        idx_j = id_to_idx[j]
        mst_edge_set.add((min(idx_i, idx_j), max(idx_i, idx_j)))

    candidate_edges = []
    for i, patch_i in enumerate(landscape.patches):
        for j, patch_j in enumerate(landscape.patches):
            if i < j:
                edge_tuple = (i, j)
                if edge_tuple not in mst_edge_set:
                    distance = landscape.graph[patch_i.id][patch_j.id]['distance']
                    candidate_edges.append((patch_i.id, patch_j.id, distance))

    if len(candidate_edges) == 0 or n_loops == 0:
        # No loops to add
        return mst_edges

    # Score candidate edges based on criterion
    if criterion == 'shortest':
        # Sort by distance (shortest first)
        candidate_edges.sort(key=lambda x: x[2])
        selected_edges = [(e[0], e[1]) for e in candidate_edges[:n_loops]]

    elif criterion == 'betweenness':
        # Select edges that most reduce betweenness centrality
        edge_betweenness = nx.edge_betweenness_centrality(G_mst)

        scores = []
        for (patch_i_id, patch_j_id, distance) in candidate_edges:
            idx_i = id_to_idx[patch_i_id]
            idx_j = id_to_idx[patch_j_id]

            # Test adding this edge
            G_test = G_mst.copy()
            G_test.add_edge(idx_i, idx_j)

            # Calculate new betweenness
            new_betweenness = nx.edge_betweenness_centrality(G_test)

            # Score = reduction in maximum betweenness
            old_max = max(edge_betweenness.values())
            new_max = max(new_betweenness.values())
            reduction = old_max - new_max

            scores.append((patch_i_id, patch_j_id, reduction))

        # Sort by betweenness reduction (largest first)
        scores.sort(key=lambda x: x[2], reverse=True)
        selected_edges = [(e[0], e[1]) for e in scores[:n_loops]]

    elif criterion == 'bridge_protection':
        # Find bridges (cut edges)
        bridges = list(nx.bridges(G_mst))

        if len(bridges) == 0:
            # No bridges to protect, fall back to shortest
            candidate_edges.sort(key=lambda x: x[2])
            selected_edges = [(e[0], e[1]) for e in candidate_edges[:n_loops]]
        else:
            # Add edges parallel to bridges
            scores = []
            for (patch_i_id, patch_j_id, distance) in candidate_edges:
                idx_i = id_to_idx[patch_i_id]
                idx_j = id_to_idx[patch_j_id]

                # Score = how well this edge protects bridges
                # Check if adding this edge eliminates any bridges
                G_test = G_mst.copy()
                G_test.add_edge(idx_i, idx_j)
                new_bridges = list(nx.bridges(G_test))

                bridges_eliminated = len(bridges) - len(new_bridges)
                scores.append((patch_i_id, patch_j_id, bridges_eliminated, distance))

            # Sort by bridges eliminated (most first), then by distance (shortest)
            scores.sort(key=lambda x: (-x[2], x[3]))
            selected_edges = [(e[0], e[1]) for e in scores[:n_loops]]

    else:  # random
        np.random.shuffle(candidate_edges)
        selected_edges = [(e[0], e[1]) for e in candidate_edges[:n_loops]]

    # Combine MST with selected loop edges
    result_edges = mst_edges + selected_edges

    return result_edges


def calculate_robustness_metrics(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    k_failures: int = 1
) -> RobustnessMetrics:
    """
    Calculate comprehensive robustness metrics for a network.

    Computes all robustness measures:
    - 2-edge-connectivity ρ₂
    - Overall redundancy score
    - Catastrophic failure probability
    - Number of loops
    - Per-edge redundancy scores

    Args:
        landscape: Landscape object
        edges: List of corridor edges
        k_failures: Number of simultaneous failures for P_fail

    Returns:
        RobustnessMetrics object with all computed metrics

    Example:
        >>> metrics = calculate_robustness_metrics(landscape, edges)
        >>> print(f"ρ₂ = {metrics.two_edge_connectivity:.3f}")
        >>> print(f"P_fail = {metrics.failure_probability:.3f}")
    """
    # Calculate 2-edge-connectivity
    rho_2, rho_components = calculate_two_edge_connectivity(landscape, edges)

    # Calculate per-edge redundancy
    edge_redundancy = calculate_edge_redundancy_scores(landscape, edges)

    # Calculate failure probability
    P_fail, fail_components = calculate_failure_probability(
        landscape, edges, k=k_failures
    )

    # Calculate number of loops
    n = landscape.n_patches
    n_edges = len(edges)

    # Build network to check connectivity
    G = nx.Graph()
    G.add_nodes_from(range(n))
    id_to_idx = {patch.id: i for i, patch in enumerate(landscape.patches)}

    for (i, j) in edges:
        idx_i = id_to_idx[i]
        idx_j = id_to_idx[j]
        G.add_edge(idx_i, idx_j)

    if nx.is_connected(G):
        n_loops = n_edges - (n - 1)
    else:
        n_components = nx.number_connected_components(G)
        min_edges = n - n_components
        n_loops = n_edges - min_edges

    n_loops = max(0, n_loops)

    # Overall redundancy score (average of metrics)
    # High redundancy = high ρ₂, low P_fail
    redundancy_score = (rho_2 + (1.0 - P_fail)) / 2.0

    return RobustnessMetrics(
        two_edge_connectivity=rho_2,
        redundancy_score=redundancy_score,
        failure_probability=P_fail,
        n_loops=n_loops,
        edge_redundancy=edge_redundancy
    )


def pareto_frontier_analysis(
    landscape: Landscape,
    max_loops: int = 10,
    loop_criterion: str = 'betweenness',
    **entropy_kwargs
) -> Dict[str, any]:
    """
    Generate Pareto frontier for entropy-robustness tradeoff.

    Explores the tradeoff between network entropy (flow efficiency) and
    robustness (redundancy) by adding loops to the MST.

    For each n_loops from 0 to max_loops:
    1. Build MST + n_loops network
    2. Calculate entropy H(L)
    3. Calculate robustness ρ₂
    4. Record tradeoff point

    Args:
        landscape: Landscape object
        max_loops: Maximum number of loops to explore
        loop_criterion: Strategy for loop addition (see add_strategic_loops)
        **entropy_kwargs: Parameters for entropy calculation

    Returns:
        Dict containing:
        - 'n_loops': Array of loop counts
        - 'entropy': Array of entropy values
        - 'robustness': Array of ρ₂ values
        - 'failure_prob': Array of P_fail values
        - 'pareto_points': List of Pareto-optimal (entropy, robustness) points
        - 'networks': List of edge lists for each configuration

    Example:
        >>> results = pareto_frontier_analysis(landscape, max_loops=5)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(results['entropy'], results['robustness'], 'o-')
        >>> plt.xlabel('Entropy H(L)')
        >>> plt.ylabel('Robustness ρ₂')
        >>> plt.title('Entropy-Robustness Pareto Frontier')
        >>> plt.show()

    Reference:
        Hart (2024), Figure 4C-D: Pareto frontier analysis
    """
    assert max_loops >= 0, "max_loops must be non-negative"

    # Build base MST
    dendritic_network = build_dendritic_network(landscape)
    mst_edges = dendritic_network.edges

    n_loops_array = []
    entropy_array = []
    robustness_array = []
    failure_prob_array = []
    networks = []

    # Explore loop additions from 0 to max_loops
    for n_loops in range(max_loops + 1):
        # Build network with n_loops
        if n_loops == 0:
            edges = mst_edges
        else:
            edges = add_strategic_loops(
                landscape, mst_edges, n_loops=n_loops, criterion=loop_criterion
            )

        # Calculate entropy
        H_total, entropy_components = calculate_entropy(landscape, edges, **entropy_kwargs)

        # Calculate robustness
        metrics = calculate_robustness_metrics(landscape, edges, k_failures=1)

        # Record results
        n_loops_array.append(n_loops)
        entropy_array.append(H_total)
        robustness_array.append(metrics.two_edge_connectivity)
        failure_prob_array.append(metrics.failure_probability)
        networks.append(edges)

    # Convert to numpy arrays
    n_loops_array = np.array(n_loops_array)
    entropy_array = np.array(entropy_array)
    robustness_array = np.array(robustness_array)
    failure_prob_array = np.array(failure_prob_array)

    # Identify Pareto-optimal points
    # (minimize entropy, maximize robustness)
    pareto_points = []
    for i in range(len(entropy_array)):
        is_pareto = True
        for j in range(len(entropy_array)):
            if i != j:
                # Check if j dominates i
                if (entropy_array[j] <= entropy_array[i] and
                    robustness_array[j] >= robustness_array[i] and
                    (entropy_array[j] < entropy_array[i] or
                     robustness_array[j] > robustness_array[i])):
                    is_pareto = False
                    break

        if is_pareto:
            pareto_points.append((entropy_array[i], robustness_array[i], n_loops_array[i]))

    results = {
        'n_loops': n_loops_array,
        'entropy': entropy_array,
        'robustness': robustness_array,
        'failure_prob': failure_prob_array,
        'pareto_points': pareto_points,
        'networks': networks
    }

    return results
