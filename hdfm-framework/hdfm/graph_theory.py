"""
Graph-theoretic constructs from Hart (2024) HDFM paper.

Implements the mathematical framework for testing paper claims:
- Strahler ordering for hierarchical corridor management
- Branching entropy (Hb) as connectivity metric
- Cyclomatic complexity (μ) for augmentation level
- Equivalent Connected Area (ECA) for connectivity measurement
- Resistance-weighted landscape graphs
- Corridor width prescription: W(e) = Wmin · α^(S(e)−1)
- Monte Carlo failure simulation (Table 1 reproduction)

Reference:
    Hart, J. (2024). "Hierarchical Dendritic Forest Management:
    A Graph-Theoretic Framework for Optimized Landscape Connectivity"
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from .landscape import Landscape, Patch


@dataclass
class StrahlerResult:
    """
    Result of Strahler ordering computation.

    Attributes:
        node_orders: Dict mapping node ID to Strahler order
        edge_orders: Dict mapping edge to Strahler order
        max_order: Maximum Strahler order in network
        order_distribution: Dict mapping order to count of edges
        root_node: Node used as root for ordering
    """
    node_orders: Dict[int, int]
    edge_orders: Dict[Tuple[int, int], int]
    max_order: int
    order_distribution: Dict[int, int]
    root_node: int


@dataclass
class AugmentedNetwork:
    """
    Represents an augmented dendritic network (MST + strategic loops).

    As defined in paper Definition 2.2:
    "A network Taug = (V, Eaug) is augmented dendritic if it consists of
    a minimum spanning tree backbone TMST plus k strategically selected
    augmentation edges, yielding cyclomatic complexity μ = k"

    Attributes:
        mst_edges: Edges in the MST backbone
        augmentation_edges: Strategic loop edges added for resilience
        all_edges: Combined edge list (mst_edges + augmentation_edges)
        mu: Cyclomatic complexity (number of independent cycles)
        landscape: Associated landscape object
    """
    mst_edges: List[Tuple[int, int]]
    augmentation_edges: List[Tuple[int, int]]
    all_edges: List[Tuple[int, int]]
    mu: int
    landscape: Landscape


@dataclass
class ResilienceResult:
    """
    Results from Monte Carlo failure simulation (Table 1 format).

    Attributes:
        edge_failure_prob: p value used in simulation
        mu: Cyclomatic complexity of network
        eca_retention: Expected ECA as fraction of intact network
        n_simulations: Number of Monte Carlo runs
        std_error: Standard error of ECA retention estimate
    """
    edge_failure_prob: float
    mu: int
    eca_retention: float
    n_simulations: int
    std_error: float


def compute_strahler_order(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    root: Optional[int] = None,
    root_selection: str = 'max_area'
) -> StrahlerResult:
    """
    Compute Strahler ordering for a dendritic network.

    Strahler ordering (Definition 2.4 from paper):
    (i) Leaf nodes: S(v) = 1
    (ii) If v has children with maximum order k, and ≥2 children have order k: S(v) = k + 1
    (iii) Otherwise: S(v) = max(child orders)

    Args:
        landscape: Landscape object
        edges: List of corridor edges (must form a tree)
        root: Optional root node ID. If None, selected by root_selection strategy
        root_selection: Strategy for root selection:
            - 'max_area': Largest patch (default)
            - 'max_quality': Highest quality patch
            - 'centroid': Patch closest to landscape centroid
            - 'max_Hb': Root that maximizes branching entropy

    Returns:
        StrahlerResult with node orders, edge orders, and distribution

    Raises:
        ValueError: If edges do not form a tree

    Reference:
        Hart (2024), Definition 2.4 and Section 2.5
    """
    n = landscape.n_patches

    # Build graph
    G = nx.Graph()
    id_to_idx = {patch.id: i for i, patch in enumerate(landscape.patches)}
    idx_to_id = {i: patch.id for i, patch in enumerate(landscape.patches)}

    G.add_nodes_from(range(n))
    for (i, j) in edges:
        G.add_edge(id_to_idx[i], id_to_idx[j])

    # Verify tree structure
    if not nx.is_tree(G):
        raise ValueError("Edges must form a tree for Strahler ordering. "
                        f"Got {len(edges)} edges for {n} nodes (need {n-1})")

    # Select root if not provided
    if root is None:
        if root_selection == 'max_area':
            root = max(landscape.patches, key=lambda p: p.area).id
        elif root_selection == 'max_quality':
            root = max(landscape.patches, key=lambda p: p.quality).id
        elif root_selection == 'centroid':
            cx = np.mean([p.x for p in landscape.patches])
            cy = np.mean([p.y for p in landscape.patches])
            root = min(landscape.patches,
                      key=lambda p: (p.x - cx)**2 + (p.y - cy)**2).id
        elif root_selection == 'max_Hb':
            # Try each node as root, pick one with max branching entropy
            best_root = landscape.patches[0].id
            best_Hb = -np.inf
            for patch in landscape.patches:
                result = _compute_strahler_with_root(G, id_to_idx[patch.id],
                                                     idx_to_id, edges)
                Hb = compute_branching_entropy_from_orders(result.edge_orders, edges)
                if Hb > best_Hb:
                    best_Hb = Hb
                    best_root = patch.id
            root = best_root
        else:
            raise ValueError(f"Unknown root_selection: {root_selection}")

    return _compute_strahler_with_root(G, id_to_idx[root], idx_to_id, edges)


def _compute_strahler_with_root(
    G: nx.Graph,
    root_idx: int,
    idx_to_id: Dict[int, int],
    edges: List[Tuple[int, int]]
) -> StrahlerResult:
    """Internal helper to compute Strahler order with given root."""
    # Create directed tree rooted at root
    T = nx.bfs_tree(G, root_idx)

    # Compute Strahler order via post-order traversal
    node_orders = {}

    def compute_order(node):
        children = list(T.successors(node))

        if not children:
            # Leaf node: order 1
            node_orders[node] = 1
            return 1

        # Get child orders
        child_orders = [compute_order(c) for c in children]
        max_order = max(child_orders)
        count_max = child_orders.count(max_order)

        if count_max >= 2:
            order = max_order + 1
        else:
            order = max_order

        node_orders[node] = order
        return order

    compute_order(root_idx)

    # Convert to patch IDs
    node_orders_by_id = {idx_to_id[idx]: order
                         for idx, order in node_orders.items()}

    # Compute edge orders (order of downstream node)
    edge_orders = {}
    for (i, j) in edges:
        # Edge order is the minimum of the two node orders
        # (represents the "stream order" of that corridor segment)
        order = min(node_orders_by_id[i], node_orders_by_id[j])
        edge_orders[(i, j)] = order

    # Compute distribution
    max_order = max(node_orders.values())
    order_distribution = {s: 0 for s in range(1, max_order + 1)}
    for order in edge_orders.values():
        order_distribution[order] = order_distribution.get(order, 0) + 1

    return StrahlerResult(
        node_orders=node_orders_by_id,
        edge_orders=edge_orders,
        max_order=max_order,
        order_distribution=order_distribution,
        root_node=idx_to_id[root_idx]
    )


def compute_branching_entropy(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    corridor_widths: Optional[Dict[Tuple[int, int], float]] = None,
    root: Optional[int] = None
) -> Tuple[float, Dict[str, any]]:
    """
    Compute branching entropy Hb as defined in paper Equation (1).

    Hb(T) = −Σs ps log2 ps

    where ps = As/ΣA is fractional area in Strahler order s.

    High Hb indicates balanced representation across hierarchical scales—
    hypothetically beneficial for species with diverse dispersal ranges.

    Args:
        landscape: Landscape object
        edges: List of corridor edges (must form a tree)
        corridor_widths: Optional dict mapping edge to width (meters).
                        If None, uses default width of 100m for all edges.
        root: Optional root node for Strahler ordering

    Returns:
        (Hb, components) where components contains:
        - 'Hb': Branching entropy value
        - 'order_areas': Dict mapping Strahler order to total corridor area
        - 'order_fractions': Dict mapping order to fractional area (ps)
        - 'strahler_result': Full Strahler ordering result

    Reference:
        Hart (2024), Definition 2.3 (Branching Entropy)
    """
    # Compute Strahler ordering
    strahler = compute_strahler_order(landscape, edges, root)

    # Default corridor widths if not provided
    if corridor_widths is None:
        corridor_widths = {edge: 100.0 for edge in edges}

    # Compute corridor areas by Strahler order
    # Area = length × width for each edge
    order_areas = {}
    for edge in edges:
        order = strahler.edge_orders.get(edge,
                    strahler.edge_orders.get((edge[1], edge[0]), 1))

        # Get edge length
        i, j = edge
        patch_i = landscape.get_patch(i)
        patch_j = landscape.get_patch(j)
        length = patch_i.distance_to(patch_j)

        # Get width
        width = corridor_widths.get(edge, corridor_widths.get((j, i), 100.0))

        # Accumulate area
        area = length * width
        order_areas[order] = order_areas.get(order, 0) + area

    # Compute fractional areas
    total_area = sum(order_areas.values())
    order_fractions = {order: area / total_area
                       for order, area in order_areas.items()}

    # Compute branching entropy
    Hb = 0.0
    for order, ps in order_fractions.items():
        if ps > 0:
            Hb -= ps * np.log2(ps)

    components = {
        'Hb': Hb,
        'order_areas': order_areas,
        'order_fractions': order_fractions,
        'strahler_result': strahler,
        'total_corridor_area': total_area
    }

    return Hb, components


def compute_branching_entropy_from_orders(
    edge_orders: Dict[Tuple[int, int], int],
    edges: List[Tuple[int, int]],
    equal_length: bool = True
) -> float:
    """
    Simplified branching entropy using edge counts instead of areas.

    Useful when corridor widths/lengths are not specified.
    Assumes all edges have equal contribution to area.
    """
    order_counts = {}
    for edge in edges:
        order = edge_orders.get(edge, edge_orders.get((edge[1], edge[0]), 1))
        order_counts[order] = order_counts.get(order, 0) + 1

    total = sum(order_counts.values())
    Hb = 0.0
    for count in order_counts.values():
        ps = count / total
        if ps > 0:
            Hb -= ps * np.log2(ps)

    return Hb


def compute_cyclomatic_complexity(
    landscape: Landscape,
    edges: List[Tuple[int, int]]
) -> int:
    """
    Compute cyclomatic complexity μ of a network.

    μ = |E| - |V| + 1 for a connected graph

    For a tree (MST): μ = 0
    For augmented dendritic with k loops: μ = k

    Args:
        landscape: Landscape object
        edges: List of corridor edges

    Returns:
        Cyclomatic complexity μ (number of independent cycles)

    Reference:
        Hart (2024), Definition 2.1 and 2.2
    """
    n = landscape.n_patches
    m = len(edges)

    # Build graph to check connectivity
    G = nx.Graph()
    id_to_idx = {patch.id: i for i, patch in enumerate(landscape.patches)}
    G.add_nodes_from(range(n))
    for (i, j) in edges:
        G.add_edge(id_to_idx[i], id_to_idx[j])

    if not nx.is_connected(G):
        # For disconnected graphs, μ = m - n + c where c = components
        c = nx.number_connected_components(G)
        mu = m - n + c
    else:
        mu = m - n + 1

    return max(0, mu)


def compute_equivalent_connected_area(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    resistance_weights: Optional[Dict[Tuple[int, int], float]] = None
) -> Tuple[float, Dict[str, any]]:
    """
    Compute Equivalent Connected Area (ECA) for connectivity measurement.

    ECA is a connectivity metric that accounts for patch areas and
    inter-patch connectivity:

    ECA = Σᵢ Σⱼ √(Aᵢ × Aⱼ) × pᵢⱼ

    where pᵢⱼ is the connectivity (probability of movement) between
    patches i and j, computed as exp(-α × d_eff) where d_eff is the
    effective distance through the corridor network.

    Args:
        landscape: Landscape object
        edges: List of corridor edges
        resistance_weights: Optional dict mapping edge to resistance weight.
                          If None, uses Euclidean distance.

    Returns:
        (ECA, components) where components contains:
        - 'ECA': Equivalent Connected Area
        - 'ECA_max': Maximum possible ECA (fully connected)
        - 'ECA_normalized': ECA / ECA_max
        - 'connectivity_matrix': pᵢⱼ values

    Reference:
        Hart (2024), Section 6.3 (ECA retention metric in Table 1)
    """
    n = landscape.n_patches
    alpha = 0.001  # Dispersal decay parameter (1/m)

    # Build weighted graph
    G = nx.Graph()
    id_to_idx = {patch.id: i for i, patch in enumerate(landscape.patches)}

    G.add_nodes_from(range(n))
    for (i, j) in edges:
        idx_i, idx_j = id_to_idx[i], id_to_idx[j]

        if resistance_weights and (i, j) in resistance_weights:
            weight = resistance_weights[(i, j)]
        elif resistance_weights and (j, i) in resistance_weights:
            weight = resistance_weights[(j, i)]
        else:
            # Default: Euclidean distance
            weight = landscape.get_patch(i).distance_to(landscape.get_patch(j))

        G.add_edge(idx_i, idx_j, weight=weight)

    # Compute shortest path distances through network
    try:
        path_lengths = dict(nx.all_pairs_dijkstra_path_length(G))
    except nx.NetworkXError:
        # Graph is disconnected
        path_lengths = {}
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            component_lengths = dict(nx.all_pairs_dijkstra_path_length(subgraph))
            for source, targets in component_lengths.items():
                if source not in path_lengths:
                    path_lengths[source] = {}
                path_lengths[source].update(targets)

    # Compute connectivity matrix pᵢⱼ = exp(-α × d_eff)
    connectivity = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                connectivity[i, j] = 1.0
            elif i in path_lengths and j in path_lengths[i]:
                d_eff = path_lengths[i][j]
                connectivity[i, j] = np.exp(-alpha * d_eff)
            else:
                connectivity[i, j] = 0.0  # Disconnected

    # Compute ECA
    areas = np.array([p.area for p in landscape.patches])
    ECA = 0.0
    for i in range(n):
        for j in range(n):
            ECA += np.sqrt(areas[i] * areas[j]) * connectivity[i, j]

    # Compute max ECA (fully connected, pᵢⱼ = 1 for all pairs)
    ECA_max = 0.0
    for i in range(n):
        for j in range(n):
            ECA_max += np.sqrt(areas[i] * areas[j])

    components = {
        'ECA': ECA,
        'ECA_max': ECA_max,
        'ECA_normalized': ECA / ECA_max if ECA_max > 0 else 0.0,
        'connectivity_matrix': connectivity
    }

    return ECA, components


def compute_corridor_widths_strahler(
    strahler_result: StrahlerResult,
    W_min: float = 50.0,
    alpha: float = 2.0
) -> Dict[Tuple[int, int], float]:
    """
    Compute corridor widths using Strahler-scaled prescription.

    W(e) = W_min × α^(S(e)-1)

    From paper Section 2.5:
    "With W_min = 50m and α = 2.0 (doubling width per order), this yields:
    Order 1 = 50m, Order 2 = 100m, Order 3 = 200m, Order 4 = 400m"

    Args:
        strahler_result: Result from compute_strahler_order
        W_min: Minimum corridor width for Order 1 (meters)
        alpha: Scaling factor (width multiplier per Strahler order increment)

    Returns:
        Dict mapping edge (i, j) to prescribed width in meters

    Reference:
        Hart (2024), Equation (2)
    """
    widths = {}
    for edge, order in strahler_result.edge_orders.items():
        width = W_min * (alpha ** (order - 1))
        widths[edge] = width

    return widths


def build_augmented_dendritic_network(
    landscape: Landscape,
    mu: int = 2,
    augmentation_criterion: str = 'betweenness'
) -> AugmentedNetwork:
    """
    Build an augmented dendritic network with μ strategic loops.

    As defined in paper Definition 2.2:
    "A network Taug = (V, Eaug) is augmented dendritic if it consists of
    a minimum spanning tree backbone TMST plus k strategically selected
    augmentation edges, yielding cyclomatic complexity μ = k"

    Args:
        landscape: Landscape object
        mu: Target cyclomatic complexity (number of loops to add)
        augmentation_criterion: Strategy for selecting augmentation edges:
            - 'betweenness': Maximize betweenness reduction (default)
            - 'bridge_protection': Protect vulnerable bridges
            - 'shortest': Add shortest non-MST edges

    Returns:
        AugmentedNetwork object

    Reference:
        Hart (2024), Definition 2.2 and Section 6.2
    """
    from .network import build_dendritic_network
    from .robustness import add_strategic_loops

    # Build MST backbone
    mst_network = build_dendritic_network(landscape)
    mst_edges = list(mst_network.edges)

    if mu == 0:
        return AugmentedNetwork(
            mst_edges=mst_edges,
            augmentation_edges=[],
            all_edges=mst_edges,
            mu=0,
            landscape=landscape
        )

    # Add strategic loops
    all_edges = add_strategic_loops(
        landscape,
        mst_edges,
        n_loops=mu,
        criterion=augmentation_criterion
    )

    # Identify which edges are augmentation edges
    mst_set = set(mst_edges)
    augmentation_edges = [e for e in all_edges if e not in mst_set]

    return AugmentedNetwork(
        mst_edges=mst_edges,
        augmentation_edges=augmentation_edges,
        all_edges=all_edges,
        mu=len(augmentation_edges),
        landscape=landscape
    )


def monte_carlo_resilience_simulation(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    edge_failure_probs: List[float] = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
    n_simulations: int = 1000
) -> List[ResilienceResult]:
    """
    Run Monte Carlo failure simulation to compute ECA retention.

    Reproduces Table 1 from the paper:
    "We simulate 1,000 Monte Carlo realizations at each edge failure
    probability p ∈ {0.01, 0.02, 0.05, 0.10, 0.15, 0.20}"

    For each simulation:
    1. Each edge fails independently with probability p
    2. Compute ECA of resulting (possibly disconnected) network
    3. Report mean ECA retention as fraction of intact-network ECA

    Args:
        landscape: Landscape object
        edges: List of corridor edges
        edge_failure_probs: List of failure probabilities to test
        n_simulations: Number of Monte Carlo simulations per probability

    Returns:
        List of ResilienceResult for each failure probability

    Reference:
        Hart (2024), Table 1 and Section 6.3
    """
    # Compute baseline ECA (intact network)
    baseline_eca, _ = compute_equivalent_connected_area(landscape, edges)

    # Compute cyclomatic complexity
    mu = compute_cyclomatic_complexity(landscape, edges)

    results = []

    for p in edge_failure_probs:
        eca_retentions = []

        for _ in range(n_simulations):
            # Randomly fail edges
            surviving_edges = []
            for edge in edges:
                if np.random.random() > p:  # Edge survives
                    surviving_edges.append(edge)

            # Compute ECA of surviving network
            if len(surviving_edges) > 0:
                eca, _ = compute_equivalent_connected_area(landscape, surviving_edges)
            else:
                eca = 0.0

            eca_retentions.append(eca / baseline_eca if baseline_eca > 0 else 0.0)

        mean_retention = np.mean(eca_retentions)
        std_error = np.std(eca_retentions) / np.sqrt(n_simulations)

        results.append(ResilienceResult(
            edge_failure_prob=p,
            mu=mu,
            eca_retention=mean_retention,
            n_simulations=n_simulations,
            std_error=std_error
        ))

    return results


def generate_table_1(
    landscape: Landscape,
    max_mu: int = 3,
    edge_failure_probs: List[float] = [0.01, 0.05, 0.10, 0.20],
    n_simulations: int = 1000
) -> Dict[str, any]:
    """
    Generate Table 1 from the paper: ECA retention under stochastic edge failure.

    Creates a table comparing networks with different cyclomatic complexity (μ)
    under various edge failure probabilities.

    Args:
        landscape: Landscape object
        max_mu: Maximum μ to test (0, 1, 2, ..., max_mu)
        edge_failure_probs: Failure probabilities to test
        n_simulations: Monte Carlo simulations per configuration

    Returns:
        Dict containing:
        - 'table': 2D array of ECA retention values [prob_idx, mu]
        - 'probabilities': List of tested probabilities
        - 'mu_values': List of tested μ values
        - 'corridor_areas': Dict mapping μ to corridor area
        - 'formatted_table': String representation of table

    Reference:
        Hart (2024), Table 1
    """
    mu_values = list(range(max_mu + 1))

    # Build networks for each μ
    networks = {}
    corridor_areas = {}

    for mu in mu_values:
        aug_network = build_augmented_dendritic_network(landscape, mu=mu)
        networks[mu] = aug_network.all_edges

        # Calculate corridor area (assuming 100m default width)
        total_length = sum(
            landscape.get_patch(e[0]).distance_to(landscape.get_patch(e[1]))
            for e in aug_network.all_edges
        )
        corridor_areas[mu] = total_length * 100  # Area in m²

    # Run simulations
    table = np.zeros((len(edge_failure_probs), len(mu_values)))

    for mu_idx, mu in enumerate(mu_values):
        results = monte_carlo_resilience_simulation(
            landscape,
            networks[mu],
            edge_failure_probs,
            n_simulations
        )

        for prob_idx, result in enumerate(results):
            table[prob_idx, mu_idx] = result.eca_retention

    # Format table as string
    header = "Edge Failure Prob. | " + " | ".join([f"μ = {mu}" for mu in mu_values])
    separator = "-" * len(header)

    rows = [header, separator]
    for prob_idx, p in enumerate(edge_failure_probs):
        values = " | ".join([f"{table[prob_idx, mu_idx]:.2f}"
                            for mu_idx in range(len(mu_values))])
        rows.append(f"p = {p:.2f}           | {values}")

    formatted_table = "\n".join(rows)

    return {
        'table': table,
        'probabilities': edge_failure_probs,
        'mu_values': mu_values,
        'corridor_areas': corridor_areas,
        'formatted_table': formatted_table,
        'networks': networks
    }


def compute_edge_criticality(
    landscape: Landscape,
    edges: List[Tuple[int, int]]
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Compute edge criticality for identifying high-vulnerability links.

    From paper Section 6.2:
    "Criticality analysis identifies two high-vulnerability edges."

    Criticality is based on:
    1. Impact on ECA if edge is removed
    2. Betweenness centrality
    3. Whether edge is a bridge (cut edge)

    Args:
        landscape: Landscape object
        edges: List of corridor edges

    Returns:
        Dict mapping edge to criticality metrics:
        - 'eca_impact': Fractional ECA loss if removed
        - 'betweenness': Edge betweenness centrality
        - 'is_bridge': Whether removal disconnects network
        - 'criticality_score': Combined score [0, 1]

    Reference:
        Hart (2024), Section 6.2
    """
    n = landscape.n_patches

    # Compute baseline ECA
    baseline_eca, _ = compute_equivalent_connected_area(landscape, edges)

    # Build graph
    G = nx.Graph()
    id_to_idx = {patch.id: i for i, patch in enumerate(landscape.patches)}
    G.add_nodes_from(range(n))
    for (i, j) in edges:
        G.add_edge(id_to_idx[i], id_to_idx[j])

    # Compute betweenness
    edge_betweenness = nx.edge_betweenness_centrality(G)
    max_betweenness = max(edge_betweenness.values()) if edge_betweenness else 1.0

    # Find bridges
    bridges = set(nx.bridges(G))

    criticality = {}

    for edge in edges:
        i, j = edge
        idx_i, idx_j = id_to_idx[i], id_to_idx[j]

        # Compute ECA impact
        remaining_edges = [e for e in edges if e != edge]
        if len(remaining_edges) > 0:
            reduced_eca, _ = compute_equivalent_connected_area(landscape, remaining_edges)
            eca_impact = 1.0 - (reduced_eca / baseline_eca) if baseline_eca > 0 else 1.0
        else:
            eca_impact = 1.0

        # Get betweenness (normalized)
        edge_key = (idx_i, idx_j) if (idx_i, idx_j) in edge_betweenness else (idx_j, idx_i)
        betweenness = edge_betweenness.get(edge_key, 0.0) / max_betweenness

        # Check if bridge
        is_bridge = (idx_i, idx_j) in bridges or (idx_j, idx_i) in bridges

        # Combined criticality score
        criticality_score = 0.4 * eca_impact + 0.4 * betweenness + 0.2 * float(is_bridge)

        criticality[edge] = {
            'eca_impact': eca_impact,
            'betweenness': betweenness,
            'is_bridge': is_bridge,
            'criticality_score': criticality_score
        }

    return criticality


def identify_critical_edges(
    landscape: Landscape,
    edges: List[Tuple[int, int]],
    top_k: int = 2
) -> List[Tuple[Tuple[int, int], Dict[str, float]]]:
    """
    Identify the top-k most critical edges.

    Args:
        landscape: Landscape object
        edges: List of corridor edges
        top_k: Number of critical edges to return

    Returns:
        List of (edge, criticality_metrics) tuples, sorted by criticality

    Reference:
        Hart (2024), Section 6.2: "Criticality analysis identifies
        two high-vulnerability edges"
    """
    criticality = compute_edge_criticality(landscape, edges)

    # Sort by criticality score
    sorted_edges = sorted(
        criticality.items(),
        key=lambda x: x[1]['criticality_score'],
        reverse=True
    )

    return sorted_edges[:top_k]


class ResistanceWeightedLandscape(Landscape):
    """
    Landscape with resistance-weighted edges for movement cost modeling.

    From paper Section 2.1:
    "We represent a forest landscape as a weighted graph G = (V, E, w)
    where w: E → ℝ⁺ assigns resistance weights to edges."

    Resistance weights incorporate land cover, topography, and
    anthropogenic barriers following circuit-theory parameterization.
    """

    # Standard resistance values from paper Section 6.1
    RESISTANCE_VALUES = {
        'old_growth': 1.0,
        'mature_forest': 1.2,
        'young_forest': 2.0,
        'clearcut': 8.0,
        'developed': 500.0,
        'water': 100.0,
        'road': 50.0
    }

    def __init__(
        self,
        patches: List[Patch],
        edge_resistances: Optional[Dict[Tuple[int, int], float]] = None,
        default_resistance: float = 1.0
    ):
        """
        Initialize resistance-weighted landscape.

        Args:
            patches: List of Patch objects
            edge_resistances: Optional dict mapping (i, j) to resistance weight
            default_resistance: Default resistance for unspecified edges
        """
        super().__init__(patches)

        self.edge_resistances = edge_resistances or {}
        self.default_resistance = default_resistance

        # Update graph with resistance weights
        self._add_resistance_weights()

    def _add_resistance_weights(self):
        """Add resistance weights to graph edges."""
        for (i, j) in self.graph.edges():
            if (i, j) in self.edge_resistances:
                resistance = self.edge_resistances[(i, j)]
            elif (j, i) in self.edge_resistances:
                resistance = self.edge_resistances[(j, i)]
            else:
                # Default: resistance proportional to distance
                distance = self.graph[i][j]['distance']
                resistance = distance * self.default_resistance

            self.graph[i][j]['resistance'] = resistance
            self.graph[i][j]['weight'] = resistance  # Use for MST

    def get_resistance(self, i: int, j: int) -> float:
        """Get resistance weight for edge (i, j)."""
        if self.graph.has_edge(i, j):
            return self.graph[i][j].get('resistance', self.default_resistance)
        return float('inf')

    def resistance_matrix(self) -> np.ndarray:
        """
        Compute pairwise resistance matrix.

        Returns:
            n x n matrix where R[i,j] is resistance between patches i and j
        """
        n = self.n_patches
        R = np.full((n, n), np.inf)
        np.fill_diagonal(R, 0)

        id_to_idx = {p.id: i for i, p in enumerate(self.patches)}

        for (i, j) in self.graph.edges():
            idx_i = id_to_idx[i]
            idx_j = id_to_idx[j]
            resistance = self.get_resistance(i, j)
            R[idx_i, idx_j] = resistance
            R[idx_j, idx_i] = resistance

        return R


def verify_proposition_3_1(
    landscape: Landscape,
    mst_edges: List[Tuple[int, int]],
    alternative_edges: List[Tuple[int, int]]
) -> Dict[str, any]:
    """
    Verify Proposition 3.1: MST achieves minimum total edge resistance.

    "Under graph-theoretic assumptions, the minimum total edge resistance
    achieving full connectivity is attained by a tree (minimum spanning tree, MST)."

    Args:
        landscape: Landscape object
        mst_edges: Edges from MST
        alternative_edges: Alternative connected edge set

    Returns:
        Dict containing verification results
    """
    # Compute total resistance (using distance as proxy)
    mst_total = sum(
        landscape.get_patch(e[0]).distance_to(landscape.get_patch(e[1]))
        for e in mst_edges
    )

    alt_total = sum(
        landscape.get_patch(e[0]).distance_to(landscape.get_patch(e[1]))
        for e in alternative_edges
    )

    # Check connectivity of alternative
    n = landscape.n_patches
    G = nx.Graph()
    id_to_idx = {p.id: i for i, p in enumerate(landscape.patches)}
    G.add_nodes_from(range(n))
    for (i, j) in alternative_edges:
        G.add_edge(id_to_idx[i], id_to_idx[j])
    alt_connected = nx.is_connected(G)

    return {
        'mst_total_resistance': mst_total,
        'alternative_total_resistance': alt_total,
        'alternative_is_connected': alt_connected,
        'proposition_holds': mst_total <= alt_total or not alt_connected,
        'difference': alt_total - mst_total,
        'difference_percent': ((alt_total - mst_total) / mst_total * 100)
                              if mst_total > 0 else 0
    }


def verify_proposition_3_3(
    landscape: Landscape,
    edge_failure_prob: float = 0.05,
    n_simulations: int = 1000
) -> Dict[str, any]:
    """
    Verify Proposition 3.3: Augmented dendritic networks achieve
    near-optimal efficiency while substantially improving resilience.

    "Augmented dendritic networks with small μ (2–3) achieve near-optimal
    efficiency while substantially improving resilience compared to pure
    dendritic (μ = 0) networks."

    Args:
        landscape: Landscape object
        edge_failure_prob: Edge failure probability for resilience testing
        n_simulations: Number of Monte Carlo simulations

    Returns:
        Dict containing verification results matching Table 1 format
    """
    results = generate_table_1(
        landscape,
        max_mu=3,
        edge_failure_probs=[edge_failure_prob],
        n_simulations=n_simulations
    )

    mu_0_retention = results['table'][0, 0]
    mu_2_retention = results['table'][0, 2]

    # Calculate efficiency (corridor area)
    area_0 = results['corridor_areas'][0]
    area_2 = results['corridor_areas'][2]
    area_increase = (area_2 - area_0) / area_0 * 100

    # Calculate resilience improvement
    resilience_improvement = (mu_2_retention - mu_0_retention) / mu_0_retention * 100

    return {
        'mu_0_eca_retention': mu_0_retention,
        'mu_2_eca_retention': mu_2_retention,
        'resilience_improvement_percent': resilience_improvement,
        'area_increase_percent': area_increase,
        'proposition_holds': resilience_improvement > 0 and area_increase < 15,
        'full_table': results['formatted_table']
    }
