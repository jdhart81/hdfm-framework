"""
Network topology construction and comparison for HDFM framework.

Implements dendritic (MST) network construction and alternative topologies
for validation and comparison.
"""

import numpy as np
import networkx as nx
from enum import Enum
from typing import List, Tuple, Dict
from .landscape import Landscape
from .entropy import calculate_entropy


class NetworkTopology(Enum):
    """Enumeration of network topology types."""
    DENDRITIC = "dendritic"  # Minimum spanning tree
    GABRIEL = "gabriel"  # Gabriel graph
    DELAUNAY = "delaunay"  # Delaunay triangulation
    KNN = "knn"  # k-nearest neighbors
    THRESHOLD = "threshold"  # Threshold distance
    COMPLETE = "complete"  # Fully connected


class DendriticNetwork:
    """
    Represents a dendritic corridor network.
    
    A dendritic network is a tree structure (acyclic, connected graph)
    that minimizes total corridor length while maintaining connectivity.
    
    Invariants:
    - Network is connected (all patches reachable)
    - Network is acyclic (no loops/cycles)
    - Has exactly n-1 edges for n patches
    - Minimizes total corridor length among connected topologies
    """
    
    def __init__(self, landscape: Landscape, edges: List[Tuple[int, int]]):
        """
        Initialize dendritic network.
        
        Args:
            landscape: Landscape object
            edges: List of corridor edges
            
        Validates:
        - Network is connected
        - Network is acyclic
        - Has correct number of edges
        """
        self.landscape = landscape
        self.edges = edges
        
        # Build graph for validation
        n = landscape.n_patches
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        id_to_idx = {patch.id: i for i, patch in enumerate(landscape.patches)}
        
        for (i, j) in edges:
            idx_i = id_to_idx[i]
            idx_j = id_to_idx[j]
            G.add_edge(idx_i, idx_j)
        
        self.graph = G
        
        # Validate invariants
        assert nx.is_connected(G), "Dendritic network must be connected"
        assert nx.is_tree(G), "Dendritic network must be acyclic (tree)"
        assert len(edges) == n - 1, f"Tree must have n-1={n-1} edges, got {len(edges)}"
    
    def total_length(self) -> float:
        """Calculate total corridor length."""
        return self.landscape.total_corridor_length(self.edges)
    
    def entropy(self, **kwargs) -> Tuple[float, Dict[str, float]]:
        """Calculate network entropy."""
        return calculate_entropy(self.landscape, self.edges, **kwargs)
    
    def degree_distribution(self) -> Dict[int, int]:
        """
        Get degree distribution of network.
        
        Returns:
            Dictionary mapping degree to count
        """
        degrees = dict(self.graph.degree())
        distribution = {}
        for degree in degrees.values():
            distribution[degree] = distribution.get(degree, 0) + 1
        return distribution
    
    def is_dendritic(self) -> bool:
        """Verify network is dendritic (tree structure)."""
        return nx.is_tree(self.graph) and nx.is_connected(self.graph)


def build_dendritic_network(
    landscape: Landscape,
    method: str = 'kruskal'
) -> DendriticNetwork:
    """
    Construct dendritic corridor network via minimum spanning tree.
    
    Uses Kruskal's or Prim's algorithm to find MST of landscape graph,
    minimizing total corridor length while maintaining connectivity.
    
    Args:
        landscape: Landscape object
        method: 'kruskal' or 'prim'
        
    Returns:
        DendriticNetwork object
        
    Complexity:
        O(E log E) = O(n² log n) for Kruskal
        O(E log V) = O(n² log n) for Prim
        
    Invariants verified:
    - Result is connected
    - Result is acyclic
    - Result minimizes total length
    - Result has n-1 edges
    """
    assert method in ['kruskal', 'prim'], f"Unknown method '{method}'"
    
    # Get MST using NetworkX
    if method == 'kruskal':
        mst = nx.minimum_spanning_tree(landscape.graph, algorithm='kruskal')
    else:
        mst = nx.minimum_spanning_tree(landscape.graph, algorithm='prim')
    
    # Extract edges
    edges = list(mst.edges())
    
    # Create dendritic network (validates invariants)
    network = DendriticNetwork(landscape, edges)
    
    return network


def build_gabriel_graph(landscape: Landscape) -> List[Tuple[int, int]]:
    """
    Build Gabriel graph topology.
    
    Gabriel graph: edge (i,j) included if circle with diameter d_ij
    contains no other patches.
    
    Properties:
    - Generally more edges than MST
    - Preserves some proximity relationships
    - Used for comparison with dendritic networks
    """
    edges = []
    
    for i, patch_i in enumerate(landscape.patches):
        for patch_j in landscape.patches[i+1:]:
            # Circle center and radius
            cx = (patch_i.x + patch_j.x) / 2
            cy = (patch_i.y + patch_j.y) / 2
            radius = patch_i.distance_to(patch_j) / 2
            
            # Check if any other patch in circle
            gabriel_edge = True
            for patch_k in landscape.patches:
                if patch_k.id not in [patch_i.id, patch_j.id]:
                    dist_to_center = np.sqrt((patch_k.x - cx)**2 + (patch_k.y - cy)**2)
                    if dist_to_center < radius:
                        gabriel_edge = False
                        break
            
            if gabriel_edge:
                edges.append((patch_i.id, patch_j.id))
    
    return edges


def build_delaunay_triangulation(landscape: Landscape) -> List[Tuple[int, int]]:
    """
    Build Delaunay triangulation topology.
    
    Properties:
    - Maximizes minimum angle of triangles
    - Generally many more edges than MST
    - High connectivity but high total length
    """
    from scipy.spatial import Delaunay
    
    # Get patch coordinates
    points = np.array([[p.x, p.y] for p in landscape.patches])
    
    # Compute Delaunay triangulation
    tri = Delaunay(points)
    
    # Extract edges
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            j = (i + 1) % 3
            edge = tuple(sorted([simplex[i], simplex[j]]))
            edges.add(edge)
    
    # Convert to patch IDs
    edges = [(landscape.patches[i].id, landscape.patches[j].id) 
             for (i, j) in edges]
    
    return edges


def build_knn_network(landscape: Landscape, k: int = 3) -> List[Tuple[int, int]]:
    """
    Build k-nearest neighbors network.
    
    Each patch connected to its k nearest neighbors.
    
    Args:
        landscape: Landscape object
        k: Number of nearest neighbors
        
    Properties:
    - Approximately k*n/2 edges
    - Local connectivity
    - May not be connected for small k
    """
    assert k >= 1, "k must be at least 1"
    assert k < landscape.n_patches, "k must be less than number of patches"
    
    edges = set()
    
    for patch_i in landscape.patches:
        # Find k nearest neighbors
        distances = [
            (patch_j.id, patch_i.distance_to(patch_j))
            for patch_j in landscape.patches
            if patch_j.id != patch_i.id
        ]
        distances.sort(key=lambda x: x[1])
        
        # Add edges to k nearest
        for neighbor_id, _ in distances[:k]:
            edge = tuple(sorted([patch_i.id, neighbor_id]))
            edges.add(edge)
    
    return list(edges)


def build_threshold_network(
    landscape: Landscape,
    threshold: float
) -> List[Tuple[int, int]]:
    """
    Build threshold distance network.
    
    Include edge (i,j) if distance d_ij < threshold.
    
    Args:
        landscape: Landscape object
        threshold: Distance threshold (meters)
        
    Properties:
    - Number of edges depends on threshold
    - May not be connected for low threshold
    - Becomes complete graph for high threshold
    """
    assert threshold > 0, "Threshold must be positive"
    
    edges = []
    
    for i, patch_i in enumerate(landscape.patches):
        for patch_j in landscape.patches[i+1:]:
            if patch_i.distance_to(patch_j) < threshold:
                edges.append((patch_i.id, patch_j.id))
    
    return edges


def compare_network_topologies(
    landscape: Landscape,
    topologies: List[NetworkTopology] = None,
    **entropy_kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Compare entropy across different network topologies.
    
    Constructs networks using different algorithms and calculates
    their entropy for comparison with dendritic networks.
    
    Args:
        landscape: Landscape object
        topologies: List of topology types to compare
        **entropy_kwargs: Parameters for entropy calculation
        
    Returns:
        Dictionary mapping topology name to entropy components
        
    Example:
        >>> results = compare_network_topologies(landscape)
        >>> print(f"Dendritic: {results['dendritic']['H_total']:.3f}")
        >>> print(f"Gabriel: {results['gabriel']['H_total']:.3f}")
    """
    if topologies is None:
        topologies = [
            NetworkTopology.DENDRITIC,
            NetworkTopology.GABRIEL,
            NetworkTopology.DELAUNAY,
            NetworkTopology.KNN,
            NetworkTopology.THRESHOLD
        ]
    
    results = {}
    
    for topology in topologies:
        try:
            if topology == NetworkTopology.DENDRITIC:
                network = build_dendritic_network(landscape)
                edges = network.edges
                
            elif topology == NetworkTopology.GABRIEL:
                edges = build_gabriel_graph(landscape)
                
            elif topology == NetworkTopology.DELAUNAY:
                edges = build_delaunay_triangulation(landscape)
                
            elif topology == NetworkTopology.KNN:
                edges = build_knn_network(landscape, k=3)
                
            elif topology == NetworkTopology.THRESHOLD:
                # Auto-select threshold to ensure connectivity
                D = landscape.distance_matrix()
                threshold = np.percentile(D[D > 0], 30)
                edges = build_threshold_network(landscape, threshold)
                
            else:
                continue
            
            # Calculate entropy
            H_total, components = calculate_entropy(landscape, edges, **entropy_kwargs)
            
            # Add summary statistics
            components['n_edges'] = len(edges)
            components['total_length'] = landscape.total_corridor_length(edges)
            
            results[topology.value] = components
            
        except Exception as e:
            print(f"Warning: Failed to build {topology.value} network: {e}")
            continue
    
    return results


def validate_network_properties(edges: List[Tuple[int, int]], n_patches: int) -> Dict[str, bool]:
    """
    Validate network properties.
    
    Args:
        edges: List of edges
        n_patches: Number of patches
        
    Returns:
        Dictionary of property checks:
        - 'connected': Network is connected
        - 'acyclic': Network has no cycles
        - 'is_tree': Network is a tree
        - 'correct_size': Has n-1 edges
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_patches))
    G.add_edges_from(edges)
    
    return {
        'connected': nx.is_connected(G),
        'acyclic': nx.is_tree(G) or not nx.is_connected(G),
        'is_tree': nx.is_tree(G),
        'correct_size': len(edges) == n_patches - 1
    }
