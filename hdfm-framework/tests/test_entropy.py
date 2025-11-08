"""
Unit tests for entropy calculations.

Tests invariants and properties of entropy functions.
"""

import pytest
import numpy as np
from hdfm import (
    SyntheticLandscape,
    build_dendritic_network,
    movement_entropy,
    connectivity_constraint,
    forest_topology_penalty,
    disturbance_response_penalty,
    calculate_entropy
)


class TestMovementEntropy:
    """Test movement entropy calculations."""
    
    def test_non_negative(self):
        """Invariant: Movement entropy must be non-negative."""
        landscape = SyntheticLandscape(n_patches=10, random_seed=42)
        network = build_dendritic_network(landscape)
        
        H_mov = movement_entropy(landscape, network.edges)
        
        assert H_mov >= 0, "Movement entropy must be non-negative"
    
    def test_decreases_with_fewer_corridors(self):
        """Property: Fewer corridors → lower movement entropy."""
        landscape = SyntheticLandscape(n_patches=10, random_seed=42)
        network = build_dendritic_network(landscape)
        
        # Full network
        H_full = movement_entropy(landscape, network.edges)
        
        # Reduced network
        H_reduced = movement_entropy(landscape, network.edges[:5])
        
        assert H_reduced <= H_full, "Reduced network should have lower or equal entropy"
    
    def test_zero_for_no_corridors(self):
        """Boundary: Zero corridors → zero or near-zero entropy."""
        landscape = SyntheticLandscape(n_patches=10, random_seed=42)
        
        H = movement_entropy(landscape, [])
        
        assert H == 0 or H < 0.1, "No corridors should give minimal entropy"


class TestConnectivityConstraint:
    """Test connectivity constraint penalty."""
    
    def test_non_negative(self):
        """Invariant: Connectivity penalty must be non-negative."""
        landscape = SyntheticLandscape(n_patches=10, random_seed=42)
        network = build_dendritic_network(landscape)
        
        C = connectivity_constraint(landscape, network.edges)
        
        assert C >= 0, "Connectivity penalty must be non-negative"
    
    def test_zero_for_connected(self):
        """Property: Connected network → minimal penalty."""
        landscape = SyntheticLandscape(n_patches=10, random_seed=42)
        network = build_dendritic_network(landscape)
        
        C = connectivity_constraint(landscape, network.edges)
        
        # Dendritic network is connected, should have low penalty
        assert C < 1.0, "Connected network should have low connectivity penalty"
    
    def test_high_for_disconnected(self):
        """Property: Disconnected network → high penalty."""
        landscape = SyntheticLandscape(n_patches=10, random_seed=42)
        
        # Single edge - creates disconnected network
        edges = [(landscape.patches[0].id, landscape.patches[1].id)]
        
        C = connectivity_constraint(landscape, edges)
        
        assert C > 1.0, "Disconnected network should have high connectivity penalty"


class TestForestTopologyPenalty:
    """Test forest topology penalty."""
    
    def test_non_negative(self):
        """Invariant: Forest penalty must be non-negative."""
        landscape = SyntheticLandscape(n_patches=10, random_seed=42)
        network = build_dendritic_network(landscape)
        
        F = forest_topology_penalty(landscape, network.edges)
        
        assert F >= 0, "Forest penalty must be non-negative"
    
    def test_zero_for_tree(self):
        """Property: Tree structure → zero penalty."""
        landscape = SyntheticLandscape(n_patches=10, random_seed=42)
        network = build_dendritic_network(landscape)
        
        F = forest_topology_penalty(landscape, network.edges)
        
        assert F == 0, "Tree structure should have zero forest penalty"
    
    def test_positive_for_cycles(self):
        """Property: Cycles → positive penalty."""
        landscape = SyntheticLandscape(n_patches=5, random_seed=42)
        
        # Create a cycle: connect all patches in a ring
        edges = [
            (landscape.patches[i].id, landscape.patches[(i+1) % 5].id)
            for i in range(5)
        ]
        
        F = forest_topology_penalty(landscape, edges)
        
        assert F > 0, "Cycle should have positive forest penalty"


class TestDisturbanceResponsePenalty:
    """Test disturbance response penalty."""
    
    def test_non_negative(self):
        """Invariant: Disturbance penalty must be non-negative."""
        landscape = SyntheticLandscape(n_patches=10, random_seed=42)
        network = build_dendritic_network(landscape)
        
        D = disturbance_response_penalty(landscape, network.edges)
        
        assert D >= 0, "Disturbance penalty must be non-negative"


class TestTotalEntropy:
    """Test total entropy calculation."""
    
    def test_non_negative(self):
        """Invariant: Total entropy must be non-negative."""
        landscape = SyntheticLandscape(n_patches=10, random_seed=42)
        network = build_dendritic_network(landscape)
        
        H_total, components = calculate_entropy(landscape, network.edges)
        
        assert H_total >= 0, "Total entropy must be non-negative"
        assert all(v >= 0 for v in components.values()), "All components must be non-negative"
    
    def test_component_sum(self):
        """Property: Total entropy equals sum of weighted components."""
        landscape = SyntheticLandscape(n_patches=10, random_seed=42)
        network = build_dendritic_network(landscape)
        
        lambda1, lambda2, lambda3 = 1.0, 1.0, 1.0
        
        H_total, components = calculate_entropy(
            landscape,
            network.edges,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3
        )
        
        expected = (
            components['H_mov'] +
            lambda1 * components['C'] +
            lambda2 * components['F'] +
            lambda3 * components['D']
        )
        
        assert np.isclose(H_total, expected), "Total entropy must equal component sum"
    
    def test_dendritic_minimizes_entropy(self):
        """Property: Dendritic network should minimize entropy vs alternatives."""
        landscape = SyntheticLandscape(n_patches=10, random_seed=42)
        
        # Dendritic network
        dendritic = build_dendritic_network(landscape)
        H_dendritic, _ = calculate_entropy(landscape, dendritic.edges)
        
        # Alternative: Random spanning tree
        import networkx as nx
        random_tree = nx.random_spanning_tree(landscape.graph)
        random_edges = list(random_tree.edges())
        H_random, _ = calculate_entropy(landscape, random_edges)
        
        # Dendritic should be at least as good
        assert H_dendritic <= H_random + 0.1, "Dendritic should minimize or tie with random tree"


class TestLandscapeProperties:
    """Test landscape representation invariants."""
    
    def test_distance_matrix_symmetric(self):
        """Invariant: Distance matrix must be symmetric."""
        landscape = SyntheticLandscape(n_patches=10, random_seed=42)
        D = landscape.distance_matrix()
        
        assert np.allclose(D, D.T), "Distance matrix must be symmetric"
    
    def test_distance_matrix_positive(self):
        """Invariant: Off-diagonal distances must be positive."""
        landscape = SyntheticLandscape(n_patches=10, random_seed=42)
        D = landscape.distance_matrix()
        
        n = len(D)
        mask = ~np.eye(n, dtype=bool)
        
        assert np.all(D[mask] > 0), "Off-diagonal distances must be positive"
    
    def test_adjacency_matrix_symmetric(self):
        """Invariant: Adjacency matrix must be symmetric."""
        landscape = SyntheticLandscape(n_patches=10, random_seed=42)
        network = build_dendritic_network(landscape)
        
        A = landscape.adjacency_matrix(network.edges)
        
        assert np.allclose(A, A.T), "Adjacency matrix must be symmetric"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
