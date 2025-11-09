"""
Tests for robustness analysis with looped topologies and redundancy scoring.
"""

import pytest
import numpy as np
from hdfm import (
    Landscape,
    SyntheticLandscape,
    Patch,
    build_dendritic_network,
    calculate_two_edge_connectivity,
    calculate_edge_redundancy_scores,
    calculate_failure_probability,
    add_strategic_loops,
    calculate_robustness_metrics,
    pareto_frontier_analysis,
    RobustnessMetrics
)


class TestTwoEdgeConnectivity:
    """Test 2-edge-connectivity calculations."""

    def setup_method(self):
        """Create test landscape."""
        patches = [
            Patch(id=0, x=0, y=0, area=50.0),
            Patch(id=1, x=1000, y=0, area=50.0),
            Patch(id=2, x=2000, y=0, area=50.0),
            Patch(id=3, x=1000, y=1000, area=50.0),
        ]
        self.landscape = Landscape(patches)

    def test_two_edge_connectivity_mst(self):
        """Test ρ₂ = 0 for MST (no redundancy)."""
        # Build MST (tree structure, no loops)
        network = build_dendritic_network(self.landscape)

        rho_2, components = calculate_two_edge_connectivity(
            self.landscape,
            network.edges
        )

        assert rho_2 == 0.0  # MST has no redundancy
        assert components['two_connected_pairs'] == 0
        assert components['n_bridges'] > 0

    def test_two_edge_connectivity_with_loop(self):
        """Test ρ₂ > 0 when loop added."""
        # Build MST
        network = build_dendritic_network(self.landscape)

        # Add one loop
        edges_with_loop = add_strategic_loops(
            self.landscape,
            network.edges,
            n_loops=1,
            criterion='shortest'
        )

        rho_2, components = calculate_two_edge_connectivity(
            self.landscape,
            edges_with_loop
        )

        # Should have some redundancy now
        assert rho_2 > 0.0
        assert components['two_connected_pairs'] > 0

    def test_two_edge_connectivity_complete_graph(self):
        """Test ρ₂ for highly connected graph."""
        # Create complete graph (all pairs connected)
        edges = []
        for i, patch_i in enumerate(self.landscape.patches):
            for patch_j in self.landscape.patches[i+1:]:
                edges.append((patch_i.id, patch_j.id))

        rho_2, components = calculate_two_edge_connectivity(
            self.landscape,
            edges
        )

        # Complete graph should have high ρ₂
        assert rho_2 > 0.5
        assert components['n_bridges'] == 0  # No bridges in complete graph


class TestEdgeRedundancyScores:
    """Test edge redundancy scoring."""

    def setup_method(self):
        """Create test landscape."""
        patches = [
            Patch(id=0, x=0, y=0, area=50.0),
            Patch(id=1, x=1000, y=0, area=50.0),
            Patch(id=2, x=2000, y=0, area=50.0),
            Patch(id=3, x=1000, y=1000, area=50.0),
        ]
        self.landscape = Landscape(patches)

    def test_redundancy_scores_mst(self):
        """Test redundancy scores for MST (all bridges)."""
        network = build_dendritic_network(self.landscape)

        scores = calculate_edge_redundancy_scores(
            self.landscape,
            network.edges
        )

        # All edges in MST are bridges (score = 1.0)
        for edge, score in scores.items():
            assert score == 1.0

    def test_redundancy_scores_with_loop(self):
        """Test redundancy scores with loop added."""
        network = build_dendritic_network(self.landscape)

        # Add loop
        edges_with_loop = add_strategic_loops(
            self.landscape,
            network.edges,
            n_loops=2
        )

        scores = calculate_edge_redundancy_scores(
            self.landscape,
            edges_with_loop
        )

        # Should have some edges with score < 1.0 (redundant)
        min_score = min(scores.values())
        assert min_score < 1.0

        # All scores should be in [0, 1]
        for score in scores.values():
            assert 0 <= score <= 1


class TestFailureProbability:
    """Test catastrophic failure probability calculations."""

    def setup_method(self):
        """Create test landscape."""
        self.landscape = SyntheticLandscape(n_patches=10, random_seed=42)

    def test_failure_probability_mst(self):
        """Test P_fail for MST (high vulnerability)."""
        network = build_dendritic_network(self.landscape)

        P_fail, components = calculate_failure_probability(
            self.landscape,
            network.edges,
            k=1
        )

        # MST should have high failure probability
        assert 0 <= P_fail <= 1
        assert components['n_simulations'] == 1000
        assert components['k_failures'] == 1

    def test_failure_probability_with_loops(self):
        """Test P_fail decreases with loops."""
        network = build_dendritic_network(self.landscape)

        # MST failure probability
        P_fail_mst, _ = calculate_failure_probability(
            self.landscape,
            network.edges,
            k=1
        )

        # Add loops
        edges_with_loops = add_strategic_loops(
            self.landscape,
            network.edges,
            n_loops=5
        )

        P_fail_loops, _ = calculate_failure_probability(
            self.landscape,
            edges_with_loops,
            k=1
        )

        # Failure probability should decrease with redundancy
        assert P_fail_loops < P_fail_mst

    def test_failure_probability_increases_with_k(self):
        """Test P_fail increases with number of failures."""
        network = build_dendritic_network(self.landscape)

        edges_with_loops = add_strategic_loops(
            self.landscape,
            network.edges,
            n_loops=3
        )

        P_fail_1, _ = calculate_failure_probability(
            self.landscape,
            edges_with_loops,
            k=1
        )

        P_fail_2, _ = calculate_failure_probability(
            self.landscape,
            edges_with_loops,
            k=2
        )

        # More failures should increase probability
        assert P_fail_2 >= P_fail_1


class TestStrategicLoops:
    """Test strategic loop addition algorithms."""

    def setup_method(self):
        """Create test landscape."""
        self.landscape = SyntheticLandscape(n_patches=10, random_seed=42)
        network = build_dendritic_network(self.landscape)
        self.mst_edges = network.edges

    def test_add_strategic_loops_shortest(self):
        """Test adding loops with shortest criterion."""
        edges = add_strategic_loops(
            self.landscape,
            self.mst_edges,
            n_loops=3,
            criterion='shortest'
        )

        # Should have MST edges + 3 loops
        n_patches = self.landscape.n_patches
        assert len(edges) == (n_patches - 1) + 3

    def test_add_strategic_loops_betweenness(self):
        """Test adding loops with betweenness criterion."""
        edges = add_strategic_loops(
            self.landscape,
            self.mst_edges,
            n_loops=3,
            criterion='betweenness'
        )

        n_patches = self.landscape.n_patches
        assert len(edges) == (n_patches - 1) + 3

    def test_add_strategic_loops_bridge_protection(self):
        """Test adding loops with bridge protection criterion."""
        edges = add_strategic_loops(
            self.landscape,
            self.mst_edges,
            n_loops=3,
            criterion='bridge_protection'
        )

        n_patches = self.landscape.n_patches
        assert len(edges) == (n_patches - 1) + 3

    def test_add_strategic_loops_random(self):
        """Test adding loops with random criterion."""
        edges = add_strategic_loops(
            self.landscape,
            self.mst_edges,
            n_loops=3,
            criterion='random'
        )

        n_patches = self.landscape.n_patches
        assert len(edges) == (n_patches - 1) + 3

    def test_add_zero_loops(self):
        """Test adding zero loops returns MST."""
        edges = add_strategic_loops(
            self.landscape,
            self.mst_edges,
            n_loops=0
        )

        assert edges == self.mst_edges


class TestRobustnessMetrics:
    """Test comprehensive robustness metrics calculation."""

    def setup_method(self):
        """Create test landscape."""
        self.landscape = SyntheticLandscape(n_patches=10, random_seed=42)

    def test_robustness_metrics_mst(self):
        """Test robustness metrics for MST."""
        network = build_dendritic_network(self.landscape)

        metrics = calculate_robustness_metrics(
            self.landscape,
            network.edges,
            k_failures=1
        )

        assert isinstance(metrics, RobustnessMetrics)
        assert metrics.two_edge_connectivity == 0.0  # MST has no redundancy
        assert metrics.n_loops == 0
        assert metrics.failure_probability > 0
        assert 0 <= metrics.redundancy_score <= 1

    def test_robustness_metrics_with_loops(self):
        """Test robustness metrics improve with loops."""
        network = build_dendritic_network(self.landscape)

        # MST metrics
        metrics_mst = calculate_robustness_metrics(
            self.landscape,
            network.edges
        )

        # Add loops
        edges_with_loops = add_strategic_loops(
            self.landscape,
            network.edges,
            n_loops=5
        )

        metrics_loops = calculate_robustness_metrics(
            self.landscape,
            edges_with_loops
        )

        # Robustness should improve
        assert metrics_loops.two_edge_connectivity > metrics_mst.two_edge_connectivity
        assert metrics_loops.n_loops == 5
        assert metrics_loops.failure_probability < metrics_mst.failure_probability
        assert metrics_loops.redundancy_score > metrics_mst.redundancy_score

    def test_robustness_metrics_edge_redundancy(self):
        """Test edge redundancy scores in metrics."""
        network = build_dendritic_network(self.landscape)

        metrics = calculate_robustness_metrics(
            self.landscape,
            network.edges
        )

        # Should have redundancy score for each edge
        assert len(metrics.edge_redundancy) == len(network.edges)

        for edge, score in metrics.edge_redundancy.items():
            assert 0 <= score <= 1


class TestParetoFrontier:
    """Test Pareto frontier analysis."""

    def setup_method(self):
        """Create test landscape."""
        self.landscape = SyntheticLandscape(n_patches=10, random_seed=42)

    def test_pareto_frontier_basic(self):
        """Test basic Pareto frontier analysis."""
        results = pareto_frontier_analysis(
            self.landscape,
            max_loops=5
        )

        # Check structure
        assert 'n_loops' in results
        assert 'entropy' in results
        assert 'robustness' in results
        assert 'failure_prob' in results
        assert 'pareto_points' in results
        assert 'networks' in results

        # Check sizes
        assert len(results['n_loops']) == 6  # 0 to 5 loops
        assert len(results['entropy']) == 6
        assert len(results['robustness']) == 6
        assert len(results['networks']) == 6

    def test_pareto_frontier_tradeoff(self):
        """Test entropy-robustness tradeoff."""
        results = pareto_frontier_analysis(
            self.landscape,
            max_loops=8
        )

        # Robustness should generally increase with loops
        robustness = results['robustness']
        assert robustness[-1] > robustness[0]  # More loops = more robustness

        # Entropy may increase or stay similar
        entropy = results['entropy']
        assert len(entropy) == 9

    def test_pareto_frontier_different_criteria(self):
        """Test Pareto frontier with different loop criteria."""
        results_shortest = pareto_frontier_analysis(
            self.landscape,
            max_loops=5,
            loop_criterion='shortest'
        )

        results_betweenness = pareto_frontier_analysis(
            self.landscape,
            max_loops=5,
            loop_criterion='betweenness'
        )

        # Both should produce valid results
        assert len(results_shortest['n_loops']) == 6
        assert len(results_betweenness['n_loops']) == 6

        # Results may differ
        # (different strategies produce different networks)

    def test_pareto_points_identification(self):
        """Test identification of Pareto-optimal points."""
        results = pareto_frontier_analysis(
            self.landscape,
            max_loops=5
        )

        pareto_points = results['pareto_points']

        # Should have at least some Pareto points
        assert len(pareto_points) > 0

        # Each point should be (entropy, robustness, n_loops) tuple
        for point in pareto_points:
            entropy, robustness, n_loops = point
            assert entropy > 0
            assert 0 <= robustness <= 1
            assert 0 <= n_loops <= 5


class TestIntegration:
    """Integration tests for robustness module."""

    def test_full_robustness_workflow(self):
        """Test complete robustness analysis workflow."""
        # Create landscape
        landscape = SyntheticLandscape(n_patches=15, random_seed=42)

        # Build MST
        network = build_dendritic_network(landscape)

        print(f"\n{'='*60}")
        print(f"ROBUSTNESS ANALYSIS")
        print(f"{'='*60}")

        # Analyze MST
        print(f"\n1. MST (Dendritic Network):")
        metrics_mst = calculate_robustness_metrics(landscape, network.edges)
        print(f"   ρ₂ = {metrics_mst.two_edge_connectivity:.3f}")
        print(f"   P_fail(k=1) = {metrics_mst.failure_probability:.3f}")
        print(f"   Loops = {metrics_mst.n_loops}")
        print(f"   Redundancy = {metrics_mst.redundancy_score:.3f}")

        # Add 3 strategic loops
        print(f"\n2. MST + 3 Strategic Loops:")
        edges_3_loops = add_strategic_loops(
            landscape,
            network.edges,
            n_loops=3,
            criterion='betweenness'
        )
        metrics_3 = calculate_robustness_metrics(landscape, edges_3_loops)
        print(f"   ρ₂ = {metrics_3.two_edge_connectivity:.3f}")
        print(f"   P_fail(k=1) = {metrics_3.failure_probability:.3f}")
        print(f"   Loops = {metrics_3.n_loops}")
        print(f"   Redundancy = {metrics_3.redundancy_score:.3f}")

        # Add 6 strategic loops
        print(f"\n3. MST + 6 Strategic Loops:")
        edges_6_loops = add_strategic_loops(
            landscape,
            network.edges,
            n_loops=6,
            criterion='betweenness'
        )
        metrics_6 = calculate_robustness_metrics(landscape, edges_6_loops)
        print(f"   ρ₂ = {metrics_6.two_edge_connectivity:.3f}")
        print(f"   P_fail(k=1) = {metrics_6.failure_probability:.3f}")
        print(f"   Loops = {metrics_6.n_loops}")
        print(f"   Redundancy = {metrics_6.redundancy_score:.3f}")

        # Pareto frontier
        print(f"\n4. Pareto Frontier Analysis:")
        results = pareto_frontier_analysis(landscape, max_loops=10)
        print(f"   Explored {len(results['n_loops'])} configurations")
        print(f"   Pareto-optimal points: {len(results['pareto_points'])}")

        # Assertions
        assert metrics_3.two_edge_connectivity > metrics_mst.two_edge_connectivity
        assert metrics_6.two_edge_connectivity > metrics_3.two_edge_connectivity
        assert metrics_6.failure_probability < metrics_mst.failure_probability
        assert len(results['pareto_points']) > 0

        print(f"\n{'='*60}")
        print(f"ROBUSTNESS IMPROVEMENT:")
        rho_improvement = (metrics_6.two_edge_connectivity -
                          metrics_mst.two_edge_connectivity)
        fail_reduction = (metrics_mst.failure_probability -
                         metrics_6.failure_probability)
        print(f"   Δρ₂ = +{rho_improvement:.3f}")
        print(f"   ΔP_fail = -{fail_reduction:.3f}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
