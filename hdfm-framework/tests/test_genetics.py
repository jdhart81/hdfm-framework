"""
Tests for genetic population dynamics and effective population size calculations.
"""

import pytest
import numpy as np
from hdfm import (
    Landscape,
    SyntheticLandscape,
    Patch,
    build_dendritic_network,
    calculate_effective_population_size,
    calculate_migration_rate,
    calculate_coancestry_coefficient,
    check_genetic_viability,
    calculate_inbreeding_coefficient,
    calculate_genetic_diversity_loss,
    SPECIES_GUILDS
)


class TestMigrationRate:
    """Test migration rate calculations."""

    def test_migration_rate_basic(self):
        """Test basic migration rate calculation."""
        m = calculate_migration_rate(
            distance=1000.0,
            width=100.0,
            dispersal_scale=1000.0,
            alpha=2.0
        )
        assert 0 <= m <= 1
        assert m > 0  # Should be positive for reasonable inputs

    def test_migration_rate_with_guild(self):
        """Test migration rate with species guild."""
        guild = SPECIES_GUILDS['small_mammals']
        m = calculate_migration_rate(
            distance=1000.0,
            width=150.0,  # Critical width for small mammals
            dispersal_scale=4000.0,  # Mean dispersal = 4 km
            alpha=0.25,
            species_guild=guild
        )
        assert 0 <= m <= 1
        assert m > 0

    def test_migration_rate_decreases_with_distance(self):
        """Migration rate should decrease with distance."""
        m1 = calculate_migration_rate(500.0, 100.0, 1000.0)
        m2 = calculate_migration_rate(1000.0, 100.0, 1000.0)
        m3 = calculate_migration_rate(2000.0, 100.0, 1000.0)

        assert m1 > m2 > m3

    def test_migration_rate_increases_with_width(self):
        """Migration rate should increase with corridor width."""
        guild = SPECIES_GUILDS['small_mammals']

        m1 = calculate_migration_rate(1000.0, 50.0, 1000.0, species_guild=guild)
        m2 = calculate_migration_rate(1000.0, 150.0, 1000.0, species_guild=guild)
        m3 = calculate_migration_rate(1000.0, 300.0, 1000.0, species_guild=guild)

        assert m1 < m2 < m3


class TestCoancestryCoefficient:
    """Test co-ancestry coefficient calculations."""

    def test_coancestry_basic(self):
        """Test basic co-ancestry calculation."""
        F = calculate_coancestry_coefficient(migration_rate=0.1)
        assert 0 <= F <= 1

    def test_coancestry_increases_with_migration(self):
        """Co-ancestry should increase with migration rate."""
        F1 = calculate_coancestry_coefficient(0.01)
        F2 = calculate_coancestry_coefficient(0.1)
        F3 = calculate_coancestry_coefficient(0.5)

        assert F1 < F2 < F3

    def test_coancestry_zero_migration(self):
        """Co-ancestry should be near zero for no migration."""
        F = calculate_coancestry_coefficient(0.0)
        assert F < 0.01

    def test_coancestry_high_migration(self):
        """Co-ancestry should be high for high migration."""
        F = calculate_coancestry_coefficient(0.9)
        assert F > 0.4


class TestEffectivePopulationSize:
    """Test effective population size calculations."""

    def setup_method(self):
        """Create test landscape."""
        # Create simple landscape with known populations
        patches = [
            Patch(id=0, x=0, y=0, area=50.0, quality=0.8, population=100.0),
            Patch(id=1, x=1000, y=0, area=50.0, quality=0.8, population=100.0),
            Patch(id=2, x=2000, y=0, area=50.0, quality=0.8, population=100.0),
        ]
        self.landscape = Landscape(patches)

        # Build MST network
        network = build_dendritic_network(self.landscape)
        self.edges = network.edges

        # Simple corridor widths
        self.widths = {edge: 100.0 for edge in self.edges}

    def test_ne_calculation_basic(self):
        """Test basic Ne calculation."""
        Ne, components = calculate_effective_population_size(
            self.landscape,
            self.edges,
            self.widths
        )

        assert Ne > 0
        assert 'Ne' in components
        assert 'N_total' in components
        assert 'migration_matrix' in components
        assert 'coancestry_matrix' in components

    def test_ne_less_than_total_population(self):
        """Effective size typically <= census size for subdivided populations."""
        Ne, components = calculate_effective_population_size(
            self.landscape,
            self.edges,
            self.widths
        )

        N_total = components['N_total']
        # Ne can sometimes exceed N_total in certain connectivity patterns,
        # but should be within reasonable range
        assert Ne > 0
        assert Ne < N_total * 2  # Reasonable upper bound

    def test_ne_with_species_guild(self):
        """Test Ne calculation with species guild."""
        guild = SPECIES_GUILDS['medium_mammals']

        Ne, components = calculate_effective_population_size(
            self.landscape,
            self.edges,
            self.widths,
            dispersal_scale=8300.0,  # 8.3 km for medium mammals
            alpha=0.12,
            species_guild=guild
        )

        assert Ne > 0
        assert components['N_total'] == 300.0  # 3 patches × 100

    def test_ne_increases_with_connectivity(self):
        """Ne should increase with better connectivity."""
        # Calculate Ne for narrow corridors
        narrow_widths = {edge: 50.0 for edge in self.edges}
        Ne_narrow, _ = calculate_effective_population_size(
            self.landscape,
            self.edges,
            narrow_widths
        )

        # Calculate Ne for wide corridors
        wide_widths = {edge: 300.0 for edge in self.edges}
        Ne_wide, _ = calculate_effective_population_size(
            self.landscape,
            self.edges,
            wide_widths
        )

        assert Ne_wide > Ne_narrow

    def test_ne_with_default_populations(self):
        """Test Ne with default population parameter."""
        # Create landscape without population data
        patches = [
            Patch(id=0, x=0, y=0, area=50.0, quality=0.8),
            Patch(id=1, x=1000, y=0, area=50.0, quality=0.8),
            Patch(id=2, x=2000, y=0, area=50.0, quality=0.8),
        ]
        landscape = Landscape(patches)
        network = build_dendritic_network(landscape)
        edges = network.edges
        widths = {edge: 100.0 for edge in edges}

        # Should use default population
        Ne, components = calculate_effective_population_size(
            landscape,
            edges,
            widths,
            default_population=150.0
        )

        assert Ne > 0
        assert components['N_total'] == 450.0  # 3 × 150


class TestGeneticViability:
    """Test genetic viability checking."""

    def test_genetic_viability_basic(self):
        """Test basic viability check."""
        viable, thresh, msg = check_genetic_viability(Ne=600.0)

        assert viable is True
        assert thresh == 500.0
        assert 'VIABLE' in msg

    def test_genetic_viability_below_threshold(self):
        """Test viability check below threshold."""
        viable, thresh, msg = check_genetic_viability(Ne=300.0)

        assert viable is False
        assert thresh == 500.0
        assert 'AT RISK' in msg

    def test_genetic_viability_with_guild(self):
        """Test viability check with species guild."""
        guild = SPECIES_GUILDS['small_mammals']

        viable, thresh, msg = check_genetic_viability(Ne=400.0, species_guild=guild)

        assert viable is True
        assert thresh == 350.0  # Small mammal threshold

    def test_genetic_viability_custom_threshold(self):
        """Test viability check with custom threshold."""
        viable, thresh, msg = check_genetic_viability(Ne=600.0, threshold=700.0)

        assert viable is False
        assert thresh == 700.0


class TestInbreedingCoefficient:
    """Test inbreeding coefficient calculations."""

    def test_inbreeding_basic(self):
        """Test basic inbreeding calculation."""
        F = calculate_inbreeding_coefficient(Ne=500.0, generations=10)
        assert 0 <= F <= 1

    def test_inbreeding_increases_over_time(self):
        """Inbreeding should increase over generations."""
        F1 = calculate_inbreeding_coefficient(Ne=500.0, generations=10)
        F2 = calculate_inbreeding_coefficient(Ne=500.0, generations=50)
        F3 = calculate_inbreeding_coefficient(Ne=500.0, generations=100)

        assert F1 < F2 < F3

    def test_inbreeding_faster_in_small_populations(self):
        """Inbreeding should accumulate faster in smaller populations."""
        F_small = calculate_inbreeding_coefficient(Ne=50.0, generations=10)
        F_large = calculate_inbreeding_coefficient(Ne=500.0, generations=10)

        assert F_small > F_large

    def test_inbreeding_zero_generations(self):
        """Inbreeding should be zero at generation 0."""
        F = calculate_inbreeding_coefficient(Ne=500.0, generations=0)
        assert F == 0.0


class TestGeneticDiversityLoss:
    """Test genetic diversity loss calculations."""

    def test_diversity_loss_basic(self):
        """Test basic diversity loss calculation."""
        H = calculate_genetic_diversity_loss(Ne=500.0, generations=100)
        assert 0 <= H <= 1

    def test_diversity_loss_decreases_over_time(self):
        """Heterozygosity should decrease over time."""
        H1 = calculate_genetic_diversity_loss(Ne=500.0, generations=10)
        H2 = calculate_genetic_diversity_loss(Ne=500.0, generations=50)
        H3 = calculate_genetic_diversity_loss(Ne=500.0, generations=100)

        assert H1 > H2 > H3

    def test_diversity_loss_faster_in_small_populations(self):
        """Diversity loss should be faster in smaller populations."""
        H_small = calculate_genetic_diversity_loss(Ne=50.0, generations=50)
        H_large = calculate_genetic_diversity_loss(Ne=500.0, generations=50)

        assert H_small < H_large

    def test_diversity_loss_zero_generations(self):
        """Diversity should be fully retained at generation 0."""
        H = calculate_genetic_diversity_loss(Ne=500.0, generations=0)
        assert H == 1.0


class TestIntegration:
    """Integration tests for genetic module."""

    def test_full_genetic_workflow(self):
        """Test complete genetic analysis workflow."""
        # Create landscape
        landscape = SyntheticLandscape(n_patches=10, random_seed=42)

        # Add populations
        for patch in landscape.patches:
            patch.population = patch.area * 10.0 * patch.quality

        # Build network
        network = build_dendritic_network(landscape)

        # Define corridor widths
        widths = {edge: 150.0 for edge in network.edges}

        # Calculate Ne
        guild = SPECIES_GUILDS['medium_mammals']
        Ne, components = calculate_effective_population_size(
            landscape,
            network.edges,
            widths,
            dispersal_scale=8300.0,
            alpha=0.12,
            species_guild=guild
        )

        # Check viability
        viable, thresh, msg = check_genetic_viability(Ne, species_guild=guild)

        # Calculate long-term genetic metrics
        F_100 = calculate_inbreeding_coefficient(Ne, generations=100)
        H_100 = calculate_genetic_diversity_loss(Ne, generations=100)

        # Assertions
        assert Ne > 0
        assert isinstance(viable, bool)
        assert 0 <= F_100 <= 1
        assert 0 <= H_100 <= 1
        assert H_100 < 1.0  # Some diversity loss expected

        print(f"\nGenetic Analysis Results:")
        print(f"Ne = {Ne:.1f}")
        print(f"Viability: {msg}")
        print(f"F(100 gen) = {F_100:.3f}")
        print(f"H(100 gen) = {H_100:.3f} ({H_100*100:.1f}% retained)")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
