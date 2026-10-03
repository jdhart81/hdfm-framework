"""Regression tests for width allocation units, solver status and returned widths.

Invariants checked (DFM Build Spec I7, I10):
- Patch areas (hectares) are converted to m^2 before comparison with corridor
  areas (length m x width m).
- An infeasible budget or failed solve is reported (success=False) and never
  returned as a usable set of widths.
- A successful solve returns its widths, and they satisfy bounds and budget.
"""

import pytest

from hdfm.landscape import Landscape, Patch
from hdfm.network import build_dendritic_network
from hdfm.optimization import (
    M2_PER_HECTARE,
    WidthOptimizer,
    check_allocation_constraint,
    corridor_area_budget_m2,
)
from hdfm.species import SpeciesGuild


def _guild(w_min=50.0, w_crit=150.0):
    return SpeciesGuild(name="Test guild", alpha=0.5, gamma=0.02,
                        w_min=w_min, w_crit=w_crit, Ne_threshold=500.0)


def _three_patch_landscape(area_ha):
    # Patches 1 km apart in a line: the MST is two 1,000 m corridors.
    patches = [
        Patch(id=0, x=0.0, y=0.0, area=area_ha, quality=0.9),
        Patch(id=1, x=1000.0, y=0.0, area=area_ha, quality=0.9),
        Patch(id=2, x=2000.0, y=0.0, area=area_ha, quality=0.9),
    ]
    return Landscape(patches)


def test_budget_converts_hectares_to_square_meters():
    landscape = _three_patch_landscape(area_ha=10.0)  # 30 ha total
    assert corridor_area_budget_m2(landscape, beta=0.25) == pytest.approx(0.25 * 30 * M2_PER_HECTARE)


def test_allocation_check_uses_square_meters_by_hand_calculation():
    landscape = _three_patch_landscape(area_ha=10.0)  # 300,000 m^2 of patches
    edges = build_dendritic_network(landscape).edges
    widths = {edge: 30.0 for edge in edges}  # 2 x 1,000 m x 30 m = 60,000 m^2
    ok, used, total = check_allocation_constraint(landscape, edges, widths, beta=0.25)
    assert used == pytest.approx(60_000.0)
    assert total == pytest.approx(300_000.0)
    assert ok  # 60,000 <= 75,000
    widths = {edge: 40.0 for edge in edges}  # 80,000 m^2 > 75,000
    assert not check_allocation_constraint(landscape, edges, widths, beta=0.25)[0]


def test_infeasible_budget_is_reported_not_returned_as_widths():
    # 3 ha total -> 7,500 m^2 budget; minimum need 2 x 1,000 m x 50 m = 100,000 m^2.
    landscape = _three_patch_landscape(area_ha=1.0)
    edges = build_dendritic_network(landscape).edges
    result = WidthOptimizer(landscape, edges, _guild(), beta=0.25).optimize(max_iterations=30)
    assert result.success is False
    assert result.optimal_widths is None
    assert "Infeasible" in result.message


def test_feasible_solve_returns_widths_within_bounds_and_budget():
    # 3 x 100 ha -> 750,000 m^2 budget; widths up to 375 m fit.
    landscape = _three_patch_landscape(area_ha=100.0)
    edges = build_dendritic_network(landscape).edges
    guild = _guild()
    result = WidthOptimizer(landscape, edges, guild, beta=0.25).optimize(max_iterations=60)
    assert result.success, result.message
    assert result.optimal_widths is not None
    assert set(result.optimal_widths) == set(edges)
    for width in result.optimal_widths.values():
        assert guild.w_min - 1e-6 <= width <= 500.0 + 1e-6
    ok, used, _ = check_allocation_constraint(landscape, edges, result.optimal_widths, beta=0.25)
    assert ok, used
