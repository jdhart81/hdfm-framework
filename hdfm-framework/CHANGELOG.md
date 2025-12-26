# Changelog

All notable changes to the HDFM Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2025-12-09

### Added

#### Backwards Optimization Enhancement
- **Complete width scheduling integration** in `BackwardsOptimizer`
  - `optimize_widths` parameter to enable/disable width optimization at each time step
  - `width_schedule` output providing optimal corridor widths for each year in the scenario
  - Species-specific width optimization integrated with climate trajectory
- **Enhanced temporal output**
  - `corridor_schedule` now includes edge lists for each year
  - `optimal_widths` provides present-day (first time step) width recommendations

#### Documentation
- **Jupyter notebook tutorial** (`docs/tutorials/hdfm_complete_tutorial.ipynb`)
  - Complete 10-section tutorial covering all HDFM features
  - Interactive examples for landscape creation, network construction, entropy calculation
  - Species-specific planning, width optimization, genetic viability assessment
  - Robustness analysis and climate adaptation workflows

### Changed
- `KNOWN_LIMITATIONS.md` updated to reflect 95-100% paper feature completion
- Backwards temporal optimization marked as fully complete
- Version status updated from "Research prototype" to "Research software"

### Fixed
- Improved numerical stability in width optimization edge cases

---

## [0.2.0] - 2025-11-09

### Added - Major Features

#### Genetics Module (hdfm/genetics.py)
- **Full island model implementation** for metapopulation effective population size (Nₑ)
  - Complete Nₑ(A,w) calculation: `Nₑ = [∑ᵢ nᵢ]² / [∑ᵢ nᵢ² + ∑ᵢ ∑ⱼ≠ᵢ 2nᵢnⱼ(1 - Fᵢⱼ)]`
  - Co-ancestry coefficient calculation: `Fᵢⱼ(A,w)`
  - Width-dependent migration rates
  - Species-specific genetic thresholds
- **Genetic viability assessment**
  - `check_genetic_viability()` function implementing 50/500 rule
  - Species-specific Ne thresholds (350-1200 depending on guild)
  - Genetic viability status reporting
- **Population dynamics tracking**
  - `calculate_inbreeding_coefficient()` - F(t) over generations
  - `calculate_genetic_diversity_loss()` - Heterozygosity retention
  - Integration with corridor width and network topology

#### Robustness Module (hdfm/robustness.py)
- **Network robustness analysis**
  - Two-edge connectivity calculation (ρ₂)
  - Catastrophic failure probability (P_fail)
  - Edge redundancy scoring
  - Bridge detection and protection
- **Strategic loop addition**
  - Multiple algorithms: betweenness, shortest, bridge_protection, random
  - Configurable loop budgets
  - Entropy-aware loop selection
- **Pareto frontier analysis**
  - Entropy vs. robustness tradeoff exploration
  - Pareto-optimal network identification
  - Multi-objective optimization support
- **RobustnessMetrics dataclass** for comprehensive robustness reporting

#### Width Optimization
- **WidthOptimizer class** for corridor width allocation
  - Budget-constrained optimization (20-30% landscape allocation)
  - Species-specific width requirements
  - Sequential quadratic programming (SLSQP) solver
- **Allocation constraint checking**
  - `check_allocation_constraint()` function
  - Validates: `∑ᵢⱼ dᵢⱼ wᵢⱼ ≤ β ∑ᵢ Aᵢ`
  - Configurable landscape allocation budgets (β)

#### Dual Entropy Formulations
- **Entropy rate calculation** with stationary distribution
  - `calculate_entropy_rate()` function
  - Stationary distribution: πᵢ = (Aᵢ qᵢ) / (∑ₖ Aₖ qₖ)
  - Better for heterogeneous landscapes
- **Width-dependent entropy**
  - Movement success function: φ(w) = 1 − exp(−γ(w − w_min))
  - Integration throughout entropy calculations
  - Species-specific width sensitivity (γ parameter)

### Enhanced - Existing Features

#### Species Parameters
- All 4 guilds from Hart (2024) Table 2 fully integrated:
  - Small mammals (α=0.25, γ=0.080, w_crit=150m, Ne=350)
  - Medium mammals (α=0.12, γ=0.050, w_crit=220m, Ne=500)
  - Large carnivores (α=0.05, γ=0.030, w_crit=350m, Ne=750)
  - Long-lived species (α=0.03, γ=0.020, w_crit=450m, Ne=1200)
- Helper functions: `print_guild_summary()`, `list_guilds()`, `get_guild()`

#### Entropy Framework
- Width parameters integrated into all entropy components
- Corridor length penalty for movement uncertainty
- Species guild support throughout calculations
- Improved numerical stability

#### Optimization
- `BackwardsOptimizer` enhanced with climate trajectory support
- `DendriticOptimizer` maintains MST optimality guarantees
- Convergence history tracking for all optimizers

#### Public API
- 84 functions/classes exported via `__init__.py`
- Comprehensive docstrings with usage examples
- Type hints throughout codebase
- Consistent parameter naming

### Changed

#### Documentation
- **README.md**: Updated to v0.2.0, paper coverage now 90-95%
- **KNOWN_LIMITATIONS.md**:
  - Corrected implementation status (genetics and robustness now complete)
  - Updated reproduction status for all tables and figures
  - New roadmap reflecting v0.2.0 completion
- Status changed from "Research prototype" to "Research software"

#### Code Quality
- Added 88+ assertions for invariant checking
- Enhanced error messages throughout
- Improved edge case handling
- Better numerical stability in calculations

### Paper Reproducibility

#### Now Fully Reproducible
- ✅ **Table 1**: Network topology comparisons with width-dependent entropy
- ✅ **Table 2**: Species-specific validation (all 4 guilds)
- ✅ **Table 3**: Robustness-entropy tradeoffs with loop budgets
- ✅ **Figure 1**: Visual network comparisons
- ✅ **Figure 2**: Width and allocation constraint effects
- ✅ **Figure 3**: Parameter sensitivity (α, γ, corridor widths)
- ✅ **Figure 4A**: Convergence traces
- ✅ **Figure 4B**: Computational complexity scaling
- ✅ **Figure 4C-D**: Robustness analysis and Pareto frontiers

#### Partial Implementation
- ⚠️ **Backwards temporal optimization**: Climate scenarios work, width scheduling integration pending

### Technical Details

#### Performance
- O(n² log n) MST construction maintained
- Efficient robustness calculations for networks up to 100+ patches
- Monte Carlo validation supports 100+ landscape iterations

#### Dependencies
- No new runtime dependencies required
- All features use existing numpy/scipy/networkx stack
- Optional dev dependencies for testing (pytest)

### Examples

#### New Usage Patterns

**Genetic Viability Assessment:**
```python
from hdfm import calculate_effective_population_size, check_genetic_viability, SPECIES_GUILDS

corridor_widths = {edge: 150.0 for edge in network.edges}
guild = SPECIES_GUILDS['medium_mammals']

Ne, components = calculate_effective_population_size(
    landscape, network.edges, corridor_widths, species_guild=guild
)

viable, threshold, message = check_genetic_viability(Ne, guild)
print(message)  # "VIABLE: Ne=542.3 exceeds threshold 500.0 by 42.3 (8.5%)"
```

**Robustness Analysis:**
```python
from hdfm import calculate_robustness_metrics, add_strategic_loops

# Analyze base MST
metrics = calculate_robustness_metrics(landscape, network.edges)
print(f"ρ₂ = {metrics.two_edge_connectivity:.3f}")
print(f"P_fail = {metrics.failure_probability:.3f}")

# Add 5 strategic loops
robust_edges = add_strategic_loops(
    landscape, network.edges, n_loops=5, criterion='betweenness'
)

# Re-analyze
new_metrics = calculate_robustness_metrics(landscape, robust_edges)
print(f"Improved ρ₂ = {new_metrics.two_edge_connectivity:.3f}")
```

**Width Optimization:**
```python
from hdfm import WidthOptimizer, SPECIES_GUILDS

optimizer = WidthOptimizer(
    landscape=landscape,
    edges=network.edges,
    species_guild=SPECIES_GUILDS['large_carnivores'],
    beta=0.25  # 25% landscape allocation budget
)

result = optimizer.optimize()
print(f"Optimal widths: {result.corridor_widths}")
print(f"Total allocation: {result.total_allocation:.1f}%")
```

### Breaking Changes
- None - all changes are backwards compatible

### Deprecated
- None

### Removed
- None

### Fixed
- Numerical stability in entropy calculations for edge cases
- Distance matrix symmetry enforcement
- Invariant validation in all constructors
- Edge case handling for isolated patches

### Security
- No security issues identified
- All assertions validate user inputs
- No external data sources or network calls

---

## [0.1.0] - 2024-11-01

### Added
- Initial release with core HDFM framework
- Basic entropy framework (H_mov + C + F + D)
- Dendritic network construction via MST
- Alternative topology comparisons (Gabriel, Delaunay, k-NN, threshold)
- Monte Carlo validation framework
- Synthetic landscape generation
- Basic visualization tools
- Species guild definitions
- Backwards optimization structure
- Example scripts and documentation

### Technical
- Python 3.8+ support
- Core dependencies: numpy, scipy, networkx, matplotlib, pandas
- Professional package structure with setup.py
- MIT license

---

## Release Notes

### v0.2.0 Summary

This release represents a **major milestone** for the HDFM framework, bringing paper coverage from ~75% to **90-95% complete**. The two most significant additions are:

1. **Complete genetic population dynamics** with the full island model, enabling rigorous assessment of genetic viability
2. **Comprehensive robustness analysis** with strategic loops, failure probability, and Pareto frontier exploration

Nearly all tables and figures from Hart (2024) can now be reproduced. The framework is ready for scientific use in landscape ecology, conservation biology, and corridor design research.

### What's Next (v0.3.0)

- ✅ Complete backwards optimization with width scheduling integration (done in v0.2.1)
- ✅ Jupyter notebook tutorials (started in v0.2.1)
- Real-world case study examples
- Performance optimization for very large landscapes (100+ patches)
- Additional documentation and best practices guides

---

**For questions or issues, please open a GitHub issue or contact viridisnorthllc@gmail.com**
