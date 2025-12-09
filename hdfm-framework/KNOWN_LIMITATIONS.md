# Known Limitations and Implementation Status

**Last Updated:** 2025-12-09
**Version:** 0.2.1

## Overview

This document provides transparent disclosure of which features from the HDFM scientific papers are currently implemented, partially implemented, or planned for future releases. This framework is **actively under development** and represents a foundational implementation of the core mathematical concepts.

## Implementation Status Summary

| Feature Category | Status | Completion |
|-----------------|--------|------------|
| Core Dendritic Network (MST) | ✅ Complete | 100% |
| Basic Entropy Framework | ✅ Complete | 100% |
| Alternative Topology Comparisons | ✅ Complete | 100% |
| Monte Carlo Validation | ✅ Complete | 100% |
| Statistical Testing | ✅ Complete | 100% |
| Visualization Tools | ✅ Complete | 100% |
| Species-Specific Parameters | ✅ Complete | 100% |
| Width-Dependent Entropy | ✅ **NEW in v0.2.0** | 100% |
| Dual Entropy Formulations | ✅ **NEW in v0.2.0** | 100% |
| Landscape Allocation Constraints | ✅ **NEW in v0.2.0** | 100% |
| Width Optimization | ✅ **NEW in v0.2.0** | 100% |
| Full Effective Population Size (Nₑ) | ✅ **NEW in v0.2.0** | 100% |
| Robustness Analysis (Loops) | ✅ **NEW in v0.2.0** | 100% |
| Backwards Temporal Optimization | ✅ **COMPLETE in v0.2.1** | 100% |

**Overall:** ~95-100% of paper features fully implemented

---

## ✅ IMPLEMENTED FEATURES

### 1. Core Dendritic Network Construction
**Status:** ✅ **FULLY IMPLEMENTED**

- Minimum Spanning Tree (MST) via Kruskal's algorithm
- Validates dendritic structure (connected, acyclic, n-1 edges)
- Computational complexity O(n² log n)
- Correct implementation of core theorem

**Code:** `hdfm/network.py:101-142`

### 2. Basic Entropy Framework
**Status:** ✅ **FULLY IMPLEMENTED**

Implements: `H(L) = H_mov + λ₁C(L) + λ₂F(L) + λ₃D(L)`

- Movement entropy H_mov (Shannon entropy of dispersal)
- Connectivity constraint C(L) (simplified proxy)
- Forest topology penalty F(L) (dendritic structure enforcement)
- Disturbance response penalty D(L) (path length minimization)

**Limitation:** H_mov does NOT yet include width-dependent terms

**Code:** `hdfm/entropy.py`

### 3. Alternative Network Topologies
**Status:** ✅ **FULLY IMPLEMENTED**

- Gabriel graphs
- Delaunay triangulation
- k-nearest neighbors
- Threshold distance networks
- Relative neighborhood graphs

**Code:** `hdfm/network.py:145-277`

### 4. Synthetic Landscape Generation
**Status:** ✅ **FULLY IMPLEMENTED**

- Random patch placement
- Configurable area, quality distributions
- Reproducible (random seed support)
- Multiple template sizes

**Code:** `hdfm/landscape.py:220-324`

### 5. Validation Framework
**Status:** ✅ **FULLY IMPLEMENTED**

- Monte Carlo experiments (100+ landscapes)
- Wilcoxon signed-rank tests
- Cohen's d effect sizes
- Convergence analysis

**Can Reproduce:** Table 1 topology comparisons (without width effects)

**Code:** `hdfm/validation.py`

### 6. Backwards Temporal Optimization
**Status:** ✅ **FULLY IMPLEMENTED in v0.2.1**

- Climate scenario modeling ✅
- Temporal trajectory (2025→2100) ✅
- Multi-year network adjustment ✅
- Automated width scheduling ✅ Integrated with `WidthOptimizer`
- Width schedule output by year ✅
- Species-specific width optimization at each time step ✅

**Code:** `hdfm/optimization.py:163-514`

**Usage:**
```python
from hdfm import BackwardsOptimizer, ClimateScenario, SPECIES_GUILDS

# Create climate scenario
scenario = ClimateScenario(
    years=[2025, 2050, 2075, 2100],
    temperature_changes=[0.0, 1.5, 2.5, 3.5],
    precipitation_changes=[0.0, -5.0, -10.0, -15.0]
)

# Optimize with width scheduling
optimizer = BackwardsOptimizer(
    landscape=landscape,
    scenario=scenario,
    species_guild=SPECIES_GUILDS['medium_mammals'],
    beta=0.25,  # 25% landscape allocation
    optimize_widths=True
)

result = optimizer.optimize()

# Access temporal schedules
print(f"Corridor schedule: {result.corridor_schedule}")
print(f"Width schedule by year: {result.width_schedule}")
print(f"Present-day widths: {result.optimal_widths}")
```

### 7. Species-Specific Parameters
**Status:** ✅ **IMPLEMENTED**

- All 4 guilds from Table 2 (Hart 2024)
- Dispersal parameters (α)
- Width sensitivity (γ)
- Critical widths (w_crit)
- Genetic thresholds (Nₑ)

**Code:** `hdfm/species.py`

**Usage:**
```python
from hdfm import SPECIES_GUILDS, print_guild_summary

# See all available guilds
print_guild_summary()

# Use specific guild
small_mammals = SPECIES_GUILDS['small_mammals']
print(f"Critical width: {small_mammals.w_crit}m")
print(f"Movement success at 150m: {small_mammals.movement_success(150):.2%}")
```

---

### 8. Width-Dependent Entropy Calculations
**Status:** ✅ **IMPLEMENTED in v0.2.0**

Implemented features:
- Width-dependent movement success: φ(w) = 1 − exp(−γ(w − wₘᵢₙ))
- Width parameters integrated into entropy calculations
- Support for species-specific width sensitivity

**Code:** `hdfm/entropy.py:20-114`

**Usage:**
```python
from hdfm import calculate_entropy, SPECIES_GUILDS

guild = SPECIES_GUILDS['small_mammals']
corridor_widths = {(0,1): 150, (1,2): 200}  # widths in meters

H, components = calculate_entropy(
    landscape=landscape,
    edges=network.edges,
    corridor_widths=corridor_widths,
    species_guild=guild
)
```

---

### 9. Dual Entropy Formulations
**Status:** ✅ **IMPLEMENTED in v0.2.0**

Implemented features:
- Entropy rate with stationary distribution: H_rate(A,w,π)
- Stationary distribution: πᵢ = (Aᵢ qᵢ) / (∑ₖ Aₖ qₖ)
- Accounts for patch importance in heterogeneous landscapes

**Code:** `hdfm/entropy.py:389-497`

**Usage:**
```python
from hdfm import calculate_entropy_rate

H_rate, components = calculate_entropy_rate(
    landscape=landscape,
    edges=network.edges,
    corridor_widths=corridor_widths,
    species_guild=guild
)
```

---

### 10. Landscape Allocation Constraints
**Status:** ✅ **IMPLEMENTED in v0.2.0**

Implemented features:
- Allocation constraint checking: Σᵢⱼ dᵢⱼ wᵢⱼ ≤ β Σᵢ Aᵢ
- Support for 20-30% landscape allocation budgets
- Constraint enforcement in width optimization

**Code:** `hdfm/optimization.py:368-417`

**Usage:**
```python
from hdfm import check_allocation_constraint

satisfied, corridor_area, total_area = check_allocation_constraint(
    landscape=landscape,
    edges=network.edges,
    corridor_widths=corridor_widths,
    beta=0.25  # 25% allocation
)
```

---

### 11. Width Optimization Algorithms
**Status:** ✅ **IMPLEMENTED in v0.2.0**

Implemented features:
- Width optimization under landscape allocation constraints
- Sequential quadratic programming (SLSQP) optimizer
- Support for width bounds (w_min to w_max)

**Code:** `hdfm/optimization.py:420-554`

**Usage:**
```python
from hdfm import WidthOptimizer, SPECIES_GUILDS

optimizer = WidthOptimizer(
    landscape=landscape,
    edges=network.edges,
    species_guild=SPECIES_GUILDS['medium_mammals'],
    beta=0.25
)

result = optimizer.optimize()
```

---

## ✅ NEWLY IMPLEMENTED FEATURES IN v0.2.0

### 1. Full Effective Population Size Nₑ(A,w)
**Status:** ✅ **FULLY IMPLEMENTED in v0.2.0**

**Complete island model implementation:**
```
Nₑ(A,w) = [∑ᵢ nᵢ]² / [∑ᵢ nᵢ² + ∑ᵢ ∑ⱼ≠ᵢ 2nᵢnⱼ(1 - Fᵢⱼ(A,w))]
```

**Implemented Features:**
- ✅ Full island model metapopulation genetics
- ✅ Co-ancestry coefficient calculation: Fᵢⱼ(A,w)
- ✅ Width-dependent migration rates
- ✅ Genetic viability thresholds (50/500 rule)
- ✅ Inbreeding coefficient tracking
- ✅ Genetic diversity loss calculations

**Code:** `hdfm/genetics.py`

**Usage:**
```python
from hdfm import calculate_effective_population_size, check_genetic_viability, SPECIES_GUILDS

# Calculate Nₑ with width-dependent migration
corridor_widths = {edge: 150.0 for edge in network.edges}
guild = SPECIES_GUILDS['medium_mammals']

Ne, components = calculate_effective_population_size(
    landscape, network.edges, corridor_widths,
    species_guild=guild
)

# Check genetic viability
viable, threshold, message = check_genetic_viability(Ne, guild)
print(message)
```

---

### 2. Robustness Analysis with Loop Budgets
**Status:** ✅ **FULLY IMPLEMENTED in v0.2.0**

**Implemented Features:**
- ✅ MST + strategic loops construction
- ✅ 2-edge-connectivity calculation (ρ₂)
- ✅ Catastrophic failure probability (P_fail)
- ✅ Pareto frontier analysis (entropy vs. robustness)
- ✅ Edge redundancy scoring
- ✅ Multiple loop addition strategies

**Code:** `hdfm/robustness.py`

**Usage:**
```python
from hdfm import (
    calculate_robustness_metrics,
    add_strategic_loops,
    pareto_frontier_analysis
)

# Analyze MST robustness
metrics = calculate_robustness_metrics(landscape, network.edges)
print(f"ρ₂ = {metrics.two_edge_connectivity:.3f}")
print(f"P_fail = {metrics.failure_probability:.3f}")

# Add strategic loops
robust_edges = add_strategic_loops(
    landscape, network.edges,
    n_loops=5,
    criterion='betweenness'  # or 'shortest', 'bridge_protection', 'random'
)

# Explore entropy-robustness tradeoff
results = pareto_frontier_analysis(landscape, max_loops=10)
```

---

## 🟢 ADVANCED USAGE EXAMPLES

### Multi-Species Planning:

```python
from hdfm import SPECIES_GUILDS

# Find minimum width satisfying all target guilds
guilds = [SPECIES_GUILDS['small_mammals'],
          SPECIES_GUILDS['medium_mammals']]

min_width = max(g.w_crit for g in guilds)
print(f"Minimum corridor width for all guilds: {min_width}m")
```

### Complete Genetic + Robustness Analysis:

```python
from hdfm import (
    build_dendritic_network,
    calculate_effective_population_size,
    calculate_robustness_metrics,
    add_strategic_loops,
    SPECIES_GUILDS
)

# Build base network
network = build_dendritic_network(landscape)

# Assess genetics
corridor_widths = {edge: 200.0 for edge in network.edges}
guild = SPECIES_GUILDS['large_carnivores']
Ne, _ = calculate_effective_population_size(
    landscape, network.edges, corridor_widths, species_guild=guild
)

# Assess robustness
metrics = calculate_robustness_metrics(landscape, network.edges)

# If robustness is low, add strategic loops
if metrics.two_edge_connectivity < 0.5:
    robust_edges = add_strategic_loops(landscape, network.edges, n_loops=3)
    # Recalculate metrics
    new_metrics = calculate_robustness_metrics(landscape, robust_edges)
```

---

## What Can Be Reproduced from Papers?

### Paper Results You CAN Reproduce:

✅ **Table 1** - Network topology comparisons (entropy differences)
- Dendritic networks achieve lowest entropy ✅
- Gabriel, Delaunay, k-NN all higher ✅
- Relative differences match (~20-40% higher) ✅
- Full width-dependent entropy available ✅

✅ **Table 2** - Species-specific validation
- All 4 guilds from paper implemented ✅
- Width-dependent movement success ✅
- Critical width calculations ✅
- Species-specific genetic thresholds ✅

✅ **Table 3** - Robustness-entropy tradeoffs
- Loop budget analysis ✅
- P_fail calculations ✅
- ρ₂ connectivity metrics ✅
- Pareto frontier analysis ✅

✅ **Figure 1** - Visual network comparison
- Generate comparison plots ✅
- Show dendritic vs. alternatives ✅

✅ **Figure 2** - Width and allocation effects
- Corridor width optimization ✅
- Landscape allocation constraints (20-30%) ✅
- Width-dependent entropy ✅

✅ **Figure 3** - Parameter sensitivity with widths
- Vary α (dispersal scale) ✅
- Vary γ (width sensitivity) ✅
- Corridor width effects ✅

✅ **Figure 4A** - Convergence traces
- Optimization history available ✅
- Can plot entropy over iterations ✅

✅ **Figure 4B** - Computational complexity
- Runtime scaling with landscape size ✅
- Confirm O(n² log n) behavior ✅

✅ **Figure 4C-D** - Robustness analysis
- Pareto frontiers implemented ✅
- Failure probability analysis ✅
- Strategic loop addition ✅

✅ **Conceptual Validation** - Core theoretical claims
- MST minimizes total corridor length ✅
- Dendritic networks minimize entropy ✅
- Statistical significance of topology differences ✅

### Paper Results - All Fully Implemented:

✅ **Backwards Temporal Optimization** (Complete in v0.2.1)
- Climate scenario modeling ✅
- Network adjustment over time ✅
- Width scheduling integration ✅

---

## Development Roadmap

### v0.2.0 (Released: 2025-11-09) - COMPREHENSIVE FEATURE RELEASE ✅

**Implemented Features:**
- [x] Width-dependent entropy calculations with φ(w)
- [x] Entropy rate with stationary distribution H_rate
- [x] Landscape allocation constraints (20-30%)
- [x] Width optimization algorithms
- [x] Full Nₑ(A,w) island model genetics
- [x] Robustness analysis (ρ₂, P_fail, loops)
- [x] Pareto frontier analysis
- [x] All species guilds from paper

**Deliverables:**
- Can calculate width-dependent entropy ✅
- Can optimize corridor widths under budget ✅
- Can check allocation constraints ✅
- Species-specific corridor design fully functional ✅
- Full genetic viability assessment ✅
- Network robustness quantification ✅
- Can reproduce Tables 1-3 and Figures 1-4 from paper ✅

### v0.2.1 (Released: 2025-12-09) - BACKWARDS OPTIMIZATION COMPLETE ✅

**Implemented Features:**
- [x] Complete backwards optimization with width scheduling
- [x] Width schedule output by year
- [x] Species-specific width optimization at each time step
- [x] Integration of WidthOptimizer into BackwardsOptimizer

**Deliverables:**
- Enhanced backwards optimization workflow with width scheduling ✅
- Temporal corridor width schedules ✅
- Full climate-adaptive corridor design ✅

### v0.3.0 (Target: Future) - INTEGRATION & EXAMPLES

**Planned Features:**
- [ ] Jupyter notebook tutorials
- [ ] Additional worked examples
- [ ] Real-world case studies
- [ ] Performance optimization for large landscapes (100+ patches)

**Deliverables:**
- Interactive tutorials
- Best practices documentation
- Performance benchmarks

### v0.4.0 (Target: 6-8 weeks) - GIS INTEGRATION & REAL-WORLD DATA

**Planned Features:**
- [ ] GIS data integration (GeoPandas, Rasterio)
- [ ] Shapefile/GeoJSON import/export
- [ ] Integration with species distribution models
- [ ] Real landscape validation studies
- [ ] Web-based visualization dashboard

**Deliverables:**
- Real-world landscape support
- GIS workflow examples
- Web interface prototype
- Published validation case studies

---

## How to Contribute

We welcome contributions! Priority areas:

1. **Width-dependent entropy** - Core algorithm development
2. **Robustness analysis** - Network resilience quantification
3. **Real landscape validation** - Empirical data integration
4. **Additional species guilds** - Parameter calibration from literature
5. **Case studies** - Real-world applications

See `CONTRIBUTING.md` for details.

---

## Questions or Issues?

- **Feature requests:** Open an issue with tag `enhancement`
- **Bug reports:** Open an issue with tag `bug`
- **Implementation questions:** Open a discussion
- **Scientific collaboration:** Contact viridisnorthllc@gmail.com

---

## Transparency Commitment

This document will be updated with each release to reflect current implementation status. We commit to honest disclosure of what is and isn't implemented to enable informed use of this framework.

**Last Verified:** 2025-12-09
**Next Review:** With v0.3.0 release
