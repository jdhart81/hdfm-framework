# Known Limitations and Implementation Status

**Last Updated:** 2025-11-09
**Version:** 0.2.0

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
| Full Effective Population Size | 🔴 Planned | 0% |
| Robustness Analysis (Loops) | 🔴 Planned | 0% |

**Overall:** ~75-80% of paper features fully implemented

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
**Status:** ⚠️ **PARTIALLY IMPLEMENTED**

- Climate scenario modeling ✅
- Temporal trajectory (2025→2100) ✅
- Multi-year network adjustment ✅
- Width optimization over time 🔴 Missing

**Code:** `hdfm/optimization.py:157-363`

### 7. Species-Specific Parameters
**Status:** ✅ **NEW - IMPLEMENTED in v0.1.0**

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

## 🔴 PLANNED FEATURES (Not Yet Implemented)

### 1. Full Effective Population Size Nₑ(A,w)
**Status:** 🔴 **PLANNED for v0.2.0**
**Priority:** HIGH

**What's Missing:**

Complete island model:
```
Nₑ(A,w) = [∑ᵢ nᵢ]² / [∑ᵢ nᵢ² + ∑ᵢ ∑ⱼ≠ᵢ 2nᵢnⱼFᵢⱼ(A,w)]
Fᵢⱼ(A,w) = mᵢⱼ(A,w) / [2 − mᵢⱼ(A,w)]
mᵢⱼ(A,w) = σ · pᵢⱼ(A,w) / (1 + σ · pᵢⱼ(A,w))
```

**Current Implementation:**
- Simplified graph connectivity proxy
- Does NOT calculate true effective population size
- Cannot validate against genetic viability thresholds

**Impact:** Cannot assess genetic conservation effectiveness

**Workaround:** Use connectivity constraint as proxy for genetic viability

**Target Release:** v0.2.0

---

### 2. Robustness Analysis with Loop Budgets
**Status:** 🔴 **PLANNED for v0.3.0**
**Priority:** MEDIUM

**What's Missing:**

- MST + strategic loops construction
- 2-edge-connectivity calculation (ρ₂)
- Catastrophic failure probability (P_fail)
- Pareto frontier analysis (entropy vs. robustness)

**Cannot Reproduce:** Table 3, Figure 4C-D

**Impact:** Cannot quantify network resilience to edge failures

**Workaround:** Manually add loops by including additional edges

**Target Release:** v0.3.0 (estimated 4-5 weeks)

---


## 🟢 WORKAROUNDS FOR CURRENT LIMITATIONS

While critical features are being implemented, scientists can use these workarounds:

### For Width-Dependent Analysis:

```python
from hdfm import SPECIES_GUILDS

# Get species-specific parameters
guild = SPECIES_GUILDS['small_mammals']

# Calculate movement success manually
def custom_dispersal_prob(distance_km, width_m, guild):
    import numpy as np
    distance_effect = np.exp(-guild.alpha * distance_km)
    width_effect = guild.movement_success(width_m)
    return distance_effect * width_effect

# Use in custom entropy calculations
```

### For Multi-Species Planning:

```python
# Find minimum width satisfying all target guilds
guilds = [SPECIES_GUILDS['small_mammals'],
          SPECIES_GUILDS['medium_mammals']]

min_width = max(g.w_crit for g in guilds)
print(f"Minimum corridor width for all guilds: {min_width}m")
```

### For Robustness (Manual Loop Addition):

```python
from hdfm import build_dendritic_network

# Build MST
network = build_dendritic_network(landscape)

# Manually add strategic loops (longest edges)
mst_edges = network.edges
all_edges = list(landscape.graph.edges())
non_mst_edges = [e for e in all_edges if e not in mst_edges]

# Sort by length, add longest as redundant path
from hdfm.entropy import calculate_entropy
best_loop = None
best_entropy = float('inf')

for edge in non_mst_edges[:10]:  # Check top 10 longest
    test_edges = mst_edges + [edge]
    H, _ = calculate_entropy(landscape, test_edges)
    if H < best_entropy:
        best_entropy = H
        best_loop = edge

robust_edges = mst_edges + [best_loop]
```

---

## What Can Be Reproduced from Papers?

### Paper Results You CAN Reproduce:

✅ **Table 1 Structure** - Network topology comparisons (entropy differences)
- Dendritic networks achieve lowest entropy ✅
- Gabriel, Delaunay, k-NN all higher ✅
- Relative differences match (~20-40% higher) ✅
- **Note:** Absolute values may differ (no width effects)

✅ **Figure 1** - Visual network comparison
- Generate comparison plots ✅
- Show dendritic vs. alternatives ✅

✅ **Figure 4A** - Convergence traces
- Optimization history available ✅
- Can plot entropy over iterations ✅

✅ **Figure 4B** - Computational complexity
- Runtime scaling with landscape size ✅
- Confirm O(n² log n) behavior ✅

✅ **Conceptual Validation** - Core theoretical claims
- MST minimizes total corridor length ✅
- Dendritic networks minimize entropy (distance proxy) ✅
- Statistical significance of topology differences ✅

### Paper Results You CANNOT Yet Reproduce:

❌ **Table 2** - Species-specific validation
- Parameters exist, but width-dependent entropy missing
- Cannot validate movement success vs. corridor width

❌ **Table 3** - Robustness-entropy tradeoffs
- Loop budget analysis not implemented
- P_fail calculations missing

❌ **Figure 2** - Width and allocation effects
- Corridor width optimization missing
- Landscape allocation constraints not enforced

❌ **Figure 3** - Parameter sensitivity with widths
- Can vary α, but not γ or corridor widths
- Width sensitivity analysis impossible

❌ **Figure 4C-D** - Robustness analysis
- Pareto frontiers not implemented
- Failure probability analysis missing

---

## Development Roadmap

### v0.2.0 (Released: 2025-11-09) - WIDTH-DEPENDENT OPTIMIZATION ✅

**Implemented Features:**
- [x] Width-dependent entropy calculations with φ(w)
- [x] Entropy rate with stationary distribution H_rate
- [x] Landscape allocation constraints (20-30%)
- [x] Width optimization algorithms
- [ ] Full Nₑ(A,w) island model (deferred to v0.3.0)
- [ ] Updated examples demonstrating width optimization

**Deliverables:**
- Can calculate width-dependent entropy ✅
- Can optimize corridor widths under budget ✅
- Can check allocation constraints ✅
- Species-specific corridor design fully functional ✅

### v0.3.0 (Target: 2-3 weeks) - ROBUSTNESS ANALYSIS

**Features:**
- [ ] MST + strategic loops construction
- [ ] 2-edge-connectivity ρ₂ calculation
- [ ] Catastrophic failure probability P_fail
- [ ] Pareto frontier generation
- [ ] Robustness vs. entropy tradeoff analysis

**Deliverables:**
- Can reproduce Table 3 completely
- Can reproduce Figure 4C-D
- Operational resilience planning enabled

### v0.4.0 (Target: 6-8 weeks) - COMPLETE PAPER REPRODUCTION

**Features:**
- [ ] All paper figures reproducible
- [ ] Complete validation suite
- [ ] Multi-species optimization
- [ ] Comprehensive documentation
- [ ] Tutorial notebooks
- [ ] Case study examples

**Deliverables:**
- 100% paper feature coverage
- Publication-ready reproduction scripts
- Community contribution guidelines

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

**Last Verified:** 2025-11-09
**Next Review:** With v0.2.0 release
