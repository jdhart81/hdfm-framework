# Genetic Population Size and Robustness Analysis Guide

This guide covers the advanced features for genetic population dynamics (effective population size Nₑ tracking) and robustness analysis with looped topologies.

## Table of Contents

1. [Effective Population Size (Nₑ) Tracking](#effective-population-size-nₑ-tracking)
2. [Robustness Analysis](#robustness-analysis)
3. [Complete Workflow Example](#complete-workflow-example)
4. [API Reference](#api-reference)

---

## Effective Population Size (Nₑ) Tracking

The framework now implements the full island model for calculating metapopulation effective population size, accounting for:
- Local population sizes in each patch
- Width-dependent migration rates
- Co-ancestry coefficients between patches
- Network topology effects on gene flow

### Theory

The island model formula for effective population size is:

```
Nₑ(A,w) = [∑ᵢ nᵢ]² / [∑ᵢ nᵢ² + ∑ᵢ ∑ⱼ≠ᵢ 2nᵢnⱼFᵢⱼ(A,w)]
```

Where:
- `nᵢ`: Local population size in patch i
- `Fᵢⱼ(A,w)`: Co-ancestry coefficient between patches i and j
- `A`: Network topology (adjacency matrix)
- `w`: Corridor width vector

### Basic Usage

```python
from hdfm import (
    SyntheticLandscape,
    build_dendritic_network,
    calculate_effective_population_size,
    check_genetic_viability,
    SPECIES_GUILDS
)

# Create landscape with population data
landscape = SyntheticLandscape(n_patches=15, random_seed=42)

# Assign populations to patches
for patch in landscape.patches:
    # Population based on area and quality
    patch.population = patch.area * 10.0 * patch.quality

# Build corridor network
network = build_dendritic_network(landscape)

# Define corridor widths
corridor_widths = {edge: 150.0 for edge in network.edges}

# Calculate effective population size
guild = SPECIES_GUILDS['medium_mammals']
Ne, components = calculate_effective_population_size(
    landscape,
    network.edges,
    corridor_widths,
    dispersal_scale=8300.0,  # 8.3 km for medium mammals
    alpha=0.12,
    species_guild=guild
)

print(f"Effective population size: Nₑ = {Ne:.1f}")
print(f"Total census population: N = {components['N_total']:.1f}")

# Check genetic viability
viable, threshold, message = check_genetic_viability(Ne, species_guild=guild)
print(f"Genetic viability: {message}")
```

### Migration Rates and Co-ancestry

```python
from hdfm import calculate_migration_rate, calculate_coancestry_coefficient

# Calculate migration rate between two patches
m_ij = calculate_migration_rate(
    distance=1500.0,      # 1.5 km
    width=150.0,          # 150 m corridor
    dispersal_scale=8300.0,
    alpha=0.12,
    species_guild=guild
)
print(f"Migration rate: {m_ij:.4f}")

# Calculate co-ancestry coefficient
F_ij = calculate_coancestry_coefficient(m_ij)
print(f"Co-ancestry coefficient: {F_ij:.4f}")
```

### Long-term Genetic Metrics

```python
from hdfm import (
    calculate_inbreeding_coefficient,
    calculate_genetic_diversity_loss
)

# Project inbreeding over 100 generations
F_100 = calculate_inbreeding_coefficient(Ne, generations=100)
print(f"Expected inbreeding (100 gen): F = {F_100:.3f}")

# Project heterozygosity retention
H_100 = calculate_genetic_diversity_loss(Ne, generations=100)
print(f"Heterozygosity retained (100 gen): {H_100*100:.1f}%")
```

### Species-Specific Thresholds

Different species guilds have different Nₑ thresholds for genetic viability:

| Guild | Nₑ Threshold | Rationale |
|-------|--------------|-----------|
| Small Mammals | 350 | Short generation time, high fecundity |
| Medium Mammals | 500 | Standard 50/500 rule |
| Large Carnivores | 750 | Large-bodied, long-lived |
| Long-lived Species | 1200 | Evolutionary potential, climate adaptation |

```python
# Check viability for different guilds
for guild_name, guild in SPECIES_GUILDS.items():
    viable, thresh, msg = check_genetic_viability(Ne, species_guild=guild)
    print(f"{guild_name}: {msg}")
```

---

## Robustness Analysis

The robustness module provides tools for analyzing network resilience to corridor failures, including:
- 2-edge-connectivity metrics (ρ₂)
- Strategic loop addition to MST networks
- Catastrophic failure probability (P_fail)
- Edge redundancy scoring
- Pareto frontier analysis (entropy vs. robustness tradeoffs)

### Theory

**2-Edge-Connectivity (ρ₂)**:
Measures the fraction of node pairs that remain connected after removing any single edge:

```
ρ₂ = (# node pairs with ≥2 edge-disjoint paths) / (n choose 2)
```

Properties:
- ρ₂ = 0 for trees (MST has no redundancy)
- ρ₂ > 0 requires loops/cycles
- ρ₂ = 1 for 2-edge-connected graphs

**Failure Probability (P_fail)**:
Probability that k random edge failures disconnect the network.

### Basic Usage

```python
from hdfm import (
    SyntheticLandscape,
    build_dendritic_network,
    calculate_robustness_metrics
)

# Create landscape
landscape = SyntheticLandscape(n_patches=15, random_seed=42)

# Build MST network
network = build_dendritic_network(landscape)

# Calculate robustness metrics
metrics = calculate_robustness_metrics(
    landscape,
    network.edges,
    k_failures=1  # Test single edge failure
)

print(f"2-edge-connectivity: ρ₂ = {metrics.two_edge_connectivity:.3f}")
print(f"Failure probability: P_fail = {metrics.failure_probability:.3f}")
print(f"Number of loops: {metrics.n_loops}")
print(f"Overall redundancy: {metrics.redundancy_score:.3f}")
```

### Strategic Loop Addition

Add loops to MST networks to improve robustness:

```python
from hdfm import add_strategic_loops

# Add 5 strategic loops to MST
edges_with_loops = add_strategic_loops(
    landscape,
    network.edges,
    n_loops=5,
    criterion='betweenness'  # Reduce bottleneck edges
)

# Calculate improved robustness
metrics_improved = calculate_robustness_metrics(
    landscape,
    edges_with_loops,
    k_failures=1
)

print(f"Improved ρ₂: {metrics_improved.two_edge_connectivity:.3f}")
print(f"Improved P_fail: {metrics_improved.failure_probability:.3f}")
```

Loop addition strategies:
- `'betweenness'`: Reduce betweenness centrality bottlenecks (recommended)
- `'shortest'`: Add shortest available edges
- `'bridge_protection'`: Protect critical bridge edges
- `'random'`: Random loop addition (baseline)

### Edge Redundancy Scores

Identify critical vs. redundant edges:

```python
from hdfm import calculate_edge_redundancy_scores

scores = calculate_edge_redundancy_scores(landscape, edges_with_loops)

# Find most critical edges
critical_edges = [(edge, score) for edge, score in scores.items()
                   if score > 0.8]
print(f"Critical edges (score > 0.8): {len(critical_edges)}")

# Find redundant edges
redundant_edges = [(edge, score) for edge, score in scores.items()
                    if score < 0.3]
print(f"Redundant edges (score < 0.3): {len(redundant_edges)}")
```

Redundancy score interpretation:
- **Score = 1.0**: Bridge edge (removal disconnects network)
- **Score > 0.7**: High importance (few alternative paths)
- **Score 0.3-0.7**: Moderate redundancy
- **Score < 0.3**: Highly redundant (many alternative paths)

### 2-Edge-Connectivity Analysis

```python
from hdfm import calculate_two_edge_connectivity

rho_2, components = calculate_two_edge_connectivity(landscape, network.edges)

print(f"ρ₂ = {rho_2:.3f}")
print(f"Node pairs with 2 edge-disjoint paths: {components['two_connected_pairs']}")
print(f"Total node pairs: {components['total_pairs']}")
print(f"Bridge edges: {components['n_bridges']}")
print(f"Vulnerable edges: {components['vulnerable_edges']}")
```

### Failure Probability Analysis

```python
from hdfm import calculate_failure_probability

# Test k=1 failure
P_fail_1, comp_1 = calculate_failure_probability(
    landscape, edges_with_loops, k=1
)

# Test k=2 failures
P_fail_2, comp_2 = calculate_failure_probability(
    landscape, edges_with_loops, k=2
)

print(f"P_fail(k=1) = {P_fail_1:.3f}")
print(f"P_fail(k=2) = {P_fail_2:.3f}")
```

### Pareto Frontier Analysis

Explore the entropy-robustness tradeoff:

```python
from hdfm import pareto_frontier_analysis

results = pareto_frontier_analysis(
    landscape,
    max_loops=10,
    loop_criterion='betweenness'
)

# Plot results (requires matplotlib)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(results['entropy'], results['robustness'], 'o-',
         label='Configurations')

# Highlight Pareto-optimal points
pareto_entropy = [p[0] for p in results['pareto_points']]
pareto_robust = [p[1] for p in results['pareto_points']]
plt.plot(pareto_entropy, pareto_robust, 'r*',
         markersize=15, label='Pareto optimal')

plt.xlabel('Entropy H(L)')
plt.ylabel('Robustness ρ₂')
plt.title('Entropy-Robustness Pareto Frontier')
plt.legend()
plt.grid(True)
plt.show()

# Print Pareto points
print("\nPareto-Optimal Configurations:")
for entropy, robustness, n_loops in results['pareto_points']:
    print(f"  Loops: {n_loops}, H = {entropy:.3f}, ρ₂ = {robustness:.3f}")
```

---

## Complete Workflow Example

Integrate genetic and robustness analysis:

```python
from hdfm import (
    SyntheticLandscape,
    build_dendritic_network,
    calculate_effective_population_size,
    check_genetic_viability,
    add_strategic_loops,
    calculate_robustness_metrics,
    pareto_frontier_analysis,
    SPECIES_GUILDS
)

# 1. Create landscape
landscape = SyntheticLandscape(n_patches=20, random_seed=42)

# Assign populations
for patch in landscape.patches:
    patch.population = patch.area * 12.0 * patch.quality

# 2. Build dendritic network
network = build_dendritic_network(landscape)
print(f"MST edges: {len(network.edges)}")

# 3. Genetic analysis
guild = SPECIES_GUILDS['large_carnivores']
corridor_widths = {edge: 250.0 for edge in network.edges}

Ne_mst, comp_mst = calculate_effective_population_size(
    landscape, network.edges, corridor_widths,
    dispersal_scale=20000.0,  # 20 km for large carnivores
    alpha=0.05,
    species_guild=guild
)

viable_mst, thresh, msg_mst = check_genetic_viability(Ne_mst, species_guild=guild)

print(f"\nMST Genetic Analysis:")
print(f"  Nₑ = {Ne_mst:.1f}")
print(f"  {msg_mst}")

# 4. Robustness analysis (MST)
rob_mst = calculate_robustness_metrics(landscape, network.edges)
print(f"\nMST Robustness:")
print(f"  ρ₂ = {rob_mst.two_edge_connectivity:.3f}")
print(f"  P_fail = {rob_mst.failure_probability:.3f}")

# 5. Add strategic loops
edges_robust = add_strategic_loops(
    landscape, network.edges, n_loops=8, criterion='betweenness'
)
print(f"\nRobust network edges: {len(edges_robust)}")

# Update corridor widths for new edges
for edge in edges_robust:
    if edge not in corridor_widths:
        corridor_widths[edge] = 250.0

# 6. Genetic analysis (with loops)
Ne_robust, comp_robust = calculate_effective_population_size(
    landscape, edges_robust, corridor_widths,
    dispersal_scale=20000.0,
    alpha=0.05,
    species_guild=guild
)

viable_robust, _, msg_robust = check_genetic_viability(Ne_robust, species_guild=guild)

print(f"\nRobust Network Genetic Analysis:")
print(f"  Nₑ = {Ne_robust:.1f}")
print(f"  {msg_robust}")

# 7. Robustness analysis (with loops)
rob_robust = calculate_robustness_metrics(landscape, edges_robust)
print(f"\nRobust Network Robustness:")
print(f"  ρ₂ = {rob_robust.two_edge_connectivity:.3f}")
print(f"  P_fail = {rob_robust.failure_probability:.3f}")

# 8. Compare improvements
print(f"\nIMPROVEMENTS:")
print(f"  ΔNₑ = +{Ne_robust - Ne_mst:.1f}")
print(f"  Δρ₂ = +{rob_robust.two_edge_connectivity - rob_mst.two_edge_connectivity:.3f}")
print(f"  ΔP_fail = {rob_robust.failure_probability - rob_mst.failure_probability:.3f}")

# 9. Pareto frontier
print(f"\nGenerating Pareto frontier...")
pareto_results = pareto_frontier_analysis(landscape, max_loops=15)
print(f"  Explored {len(pareto_results['n_loops'])} configurations")
print(f"  Pareto-optimal points: {len(pareto_results['pareto_points'])}")
```

---

## API Reference

### Genetics Module (`hdfm.genetics`)

#### `calculate_effective_population_size(landscape, edges, corridor_widths, ...)`
Calculate metapopulation effective population size using island model.

**Parameters:**
- `landscape`: Landscape object with patch populations
- `edges`: List of corridor edges
- `corridor_widths`: Dict mapping edges to widths
- `dispersal_scale`: Characteristic dispersal distance (meters)
- `alpha`: Dispersal cost parameter
- `species_guild`: Optional SpeciesGuild for width calculations
- `default_population`: Default population if patch.population is None

**Returns:** `(Ne, components_dict)`

---

#### `calculate_migration_rate(distance, width, dispersal_scale, alpha, species_guild)`
Calculate width-dependent migration rate between patches.

**Parameters:**
- `distance`: Corridor distance (meters)
- `width`: Corridor width (meters)
- `dispersal_scale`: Characteristic dispersal distance (meters)
- `alpha`: Dispersal cost parameter
- `species_guild`: Optional SpeciesGuild

**Returns:** Migration rate mᵢⱼ [0, 1]

---

#### `check_genetic_viability(Ne, species_guild, threshold)`
Check if effective population size meets genetic viability threshold.

**Parameters:**
- `Ne`: Effective population size
- `species_guild`: Optional SpeciesGuild with Ne_threshold
- `threshold`: Optional custom threshold

**Returns:** `(is_viable, threshold_used, status_message)`

---

### Robustness Module (`hdfm.robustness`)

#### `calculate_robustness_metrics(landscape, edges, k_failures)`
Calculate comprehensive robustness metrics.

**Parameters:**
- `landscape`: Landscape object
- `edges`: List of corridor edges
- `k_failures`: Number of simultaneous failures for P_fail

**Returns:** `RobustnessMetrics` object

---

#### `add_strategic_loops(landscape, mst_edges, n_loops, criterion)`
Add strategic loops to MST to improve robustness.

**Parameters:**
- `landscape`: Landscape object
- `mst_edges`: Minimum spanning tree edges
- `n_loops`: Number of loops to add
- `criterion`: Strategy ('betweenness', 'shortest', 'bridge_protection', 'random')

**Returns:** List of edges (MST + loops)

---

#### `pareto_frontier_analysis(landscape, max_loops, loop_criterion, **entropy_kwargs)`
Generate Pareto frontier for entropy-robustness tradeoff.

**Parameters:**
- `landscape`: Landscape object
- `max_loops`: Maximum number of loops to explore
- `loop_criterion`: Strategy for loop addition
- `**entropy_kwargs`: Parameters for entropy calculation

**Returns:** Dict with arrays of results and Pareto points

---

## References

1. Wang, J., & Caballero, A. (1999). Developments in predicting the effective size of subdivided populations. *Heredity*, 82(2), 212-226.

2. Wright, S. (1931). Evolution in Mendelian populations. *Genetics*, 16(2), 97-159.

3. Falconer, D.S. & Mackay, T.F.C. (1996). *Introduction to Quantitative Genetics*. 4th Edition.

4. Hart, J. (2024). Habitat diversity-flow maximization framework for conservation corridor networks.
