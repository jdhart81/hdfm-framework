# Hierarchical Dendritic Forest Management (HDFM) Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**A computational framework for entropy-minimizing corridor network optimization with backwards climate adaptation.**

This repository provides reference implementations of the algorithms and methods described in:

> Hart, J. (2024). "Hierarchical Dendritic Forest Management: A Vision for Technology-Enabled Landscape Conservation." *EcoEvoRxiv*. [Preprint]

## Overview

Forest corridor networks are critical for biodiversity conservation and climate adaptation, yet remain underimplemented due to complexity and cost. HDFM addresses this through:

- **Information-theoretic optimization**: Formulates corridor design as entropy minimization
- **Dendritic network topology**: Proves tree structures minimize landscape entropy
- **Backwards climate optimization**: Designs from 2100 projections to present implementation
- **Technology integration**: Leverages AI, GPS geofencing, and satellite monitoring

This framework enables passive, automated corridor establishment at unprecedented scale and minimal cost.

## ⚠️ Implementation Status

**Current Version:** 0.1.0 (Foundational Release)
**Paper Coverage:** ~50-55% of features fully implemented

This is an **actively developed** framework. Core dendritic optimization and validation tools are production-ready, but several advanced features from the papers are planned for future releases.

**What Works Now:**
- ✅ Dendritic network (MST) optimization
- ✅ Basic entropy framework (H_mov + penalties)
- ✅ Alternative topology comparisons
- ✅ Monte Carlo validation
- ✅ Backwards climate optimization structure
- ✅ **NEW:** Species-specific parameters (Table 2 from paper)

**Coming Soon (v0.2.0, ~2-3 weeks):**
- 🔴 Width-dependent entropy calculations
- 🔴 Dual entropy formulations (H_rate)
- 🔴 Landscape allocation constraints (20-30%)
- 🔴 Width optimization algorithms

**See [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) for complete feature status and workarounds.**
**See [`PAPER_IMPLEMENTATION_REVIEW.md`](PAPER_IMPLEMENTATION_REVIEW.md) for detailed gap analysis.**

## Key Features

- 🌲 **Landscape entropy calculation** with movement, connectivity, and topology terms
- 🔀 **Dendritic network construction** via minimum spanning trees
- ⏮️ **Backwards optimization algorithm** for climate-adaptive corridor design
- 📊 **Synthetic landscape validation** reproducing paper results
- 📈 **Comparative topology analysis** (Gabriel graphs, Delaunay, k-NN, etc.)
- 🎨 **Visualization tools** for networks, entropy landscapes, and optimization traces

## Installation

```bash
# Clone repository
git clone https://github.com/jdhart81/hdfm-framework.git
cd hdfm-framework

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

### Generate and optimize a synthetic landscape

```python
from hdfm import SyntheticLandscape, DendriticOptimizer, calculate_entropy

# Create 15-patch landscape
landscape = SyntheticLandscape(n_patches=15, extent=10000)

# Build dendritic network (MST)
optimizer = DendriticOptimizer(landscape)
network = optimizer.build_dendritic_network()

# Calculate landscape entropy
H_total, H_components = calculate_entropy(landscape, network)
print(f"Total entropy: {H_total:.3f}")
print(f"Movement entropy: {H_components['H_mov']:.3f}")
```

### Run backwards climate optimization

```python
from hdfm import BackwardsOptimizer, ClimateScenario

# Define climate trajectory
scenario = ClimateScenario(
    years=[2025, 2050, 2075, 2100],
    temperature_changes=[0, 1.5, 2.5, 3.5],  # °C
    precipitation_changes=[0, -5, -10, -15]  # %
)

# Optimize backwards from 2100
optimizer = BackwardsOptimizer(landscape, scenario)
corridor_sequence = optimizer.optimize()

# Get present-day implementation plan
current_network = corridor_sequence[0]
```

### Use species-specific parameters (NEW in v0.1.0)

```python
from hdfm import SPECIES_GUILDS, print_guild_summary

# View all available species guilds (Table 2 from paper)
print_guild_summary()

# Get species-specific parameters
small_mammals = SPECIES_GUILDS['small_mammals']
large_carnivores = SPECIES_GUILDS['large_carnivores']

# Calculate movement success at different corridor widths
print(f"Small mammals at 150m: {small_mammals.movement_success(150):.2%}")
print(f"Large carnivores at 350m: {large_carnivores.movement_success(350):.2%}")

# Find required width for target success
width = small_mammals.required_width_for_success(0.85)
print(f"Width for 85% success: {width:.0f}m")

# Available guilds: 'small_mammals', 'medium_mammals',
#                  'large_carnivores', 'long_lived'
```

### Reproduce paper validation

```python
from examples.synthetic_landscape_validation import run_full_validation

# Runs 100 iterations comparing 5 network topologies
# Outputs: entropy distributions, convergence plots, statistical tests
results = run_full_validation(n_landscapes=100, n_patches=15)
```

## Core Algorithms

### Entropy Minimization

Landscape entropy is formulated as:

```
H(L) = H_mov + λ₁·C(L) + λ₂·F(L) + λ₃·D(L)
```

Where:
- `H_mov`: Movement entropy from cost-based dispersal kernel
- `C(L)`: Connectivity constraint (ensures N_e > 500)
- `F(L)`: Forest topology penalty (favors dendritic structure)
- `D(L)`: Disturbance response time penalty

**Implementation**: `hdfm/entropy.py`

### Dendritic Network Construction

Minimum spanning tree (MST) algorithm finds the minimum-entropy connected network:

```python
def build_dendritic_network(landscape):
    """
    Construct dendritic corridor network via MST.
    
    Invariants:
    - Network is connected (all patches reachable)
    - Network is acyclic (tree structure, no loops)
    - Total corridor length is minimized
    - Entropy H(dendritic) ≤ H(alternative) for any alternative topology
    """
    # Kruskal's algorithm with entropy-weighted edges
    edges = compute_edge_costs(landscape)
    mst = kruskal_mst(edges)
    return DendriticNetwork(mst)
```

**Complexity**: O(n² log n) for n patches

**Implementation**: `hdfm/network.py`

### Backwards Optimization

Works from desired 2100 state to present implementation:

```python
def backwards_optimize(landscape, climate_trajectory):
    """
    Optimize corridor network via backwards iteration.
    
    Invariants:
    - Maintains connectivity at each time step
    - Minimizes entropy at 2100 target state
    - Converges within max_iterations
    - Each corridor appears at optimal establishment time
    """
    # Initialize at 2100 with desired connectivity
    network_2100 = initialize_target_network(landscape, climate_trajectory[-1])
    
    # Work backwards to present
    corridor_schedule = []
    for t in reversed(climate_trajectory):
        network_t = optimize_network_at_time(network_2100, landscape, t)
        corridor_schedule.insert(0, network_t)
    
    return corridor_schedule
```

**Implementation**: `hdfm/optimization.py`

## Validation Results

Synthetic landscape experiments (n=100, 15 patches each) confirm theoretical predictions:

| Network Topology | Mean Entropy | Std Dev | vs. Dendritic |
|-----------------|--------------|---------|---------------|
| **Dendritic (MST)** | **2.28** | 0.04 | — |
| Gabriel Graph | 2.87 | 0.06 | +25.9% |
| Delaunay Triangulation | 3.12 | 0.08 | +36.8% |
| k-Nearest Neighbors | 2.94 | 0.07 | +28.9% |
| Threshold Distance | 3.05 | 0.09 | +33.8% |

**Statistical significance**: All comparisons p < 0.001 (Wilcoxon signed-rank test)

**Convergence**: Backwards optimization converges within 50 iterations across all landscapes

**Climate resilience**: Backwards-optimized networks maintain 15-20% higher connectivity under 2100 scenarios vs. forward-looking designs

Run validation: `python examples/synthetic_landscape_validation.py`

## Repository Structure

```
hdfm-framework/
├── hdfm/                          # Core package
│   ├── landscape.py               # Landscape representation and graph construction
│   ├── species.py                 # Species-specific parameters (NEW in v0.1.0)
│   ├── entropy.py                 # Entropy calculations (H_mov, C, F, D terms)
│   ├── network.py                 # Network topology algorithms (MST, alternatives)
│   ├── optimization.py            # Backwards optimization algorithm
│   ├── validation.py              # Empirical validation tools
│   └── visualization.py           # Plotting and visualization
├── examples/                      # Runnable demonstrations
│   ├── synthetic_landscape_validation.py
│   ├── backwards_optimization_demo.py
│   ├── entropy_comparison.py
│   └── climate_scenarios.py
├── tests/                         # Unit tests with invariant verification
│   ├── test_entropy.py
├── PAPER_IMPLEMENTATION_REVIEW.md # Detailed gap analysis (NEW)
├── KNOWN_LIMITATIONS.md           # Implementation status & roadmap (NEW)
├── README.md                      # This file
├── QUICKSTART.md                  # 5-minute getting started guide
├── CONTRIBUTING.md                # Contribution guidelines
└── requirements.txt               # Python dependencies
│   ├── test_network.py
│   └── test_optimization.py
├── docs/                          # Extended documentation
│   ├── theory_overview.md
│   ├── algorithm_details.md
│   └── implementation_guide.md
└── data/                          # Example datasets
    └── example_landscapes/
```

## Implementation Technologies

The framework is designed for integration with:

- **AI/ML**: Landscape analysis, pattern recognition, corridor identification
- **GPS Geofencing**: Automated exclusion zones on forestry equipment
- **Satellite Monitoring**: Sentinel-2, Landsat, Planet for verification
- **Cloud Computing**: Scalable optimization for large landscapes (100+ patches)

See `docs/implementation_guide.md` for details.

## Requirements

- Python 3.8+
- NumPy >= 1.20
- SciPy >= 1.7
- NetworkX >= 2.6
- Matplotlib >= 3.3
- scikit-learn >= 0.24

See `requirements.txt` for complete dependencies.

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{hart2024hdfm,
  title={Hierarchical Dendritic Forest Management: A Vision for Technology-Enabled Landscape Conservation},
  author={Hart, Justin},
  journal={EcoEvoRxiv},
  year={2024},
  doi={[preprint DOI]}
}
```

## Contributing

Contributions welcome! Areas of particular interest:

- **Empirical validation**: Real-world landscape testing
- **Climate model integration**: Coupling with species distribution models
- **Genetic connectivity**: Integration with landscape genetics simulations
- **Cost-benefit analysis**: Economic modeling of implementation pathways
- **Technology integration**: GPS API, satellite data pipelines

See `CONTRIBUTING.md` for guidelines.

## License

MIT License - see `LICENSE` file for details.

## Contact

**Justin Hart**  
Viridis LLC  
viridisnorthllc@gmail.com

## Acknowledgments

This framework builds on foundational work in:
- Graph-theoretic landscape connectivity (Urban et al. 2009)
- Circuit theory and least-cost paths (McRae et al. 2008)
- Systematic conservation planning (Margules & Pressey 2000)
- TRIAD forest management (Seymour & Hunter 1999)

## Future Directions

Priority development areas:

1. **Real landscape validation**: Test on actual forest management units
2. **Species-specific parameterization**: Dispersal kernels for focal taxa
3. **Economic optimization**: Joint optimization of production + conservation
4. **Multi-objective frameworks**: Timber, carbon, water, biodiversity
5. **Real-time monitoring**: Automated satellite change detection pipelines

See `docs/roadmap.md` for detailed development plan.

---

**Status**: Research prototype | **Version**: 0.1.0 | **Last updated**: November 2024
