
# Hierarchical Dendritic Forest Management (HDFM) Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**A computational framework for designing optimal forest corridor networks using information theory and climate-adaptive planning.**

## What is HDFM?

The HDFM Framework is a Python-based toolkit that helps conservation planners, land managers, and researchers design wildlife corridor networks that:
- **Minimize landscape entropy** - Use information-theoretic principles to find the most efficient corridor configurations
- **Adapt to climate change** - Design corridors that remain effective under future climate scenarios using backwards optimization
- **Leverage technology** - Support integration with AI, GPS geofencing, and satellite monitoring for automated implementation

This framework provides reference implementations of algorithms described in:
> Hart, J. (2024). "Hierarchical Dendritic Forest Management: A Vision for Technology-Enabled Landscape Conservation." *EcoEvoRxiv*. [Preprint]

## Why Use This Framework?

Traditional corridor planning is complex, expensive, and often fails to account for long-term climate change. HDFM addresses these challenges by:

- **Proving dendritic (tree-like) networks are optimal** - Mathematical proofs and validation show tree structures minimize entropy
- **Starting from the future** - Backwards optimization designs from 2100 climate projections to present-day implementation
- **Enabling passive implementation** - Technology integration (GPS geofencing, satellite monitoring) allows low-cost, automated corridor establishment
- **Providing validated tools** - Reproducible algorithms with comprehensive testing and validation

## Quick Start

### Installation (5 minutes)

```bash
# Clone the repository
git clone https://github.com/jdhart81/hdfm-framework.git
cd hdfm-framework/hdfm-framework

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Test the installation
python test_installation.py
```

### Run Your First Example (2 minutes)

```python
from hdfm import SyntheticLandscape, build_dendritic_network, calculate_entropy

# Create a synthetic landscape with 15 forest patches
landscape = SyntheticLandscape(n_patches=15, extent=10000)

# Build optimal dendritic corridor network
network = build_dendritic_network(landscape)

# Calculate landscape entropy
H_total, components = calculate_entropy(landscape, network)

print(f"Total entropy: {H_total:.3f}")
print(f"Number of corridors: {len(network.edges)}")
```

### Reproduce Paper Validation (3 minutes)

```bash
# Run the full validation suite from the paper
cd hdfm-framework
python examples/synthetic_landscape_validation.py
```

This validates that dendritic networks minimize entropy compared to alternative topologies (Gabriel graphs, Delaunay triangulations, etc.).

## Repository Roadmap

This repository is organized into several key areas:

### 📦 Core Package (`hdfm-framework/hdfm/`)

The main Python package with production-ready algorithms:

| Module | Purpose | Key Features |
|--------|---------|--------------|
| **landscape.py** | Landscape representation | Patch networks, distance matrices, graph construction |
| **species.py** | Species-specific parameters | Movement success, corridor width requirements (Table 2 from paper) |
| **entropy.py** | Entropy calculations | Movement entropy (H_mov), connectivity, topology, disturbance penalties |
| **network.py** | Network topology algorithms | Dendritic (MST) construction, alternative topologies, comparisons |
| **optimization.py** | Optimization algorithms | Backwards climate optimization, dendritic optimizer |
| **validation.py** | Validation tools | Monte Carlo validation, statistical comparisons |
| **visualization.py** | Plotting & graphics | Network plots, entropy surfaces, optimization traces |

### 📚 Examples (`hdfm-framework/examples/`)

Ready-to-run demonstrations:

- **synthetic_landscape_validation.py** - Reproduces paper results comparing 5 network topologies across 100 random landscapes
- **backwards_optimization_demo.py** - Demonstrates climate-adaptive corridor planning from 2100 to present

### 🧪 Tests (`hdfm-framework/tests/`)

Unit tests with invariant verification:
- **test_entropy.py** - Validates entropy calculations and mathematical properties
- Coverage of core algorithms ensuring correctness

### 📖 Documentation (`hdfm-framework/`)

Comprehensive guides for different use cases:

| Document | Purpose | Read This If... |
|----------|---------|-----------------|
| **[README.md](hdfm-framework/README.md)** | Complete technical documentation | You want full API details and implementation info |
| **[QUICKSTART.md](hdfm-framework/QUICKSTART.md)** | 5-minute getting started guide | You want to jump in quickly |
| **[PAPER_IMPLEMENTATION_REVIEW.md](hdfm-framework/PAPER_IMPLEMENTATION_REVIEW.md)** | Gap analysis of paper features | You want to know what's implemented vs. planned |
| **[KNOWN_LIMITATIONS.md](hdfm-framework/KNOWN_LIMITATIONS.md)** | Implementation status & roadmap | You need workarounds for missing features |
| **[CONTRIBUTING.md](hdfm-framework/CONTRIBUTING.md)** | Contribution guidelines | You want to contribute code or research |
| **[GITHUB_SETUP.md](hdfm-framework/GITHUB_SETUP.md)** | GitHub setup instructions | You're setting up your own fork |
| **[REPOSITORY_SUMMARY.md](hdfm-framework/REPOSITORY_SUMMARY.md)** | Complete repository overview | You want the big picture |

## How to Use This Framework

### For Researchers

**Reproduce paper results:**
```bash
cd hdfm-framework
python examples/synthetic_landscape_validation.py
```

**Apply to your landscape:**
```python
from hdfm import Landscape, Patch, build_dendritic_network

# Define your patches (from GIS data, field surveys, etc.)
patches = [
    Patch(id=0, x=1000, y=2000, area=50, quality=0.8),
    Patch(id=1, x=3000, y=4000, area=75, quality=0.9),
    # ... your patches
]

landscape = Landscape(patches)
network = build_dendritic_network(landscape)
```

**Use species-specific parameters:**
```python
from hdfm import SPECIES_GUILDS

# Get movement parameters for your focal species
large_carnivores = SPECIES_GUILDS['large_carnivores']
required_width = large_carnivores.required_width_for_success(0.85)
print(f"Need {required_width:.0f}m width for 85% movement success")
```

**Optimize corridor widths with budget constraints:**
```python
from hdfm import WidthOptimizer, SPECIES_GUILDS, build_dendritic_network

# Build dendritic network
network = build_dendritic_network(landscape)

# Optimize widths for small mammals with 25% landscape allocation
guild = SPECIES_GUILDS['small_mammals']
optimizer = WidthOptimizer(
    landscape=landscape,
    edges=network.edges,
    species_guild=guild,
    beta=0.25  # 25% landscape budget
)

result = optimizer.optimize()
print(f"Optimized entropy: {result.entropy:.3f}")
```

**Calculate entropy rate for heterogeneous landscapes:**
```python
from hdfm import calculate_entropy_rate, SPECIES_GUILDS

# Calculate H_rate with stationary distribution
guild = SPECIES_GUILDS['medium_mammals']
corridor_widths = {edge: 200 for edge in network.edges}

H_rate, components = calculate_entropy_rate(
    landscape=landscape,
    edges=network.edges,
    corridor_widths=corridor_widths,
    species_guild=guild
)

print(f"Entropy rate: {H_rate:.3f}")
print(f"Stationary distribution: {components['stationary_dist']}")
```

### For Conservation Planners

**Design climate-adaptive corridors:**
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

### For Developers

**Extend the framework:**
```python
from hdfm import DendriticOptimizer

# Create custom optimizer
class MyOptimizer(DendriticOptimizer):
    def custom_entropy_term(self, landscape, network):
        # Your custom entropy calculation
        pass
```

**Add new network topologies:**
```python
from hdfm import NetworkTopology

# Implement your topology
class MyTopology(NetworkTopology):
    def build_network(self, landscape):
        # Your network construction algorithm
        pass
```

## Key Features

### Core Capabilities

- 🌲 **Landscape entropy calculation** - Full entropy framework (H_mov + λ₁·C + λ₂·F + λ₃·D) with width-dependent terms
- 🌳 **Dendritic network construction** - Minimum spanning tree (MST) optimization
- 📏 **Width optimization** - Corridor width allocation under landscape budget constraints (20-30%)
- 🔄 **Dual entropy formulations** - H_rate with stationary distribution for heterogeneous landscapes
- ⏮️ **Backwards climate optimization** - Design from future to present
- 📊 **Comparative topology analysis** - Test dendritic vs. 5+ alternative topologies
- 🎨 **Visualization suite** - Network plots, entropy surfaces, optimization traces
- 📈 **Statistical validation** - Monte Carlo experiments, significance testing

### Validated Results

The framework reproduces theoretical predictions from the paper:

| Network Topology | Mean Entropy | vs. Dendritic | Significance |
|-----------------|--------------|---------------|--------------|
| **Dendritic (MST)** | **2.28** | — | — |
| Gabriel Graph | 2.87 | +25.9% | p < 0.001 |
| Delaunay Triangulation | 3.12 | +36.8% | p < 0.001 |
| k-Nearest Neighbors | 2.94 | +28.9% | p < 0.001 |
| Threshold Distance | 3.05 | +33.8% | p < 0.001 |

### Implementation Status

**Current Version: 0.2.0** (~75-80% of paper features)

✅ **Fully Implemented:**
- Dendritic network (MST) optimization
- Basic entropy framework (H_mov + penalties)
- Alternative topology comparisons
- Monte Carlo validation
- Backwards climate optimization structure
- Species-specific parameters (Table 2 from paper)
- **Width-dependent entropy calculations** with φ(w) term
- **Dual entropy formulations (H_rate)** with stationary distribution
- **Landscape allocation constraints (20-30%)** enforcement
- **Width optimization algorithms** under budget constraints

See [KNOWN_LIMITATIONS.md](hdfm-framework/KNOWN_LIMITATIONS.md) for complete status and workarounds.

## Technology Integration

The framework is designed for integration with modern conservation technologies:

- **AI/ML** - Landscape pattern recognition, corridor identification
- **GPS Geofencing** - Automated forestry equipment exclusion zones
- **Satellite Monitoring** - Sentinel-2, Landsat verification
- **Cloud Computing** - Scalable optimization for large landscapes (100+ patches)

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: 4GB minimum (8GB+ recommended for large landscapes)
- **Dependencies**: NumPy, SciPy, NetworkX, Matplotlib, Pandas, scikit-learn

See [requirements.txt](hdfm-framework/requirements.txt) for exact versions.

## Getting Help

### Documentation

1. Start with [QUICKSTART.md](hdfm-framework/QUICKSTART.md) (5 minutes)
2. Read the full [README.md](hdfm-framework/README.md) for API details
3. Check [KNOWN_LIMITATIONS.md](hdfm-framework/KNOWN_LIMITATIONS.md) for current restrictions
4. Review example code in `hdfm-framework/examples/`

### Support

- **Issues**: Open an issue on GitHub for bugs or feature requests
- **Email**: viridisnorthllc@gmail.com
- **Contributing**: See [CONTRIBUTING.md](hdfm-framework/CONTRIBUTING.md)

### Common Questions

**Q: Can I use this for my real landscape?**
A: Yes! Create `Patch` objects from your GIS data and use `build_dendritic_network()`.

**Q: How do I account for climate change?**
A: Use the `BackwardsOptimizer` with a `ClimateScenario` defining your climate trajectory.

**Q: What if I need feature X from the paper?**
A: Check [KNOWN_LIMITATIONS.md](hdfm-framework/KNOWN_LIMITATIONS.md) for status and workarounds.

**Q: How can I validate the results?**
A: Run `examples/synthetic_landscape_validation.py` to reproduce paper results.

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{hart2024hdfm,
  title={Hierarchical Dendritic Forest Management: A Vision for Technology-Enabled Landscape Conservation},
  author={Hart, Justin},
  journal={EcoEvoRxiv},
  year={2024}
}
```

## Contributing

We welcome contributions! Priority areas:

- **Empirical validation** - Real-world landscape testing
- **Climate model integration** - Species distribution model coupling
- **Technology integration** - GPS APIs, satellite data pipelines
- **Performance optimization** - Large-scale landscape handling

See [CONTRIBUTING.md](hdfm-framework/CONTRIBUTING.md) for guidelines.

## License

MIT License - See [LICENSE](LICENSE) file for details.

This is permissive open-source software - you can use it for research, commercial applications, or any other purpose with attribution.

## About

**Author**: Justin Hart
**Organization**: Viridis LLC
**Contact**: viridisnorthllc@gmail.com
**Version**: 0.1.0
**Status**: Research prototype / Active development

## Next Steps

Ready to get started?

1. **Install the framework** - Follow installation instructions above
2. **Run the examples** - See what the framework can do
3. **Read the docs** - Dive into [hdfm-framework/README.md](hdfm-framework/README.md)
4. **Apply to your landscape** - Start designing optimal corridors
5. **Contribute** - Help make HDFM better for everyone

---

**Note**: This is version 0.1.0 - a foundational release with core algorithms. See the [roadmap](hdfm-framework/KNOWN_LIMITATIONS.md) for upcoming features.

For complete technical documentation, API reference, and detailed examples, see **[hdfm-framework/README.md](hdfm-framework/README.md)**.
