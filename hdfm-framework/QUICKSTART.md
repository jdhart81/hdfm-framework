# HDFM Framework - Quick Start Guide

Get started with the HDFM Framework in 5 minutes.

## Installation (2 minutes)

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/hdfm-framework.git
cd hdfm-framework

# Install with dependencies
pip install -e .
```

That's it! The framework is now installed.

## Run First Example (3 minutes)

### 1. Basic Landscape Optimization

Create a file `test_hdfm.py`:

```python
from hdfm import SyntheticLandscape, build_dendritic_network, plot_network
import matplotlib.pyplot as plt

# Create a landscape
landscape = SyntheticLandscape(n_patches=15, extent=10000)

# Optimize corridor network
network = build_dendritic_network(landscape)

# Calculate entropy
H_total, components = network.entropy()

print(f"Optimized network entropy: {H_total:.3f}")
print(f"Movement entropy: {components['H_mov']:.3f}")
print(f"Number of corridors: {len(network.edges)}")

# Visualize
plot_network(landscape, network)
plt.show()
```

Run it:
```bash
python test_hdfm.py
```

You should see:
- Console output with entropy values
- A plot showing patches and corridors

### 2. Reproduce Paper Validation

Run the full synthetic landscape validation:

```bash
python examples/synthetic_landscape_validation.py --n-landscapes 100
```

This will:
- Generate 100 random 15-patch landscapes
- Compare 5 network topologies
- Validate dendritic optimality
- Save results to `validation_results/`

Expected runtime: 2-5 minutes

Expected output:
```
HDFM SYNTHETIC LANDSCAPE VALIDATION
================================================================================

Validation parameters:
  - Landscapes: 100
  - Patches per landscape: 15
  - Extent: 10.0 km
  
NETWORK TOPOLOGY VALIDATION RESULTS
================================================================================
Topology             Mean H       Std H        vs. Dendritic        p-value   
--------------------------------------------------------------------------------
Dendritic (MST)      2.284        0.043        (reference)          —         
Gabriel              2.873        0.062        +25.8%               <0.0001   
Delaunay             3.118        0.078        +36.5%               <0.0001   
Knn                  2.936        0.071        +28.6%               <0.0001   
Threshold            3.046        0.087        +33.4%               <0.0001   

✓ VALIDATION PASSED: Dendritic networks minimize entropy
```

### 3. Backwards Climate Optimization

Run climate-adaptive optimization demo:

```bash
python examples/backwards_optimization_demo.py --mode compare
```

This demonstrates:
- Forward optimization (current conditions only)
- Backwards optimization (climate-adaptive)
- Comparison of approaches

You'll see networks optimized for climate resilience.

## Common Use Cases

### Case 1: Optimize Your Own Landscape

```python
from hdfm import Landscape, Patch, build_dendritic_network

# Define your patches
patches = [
    Patch(id=0, x=1000, y=2000, area=50, quality=0.8),
    Patch(id=1, x=3000, y=4000, area=75, quality=0.9),
    # ... add your patches
]

# Create landscape
landscape = Landscape(patches)

# Optimize
network = build_dendritic_network(landscape)

# Analyze
H_total, components = network.entropy()
print(f"Optimal network has {len(network.edges)} corridors")
print(f"Total entropy: {H_total:.3f}")
```

### Case 2: Compare Network Topologies

```python
from hdfm import SyntheticLandscape, compare_network_topologies

landscape = SyntheticLandscape(n_patches=20)

results = compare_network_topologies(landscape)

for topology, metrics in results.items():
    print(f"{topology}: H = {metrics['H_total']:.3f}, "
          f"edges = {metrics['n_edges']}")
```

### Case 3: Climate-Adaptive Design

```python
from hdfm import (
    SyntheticLandscape,
    BackwardsOptimizer,
    ClimateScenario
)

# Define landscape
landscape = SyntheticLandscape(n_patches=15)

# Define climate scenario
scenario = ClimateScenario(
    years=[2025, 2050, 2075, 2100],
    temperature_changes=[0, 1.5, 2.5, 3.5],
    precipitation_changes=[0, -5, -10, -15]
)

# Optimize backwards from 2100
optimizer = BackwardsOptimizer(landscape, scenario)
result = optimizer.optimize()

print(f"Climate-adaptive network: H = {result.entropy:.3f}")
```

### Case 4: Validate on Multiple Landscapes

```python
from hdfm import monte_carlo_validation, statistical_comparison

# Run validation
results = monte_carlo_validation(
    n_landscapes=50,
    n_patches=15
)

# Statistical comparison
comparison = statistical_comparison(results)

# Print results
for topology, stats in comparison.items():
    print(f"{topology}: {stats['percent_difference']:.1f}% "
          f"higher than dendritic (p={stats['p_value']:.4f})")
```

## Next Steps

### Learn More
- Read the full README.md
- Check out CONTRIBUTING.md for development
- See examples/ directory for more demos

### Extend the Framework
- Add your own entropy terms
- Implement alternative optimization algorithms
- Integrate with real GIS data
- Couple with species distribution models

### Get Help
- Open an issue: https://github.com/YOUR-USERNAME/hdfm-framework/issues
- Read the docs: See docs/ directory
- Email: viridisnorthllc@gmail.com

## Common Issues

### Import Error
```
ModuleNotFoundError: No module named 'hdfm'
```
**Solution**: Run `pip install -e .` from repository root

### Missing Dependencies
```
ModuleNotFoundError: No module named 'networkx'
```
**Solution**: Install requirements: `pip install -r requirements.txt`

### Plots Don't Show
```python
# Add this at the end of your script
import matplotlib.pyplot as plt
plt.show()
```

### Out of Memory (Large Landscapes)
For landscapes with 100+ patches:
```python
# Use smaller validation sets
results = monte_carlo_validation(
    n_landscapes=10,  # Reduce from 100
    n_patches=50
)
```

## Performance Tips

- **Small landscapes (< 20 patches)**: Very fast, <1 second
- **Medium landscapes (20-50 patches)**: Fast, 1-5 seconds
- **Large landscapes (50-100 patches)**: Moderate, 5-30 seconds
- **Very large landscapes (100+ patches)**: Slow, 30+ seconds

For large-scale applications, consider:
- Hierarchical decomposition
- Parallel processing (use multiprocessing)
- Algorithm tuning (reduce max_iterations)

## Verification

Run tests to verify installation:

```bash
pytest tests/ -v
```

All tests should pass. If any fail, please open an issue.

## What's Next?

You're ready to:
1. ✅ Run the framework on synthetic landscapes
2. ✅ Reproduce paper results
3. ✅ Understand the algorithms

Now try:
- Apply to your real landscape data
- Integrate with climate models
- Publish your own research using HDFM

Happy optimizing! 🌲
