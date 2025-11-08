# Hierarchical Dendritic Forest Management (HDFM) Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A computational framework for entropy-minimizing corridor network optimization with backwards climate adaptation.

## Overview

This repository provides reference implementations of the algorithms and methods for designing optimal forest corridor networks using information-theoretic principles and climate-adaptive planning.

Key innovations:
- **Information-theoretic optimization**: Formulates corridor design as entropy minimization
- **Dendritic network topology**: Tree structures that minimize landscape entropy
- **Backwards climate optimization**: Designs from 2100 projections to present implementation
- **Technology integration**: Leverages AI, GPS geofencing, and satellite monitoring

## Quick Start

```bash
# Clone repository
git clone https://github.com/jdhart81/hdfm-framework.git
cd hdfm-framework/hdfm-framework

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Run validation
python examples/synthetic_landscape_validation.py
```

## Documentation

For complete documentation, installation instructions, examples, and API reference, see:

**[hdfm-framework/README.md](hdfm-framework/README.md)**

Additional resources:
- [Quick Start Guide](hdfm-framework/QUICKSTART.md)
- [Contributing Guidelines](hdfm-framework/CONTRIBUTING.md)
- [GitHub Setup](hdfm-framework/GITHUB_SETUP.md)
- [Repository Summary](hdfm-framework/REPOSITORY_SUMMARY.md)

## Core Features

- 🌲 Landscape entropy calculation with movement, connectivity, and topology terms
- 🔀 Dendritic network construction via minimum spanning trees
- ⏮️ Backwards optimization algorithm for climate-adaptive corridor design
- 📊 Synthetic landscape validation reproducing theoretical predictions
- 🎨 Visualization tools for networks and optimization traces

## Citation

```bibtex
@article{hart2024hdfm,
  title={Hierarchical Dendritic Forest Management: A Vision for Technology-Enabled Landscape Conservation},
  author={Hart, Justin},
  journal={EcoEvoRxiv},
  year={2024}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

**Justin Hart**
Viridis LLC
viridisnorthllc@gmail.com
