# Contributing to HDFM Framework

Thank you for your interest in contributing to the Hierarchical Dendritic Forest Management (HDFM) Framework!

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in [GitHub Issues](https://github.com/yourusername/hdfm-framework/issues)
2. If not, create a new issue with:
   - Clear description of the problem or suggestion
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - Your environment (OS, Python version, package versions)

### Contributing Code

1. **Fork the repository** and clone your fork
2. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following our coding standards (see below)
4. **Add tests** for new functionality
5. **Run tests** to ensure nothing breaks:
   ```bash
   pytest tests/
   ```
6. **Commit your changes** with clear commit messages:
   ```bash
   git commit -m "Add: Brief description of your changes"
   ```
7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Create a Pull Request** with:
   - Description of changes
   - Reference to related issues
   - Any breaking changes highlighted

## Coding Standards

### Style Guidelines

- Follow [PEP 8](https://pep8.org/) for Python code
- Use [Black](https://black.readthedocs.io/) for code formatting:
  ```bash
  black hdfm/ examples/ tests/
  ```
- Use [flake8](https://flake8.pycqa.org/) for linting:
  ```bash
  flake8 hdfm/ examples/ tests/
  ```

### Documentation

- Add docstrings to all functions, classes, and modules
- Use [NumPy style](https://numpydoc.readthedocs.io/) for docstrings
- Include type hints for function parameters and returns
- Update README.md if adding new features

### Testing

- Write unit tests for all new functionality
- Maintain >80% code coverage
- Use pytest fixtures for common test setups
- Test edge cases and error conditions

Example test structure:
```python
def test_function_name():
    """Test description."""
    # Setup
    input_data = create_test_data()
    
    # Execute
    result = function_under_test(input_data)
    
    # Verify
    assert result.meets_invariant(), "Invariant violated"
    assert result.expected_property, "Expected property not satisfied"
```

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hdfm-framework.git
   cd hdfm-framework
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

4. Run tests:
   ```bash
   pytest tests/ -v --cov=hdfm
   ```

## Priority Contribution Areas

We especially welcome contributions in:

### 1. Empirical Validation
- Real-world landscape testing
- Field data integration
- Comparison with existing corridor networks
- Species-specific validation

### 2. Climate Model Integration
- Coupling with species distribution models
- Climate projection data pipelines
- Dynamic habitat quality modeling
- Migration pathway optimization

### 3. Genetic Connectivity
- Integration with landscape genetics simulations
- Effective population size estimation
- Gene flow modeling
- Genetic diversity metrics

### 4. Technology Integration
- GPS geofencing API integration
- Satellite data processing pipelines
- Automated monitoring workflows
- Mobile app prototypes

### 5. Economic Analysis
- Cost-benefit modeling
- Timber production optimization
- Carbon credit calculations
- Payment for ecosystem services frameworks

### 6. Alternative Algorithms
- Metaheuristic optimizers (genetic algorithms, simulated annealing)
- Multi-objective optimization frameworks
- Stochastic programming approaches
- Machine learning integration

## Code Review Process

All contributions go through code review:

1. **Automated checks** must pass:
   - Tests pass
   - Code coverage maintained
   - Style checks pass
   - No merge conflicts

2. **Reviewer feedback** will focus on:
   - Correctness and robustness
   - Code clarity and documentation
   - Performance implications
   - Adherence to framework principles

3. **Iteration** may be needed based on feedback

4. **Approval and merge** once checks pass and reviewers approve

## Communication

- **GitHub Discussions**: General questions, ideas, showcases
- **GitHub Issues**: Bug reports, feature requests, specific problems
- **Email**: viridisnorthllc@gmail.com for private inquiries

## Attribution

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes
- Academic citations (for significant contributions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Don't hesitate to ask! We're happy to help new contributors get started.

Thank you for helping advance landscape-scale conservation technology! 🌲
