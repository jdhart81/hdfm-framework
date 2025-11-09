#!/usr/bin/env python3
"""
Quick installation test for HDFM Framework.
Run this to verify everything is working.
"""

def _check_imports() -> bool:
    """Helper to verify that key modules can be imported."""
    print("Testing imports...")
    try:
        from hdfm import (
            Landscape, SyntheticLandscape, Patch,
            calculate_entropy, build_dendritic_network,
            DendriticOptimizer, BackwardsOptimizer, ClimateScenario
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def _check_basic_optimization() -> bool:
    """Helper to validate dendritic network optimization."""
    print("\nTesting basic optimization...")
    try:
        from hdfm import SyntheticLandscape, build_dendritic_network
        
        # Create small landscape
        landscape = SyntheticLandscape(n_patches=5, random_seed=42)
        
        # Build network
        network = build_dendritic_network(landscape)
        
        # Calculate entropy
        H, components = network.entropy()
        
        # Verify
        assert H >= 0, "Entropy must be non-negative"
        assert len(network.edges) == 4, "Tree must have n-1 edges"
        assert network.is_dendritic(), "Network must be dendritic"
        
        print(f"✓ Basic optimization works (H = {H:.3f})")
        return True
    except Exception as e:
        print(f"✗ Optimization failed: {e}")
        return False


def _check_validation() -> bool:
    """Helper to validate comparison utilities."""
    print("\nTesting validation functions...")
    try:
        from hdfm import SyntheticLandscape, compare_network_topologies, NetworkTopology
        
        landscape = SyntheticLandscape(n_patches=5, random_seed=42)
        
        # Compare topologies
        results = compare_network_topologies(
            landscape,
            topologies=[NetworkTopology.DENDRITIC, NetworkTopology.GABRIEL]
        )
        
        # Verify results
        assert 'dendritic' in results, "Dendritic results missing"
        assert results['dendritic']['H_total'] > 0, "Entropy must be positive"
        
        print(f"✓ Validation functions work")
        return True
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


def test_imports():
    """Pytest wrapper for import smoke test."""
    assert _check_imports()


def test_basic_optimization():
    """Pytest wrapper for optimization smoke test."""
    assert _check_basic_optimization()


def test_validation():
    """Pytest wrapper for validation smoke test."""
    assert _check_validation()


def main():
    """Run all tests."""
    print("="*60)
    print("HDFM Framework Installation Test")
    print("="*60)
    
    tests = [
        _check_imports,
        _check_basic_optimization,
        _check_validation
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "="*60)
    if all(results):
        print("✓ ALL TESTS PASSED")
        print("Installation is working correctly!")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please check error messages above")
    print("="*60)
    
    return all(results)


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
