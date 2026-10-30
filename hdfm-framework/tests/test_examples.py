"""
Tests that the documented examples keep running.

The synthetic validation example is the worked example in the repository's
QUICKSTART.md. These tests check that everything it imports is exported and
that a fixed seed gives the same table twice.
"""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import hdfm

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plotting_helpers_are_exported():
    for name in ("plot_topology_comparison", "plot_climate_trajectory", "create_validation_figure"):
        assert hasattr(hdfm, name)
        assert name in hdfm.__all__


def test_examples_import():
    _load("synthetic_landscape_validation")
    _load("backwards_optimization_demo")


def test_validation_example_is_deterministic():
    import matplotlib.pyplot as plt

    example = _load("synthetic_landscape_validation")
    runs = [
        example.run_full_validation(n_landscapes=5, random_seed=42, save_figures=False)
        for _ in range(2)
    ]
    plt.close("all")
    first, second = runs
    assert first["dendritic_optimal"] == second["dendritic_optimal"]
    for topology, stats in first["comparison"].items():
        assert stats["topology_mean"] == second["comparison"][topology]["topology_mean"]
        assert stats["reference_mean"] == second["comparison"][topology]["reference_mean"]


def test_topology_table_reproduces():
    """The corrected manuscript table regenerates from a fixed seed.

    Checks the first five manuscript-stated landscapes against the values in
    results/topology_manuscript-stated.csv, which examples/reproduce_topology_table.py
    writes. A change here means the published corrected table must be regenerated.
    """
    import csv

    import numpy as np

    example = _load("reproduce_topology_table")
    res = example.run("manuscript-stated", n_landscapes=5, n_patches=15, seed=42, optimize_widths=False)
    stored = {}
    with open(EXAMPLES.parent / "results" / "topology_manuscript-stated.csv") as f:
        for row in csv.DictReader(f):
            stored[(int(row["landscape"]), row["network"])] = float(row["H_shannon"])
    for network, metrics in res.items():
        for i, value in enumerate(metrics["H_shannon"]):
            assert np.isclose(value, stored[(42 + i, network)], rtol=1e-4)
    # The MST has the lowest local choice entropy of the networks compared ...
    for network in ("Gabriel", "Delaunay", "k-NN (k=3)", "MST + 3 loops"):
        assert np.all(res["MST"]["H_shannon"] < res[network]["H_shannon"])
    # ... and, as defined (an average over patches), a 15-patch tree cannot exceed ln(28/15).
    assert np.all(res["MST"]["H_shannon"] <= np.log(28 / 15) + 1e-9)
