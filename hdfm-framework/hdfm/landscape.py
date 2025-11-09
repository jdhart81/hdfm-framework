"""
Landscape representation and graph construction for HDFM framework.

This module provides classes for representing forest landscapes as graphs,
where nodes are habitat patches and edges are potential corridors.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import networkx as nx


@dataclass
class Patch:
    """
    Represents a single habitat patch in the landscape.

    Attributes:
        id: Unique patch identifier
        x: X-coordinate (meters)
        y: Y-coordinate (meters)
        area: Patch area (hectares)
        quality: Habitat quality [0, 1]
        species_richness: Optional species count
        carrying_capacity: Optional population capacity
        population: Optional local population size (nᵢ) for genetic calculations

    Invariants:
        - 0 <= quality <= 1
        - area > 0
        - species_richness >= 0 if not None
        - carrying_capacity >= 0 if not None
        - population >= 0 if not None
    """
    id: int
    x: float
    y: float
    area: float
    quality: float = 1.0
    species_richness: Optional[int] = None
    carrying_capacity: Optional[int] = None
    population: Optional[float] = None
    
    def __post_init__(self):
        """Validate patch attributes."""
        assert 0 <= self.quality <= 1, f"Quality must be in [0,1], got {self.quality}"
        assert self.area > 0, f"Area must be positive, got {self.area}"
        if self.species_richness is not None:
            assert self.species_richness >= 0, "Species richness must be non-negative"
        if self.carrying_capacity is not None:
            assert self.carrying_capacity >= 0, "Carrying capacity must be non-negative"
        if self.population is not None:
            assert self.population >= 0, "Population must be non-negative"
    
    def distance_to(self, other: 'Patch') -> float:
        """Calculate Euclidean distance to another patch."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class Landscape:
    """
    Represents a forest landscape as a graph of habitat patches.
    
    The landscape is represented as:
    - Nodes: Habitat patches (Patch objects)
    - Edges: Potential corridors with distance-based costs
    
    Invariants:
    - All patches have unique IDs
    - Graph is fully connected (all patches reachable)
    - Edge weights represent Euclidean distances
    - No self-loops
    """
    
    def __init__(self, patches: List[Patch]):
        """
        Initialize landscape from list of patches.
        
        Args:
            patches: List of Patch objects
            
        Invariants enforced:
        - At least 2 patches required
        - All patch IDs are unique
        """
        assert len(patches) >= 2, "Landscape must contain at least 2 patches"
        
        # Check unique IDs
        patch_ids = [p.id for p in patches]
        assert len(patch_ids) == len(set(patch_ids)), "Patch IDs must be unique"
        
        self.patches = patches
        self.n_patches = len(patches)
        
        # Build graph representation
        self.graph = self._build_graph()
        
        # Verify connectivity invariant
        assert nx.is_connected(self.graph), "Landscape graph must be connected"
    
    def _build_graph(self) -> nx.Graph:
        """
        Build NetworkX graph representation.
        
        Returns:
            NetworkX Graph with patches as nodes and distances as edge weights
        """
        G = nx.Graph()
        
        # Add nodes
        for patch in self.patches:
            G.add_node(
                patch.id,
                patch=patch,
                x=patch.x,
                y=patch.y,
                area=patch.area,
                quality=patch.quality
            )
        
        # Add edges (complete graph with distance weights)
        for i, patch_i in enumerate(self.patches):
            for patch_j in self.patches[i+1:]:
                distance = patch_i.distance_to(patch_j)
                G.add_edge(
                    patch_i.id,
                    patch_j.id,
                    weight=distance,
                    distance=distance
                )
        
        return G
    
    def get_patch(self, patch_id: int) -> Patch:
        """Get patch by ID."""
        for patch in self.patches:
            if patch.id == patch_id:
                return patch
        raise ValueError(f"Patch {patch_id} not found")
    
    def distance_matrix(self) -> np.ndarray:
        """
        Compute pairwise distance matrix.
        
        Returns:
            n x n matrix where D[i,j] is distance between patches i and j
            
        Invariants:
        - D is symmetric: D[i,j] == D[j,i]
        - Diagonal is zero: D[i,i] == 0
        - All distances positive: D[i,j] > 0 for i != j
        """
        n = self.n_patches
        D = np.zeros((n, n))
        
        for i, patch_i in enumerate(self.patches):
            for j, patch_j in enumerate(self.patches):
                if i != j:
                    D[i, j] = patch_i.distance_to(patch_j)
        
        # Verify invariants
        assert np.allclose(D, D.T), "Distance matrix must be symmetric"
        assert np.allclose(np.diag(D), 0), "Diagonal must be zero"
        assert np.all(D[~np.eye(n, dtype=bool)] > 0), "Off-diagonal distances must be positive"
        
        return D
    
    def adjacency_matrix(self, edges: List[Tuple[int, int]]) -> np.ndarray:
        """
        Build adjacency matrix from edge list.
        
        Args:
            edges: List of (patch_i, patch_j) tuples
            
        Returns:
            n x n binary adjacency matrix
            
        Invariants:
        - A is symmetric: A[i,j] == A[j,i]
        - Diagonal is zero: A[i,i] == 0
        - Binary values: A[i,j] in {0, 1}
        """
        n = self.n_patches
        A = np.zeros((n, n), dtype=int)
        
        # Map patch IDs to indices
        id_to_idx = {patch.id: i for i, patch in enumerate(self.patches)}
        
        for (i, j) in edges:
            idx_i = id_to_idx[i]
            idx_j = id_to_idx[j]
            A[idx_i, idx_j] = 1
            A[idx_j, idx_i] = 1
        
        # Verify invariants
        assert np.allclose(A, A.T), "Adjacency matrix must be symmetric"
        assert np.all(np.diag(A) == 0), "Diagonal must be zero"
        assert np.all(np.isin(A, [0, 1])), "Matrix must be binary"
        
        return A
    
    def total_corridor_length(self, edges: List[Tuple[int, int]]) -> float:
        """
        Calculate total corridor length for given edge set.
        
        Args:
            edges: List of (patch_i, patch_j) corridor edges
            
        Returns:
            Total corridor length in meters
        """
        return sum(
            self.graph[i][j]['distance'] 
            for (i, j) in edges
        )
    
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get landscape bounding box: (min_x, max_x, min_y, max_y)."""
        xs = [p.x for p in self.patches]
        ys = [p.y for p in self.patches]
        return (min(xs), max(xs), min(ys), max(ys))


class SyntheticLandscape(Landscape):
    """
    Generate synthetic landscapes for validation experiments.
    
    Creates random patch configurations with controlled properties
    for testing and algorithm validation.
    """
    
    def __init__(
        self,
        n_patches: int,
        extent: float = 10000.0,
        min_area: float = 10.0,
        max_area: float = 100.0,
        quality_mean: float = 0.7,
        quality_std: float = 0.15,
        random_seed: Optional[int] = None
    ):
        """
        Generate synthetic landscape with random patches.
        
        Args:
            n_patches: Number of habitat patches
            extent: Landscape extent in meters (square landscape)
            min_area: Minimum patch area (hectares)
            max_area: Maximum patch area (hectares)
            quality_mean: Mean habitat quality
            quality_std: Std dev of habitat quality
            random_seed: Random seed for reproducibility
            
        Invariants:
        - n_patches >= 2
        - extent > 0
        - 0 < min_area < max_area
        - 0 <= quality_mean <= 1
        - quality_std >= 0
        """
        assert n_patches >= 2, "Need at least 2 patches"
        assert extent > 0, "Extent must be positive"
        assert 0 < min_area < max_area, "Area bounds invalid"
        assert 0 <= quality_mean <= 1, "Quality mean must be in [0,1]"
        assert quality_std >= 0, "Quality std must be non-negative"
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate random patches
        patches = []
        for i in range(n_patches):
            x = np.random.uniform(0, extent)
            y = np.random.uniform(0, extent)
            area = np.random.uniform(min_area, max_area)
            quality = np.clip(
                np.random.normal(quality_mean, quality_std),
                0.0, 1.0
            )
            
            patches.append(Patch(
                id=i,
                x=x,
                y=y,
                area=area,
                quality=quality
            ))
        
        # Initialize parent Landscape
        super().__init__(patches)
        
        # Store generation parameters
        self.extent = extent
        self.generation_params = {
            'n_patches': n_patches,
            'extent': extent,
            'min_area': min_area,
            'max_area': max_area,
            'quality_mean': quality_mean,
            'quality_std': quality_std,
            'random_seed': random_seed
        }
    
    @classmethod
    def from_template(cls, template: str = 'default', **kwargs):
        """
        Create synthetic landscape from predefined template.
        
        Templates:
        - 'default': 15 patches, 10km extent
        - 'small': 8 patches, 5km extent
        - 'large': 50 patches, 20km extent
        - 'dense': 30 patches, 10km extent
        """
        templates = {
            'default': {'n_patches': 15, 'extent': 10000},
            'small': {'n_patches': 8, 'extent': 5000},
            'large': {'n_patches': 50, 'extent': 20000},
            'dense': {'n_patches': 30, 'extent': 10000}
        }
        
        if template not in templates:
            raise ValueError(f"Unknown template '{template}'. Choose from {list(templates.keys())}")
        
        params = templates[template]
        params.update(kwargs)
        
        return cls(**params)
