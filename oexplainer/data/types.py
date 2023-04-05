from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class GraphData:
    # PyG-style edge_index (2, E) integer array
    edge_index: np.ndarray
    x: np.ndarray
    y: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    name: str = "graph"
    # Optional: store original networkx graph for orbit counting
    nx_graph: Optional[object] = None
