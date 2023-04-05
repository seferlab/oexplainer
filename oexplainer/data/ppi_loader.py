"""PPI loader stub.

The paper uses tissue-specific PPI graphs with node features and GO labels.
Since these datasets are typically shared on request, we keep this loader as a plug-in point.

Implement `load_custom_ppi(path)` to return a GraphData object.
You can store:
- edge_index: shape (2, E), int64
- x: (N, d), float32
- y: (N,), int64 labels (or multi-label converted to binary task)
- train/val/test masks
"""
from __future__ import annotations
from pathlib import Path
from .types import GraphData

def load_custom_ppi(path: str | Path) -> GraphData:
    raise NotImplementedError("Please implement your PPI data loading here.")
