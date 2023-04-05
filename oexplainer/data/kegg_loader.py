"""KEGG pathway loader stub.

The paper uses pathway-specific metabolic networks based on KEGG.
Provide a pathway graph as GraphData.

You may parse KGML/TSV exports, or preprocessed edge lists.
"""
from __future__ import annotations
from pathlib import Path
from .types import GraphData

def load_custom_kegg(path: str | Path) -> GraphData:
    raise NotImplementedError("Please implement your KEGG data loading here.")
