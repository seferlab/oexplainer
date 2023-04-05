from __future__ import annotations
from .types import GraphData
from . import synthetic
from .ppi_loader import load_custom_ppi
from .kegg_loader import load_custom_kegg

def load_dataset(name: str) -> GraphData:
    if name == "erdos_renyi":
        return synthetic.erdos_renyi(seed=0)
    if name == "ba_shapes":
        return synthetic.ba_shapes(seed=0)
    if name == "ba_community":
        return synthetic.ba_community(seed=0)
    if name == "custom_ppi":
        # Edit this path or pass via env var in your own workflows.
        return load_custom_ppi("data/ppi")
    if name == "custom_kegg":
        return load_custom_kegg("data/kegg")
    raise ValueError(f"Unknown dataset: {name}")
