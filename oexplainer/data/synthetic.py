from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import networkx as nx

from .types import GraphData
from .splits import make_splits

def _to_edge_index(G: nx.Graph) -> np.ndarray:
    edges = np.array(list(G.edges()), dtype=np.int64)
    if edges.size == 0:
        return np.zeros((2,0), dtype=np.int64)
    # undirected -> add both directions
    rev = edges[:, ::-1]
    all_e = np.vstack([edges, rev])
    return all_e.T

def erdos_renyi(n: int = 300, p: float = 0.02, num_classes: int = 2, seed: int = 0) -> GraphData:
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    x = np.ones((n, 1), dtype=np.float32)
    # random labels by degree bins as a simple proxy
    deg = np.array([d for _, d in G.degree()], dtype=np.int64)
    y = (deg > np.median(deg)).astype(np.int64) % num_classes
    train_mask, val_mask, test_mask = make_splits(n, seed=seed)
    return GraphData(edge_index=_to_edge_index(G), x=x, y=y,
                     train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                     name="erdos_renyi", nx_graph=G)

def ba_shapes(n: int = 300, m: int = 2, num_houses: int = 80, seed: int = 0) -> GraphData:
    """A lightweight BA-Shapes-like generator.

    This is not a byte-for-byte reproduction of the original BA-Shapes from GNNExplainer,
    but provides a similar setup: BA backbone + attached house motifs.
    """
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)

    # House motif: 5 nodes (square + roof)
    house_edges = [(0,1),(1,2),(2,3),(3,0),(1,3),(2,4),(3,4)]
    # attach houses to random backbone nodes
    base_nodes = list(G.nodes())
    next_id = n
    motif_nodes = []
    for _ in range(num_houses):
        anchor = int(rng.choice(base_nodes))
        mapping = {i: next_id+i for i in range(5)}
        next_id += 5
        for u,v in house_edges:
            G.add_edge(mapping[u], mapping[v])
        # connect one motif node to anchor
        G.add_edge(anchor, mapping[0])
        motif_nodes.append(list(mapping.values()))

    N = G.number_of_nodes()
    x = np.ones((N, 1), dtype=np.float32)

    # labels: backbone=0, motif nodes split into 3 classes by role (roof vs base etc.)
    y = np.zeros((N,), dtype=np.int64)
    for nodes in motif_nodes:
        # nodes[4] = roof
        y[nodes[4]] = 3
        y[nodes[2]] = 2
        y[nodes[3]] = 2
        y[nodes[0]] = 1
        y[nodes[1]] = 1

    train_mask, val_mask, test_mask = make_splits(N, seed=seed)
    return GraphData(edge_index=_to_edge_index(G), x=x, y=y,
                     train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                     name="ba_shapes", nx_graph=G)

def ba_community(seed: int = 0) -> GraphData:
    """Two BA-Shapes graphs connected with random inter-edges (toy BA-Community)."""
    g1 = ba_shapes(seed=seed)
    g2 = ba_shapes(seed=seed+1)
    import networkx as nx
    G1 = g1.nx_graph
    G2 = g2.nx_graph
    # relabel nodes of G2
    offset = G1.number_of_nodes()
    G2r = nx.relabel_nodes(G2, {i: i+offset for i in G2.nodes()})
    G = nx.compose(G1, G2r)
    rng = np.random.default_rng(seed)
    # add random inter-edges
    for _ in range(200):
        u = int(rng.integers(0, offset))
        v = int(rng.integers(offset, G.number_of_nodes()))
        G.add_edge(u, v)

    N = G.number_of_nodes()
    x = rng.standard_normal((N, 10)).astype(np.float32)
    y = np.concatenate([g1.y, g2.y + 4]).astype(np.int64)

    train_mask, val_mask, test_mask = make_splits(N, seed=seed)
    return GraphData(edge_index=_to_edge_index(G), x=x, y=y,
                     train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                     name="ba_community", nx_graph=G)
