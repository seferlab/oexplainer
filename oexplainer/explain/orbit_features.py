from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import itertools
import numpy as np
import networkx as nx
from tqdm import tqdm

from oexplainer.data.types import GraphData
from .graphlets import enumerate_connected_graphlets, orbit_index_map, _adj_bitstring

def _ensure_nx(data: GraphData) -> nx.Graph:
    if data.nx_graph is not None:
        return data.nx_graph
    # build from edge_index
    G = nx.Graph()
    edge_index = data.edge_index
    G.add_nodes_from(range(data.x.shape[0]))
    for u, v in edge_index.T:
        G.add_edge(int(u), int(v))
    return G

def _k_hop_nodes(G: nx.Graph, v: int, k: int) -> List[int]:
    nodes = set([v])
    frontier = set([v])
    for _ in range(k):
        nxt = set()
        for u in frontier:
            nxt.update(G.neighbors(u))
        nxt -= nodes
        nodes |= nxt
        frontier = nxt
        if not frontier:
            break
    return list(nodes)

@dataclass
class OrbitFeatureExtractor:
    max_graphlet_size: int = 4
    k_hop: int = 2
    sampling: str = "none"  # none|node|subgraph
    num_samples: int = 2000
    seed: int = 0

    def fit_transform(self, data: GraphData) -> Tuple[np.ndarray, Dict]:
        G = _ensure_nx(data)
        specs = enumerate_connected_graphlets(self.max_graphlet_size)
        orbit_map, orbit_meta = orbit_index_map(specs)
        K = len(orbit_meta)
        N = G.number_of_nodes()
        Phi = np.zeros((N, K), dtype=np.float32)

        rng = np.random.default_rng(self.seed)

        if self.sampling == "none":
            # exact enumeration around each node; best for small graphs
            for v in tqdm(range(N), desc="Orbit features (exact)"):
                neigh = _k_hop_nodes(G, v, self.k_hop)
                for n in range(2, self.max_graphlet_size+1):
                    # choose n-1 other nodes from neighborhood
                    others = [u for u in neigh if u != v]
                    if len(others) < n-1:
                        continue
                    for combo in itertools.combinations(others, n-1):
                        nodes = [v, *combo]
                        H = G.subgraph(nodes)
                        if not nx.is_connected(H):
                            continue
                        # local node ordering must be stable for orbit detection
                        local_nodes = list(nodes)
                        canon = _adj_bitstring(H, local_nodes)
                        # determine which local orbit v is in
                        spec = None
                        # Instead of searching spec object, we use orbit_map keys (n, canon, oid)
                        # Find orbit by comparing v's orbit partition via automorphisms on the induced subgraph
                        # We'll compute local orbits quickly by brute-force perms (n<=5)
                        from .graphlets import _aut_group_orbits
                        orbits = _aut_group_orbits(H, local_nodes)
                        v_local = 0  # v is at index 0 in local_nodes
                        local_orbit_id = None
                        for oid, orb in enumerate(orbits):
                            if v_local in orb:
                                local_orbit_id = oid
                                break
                        if local_orbit_id is None:
                            continue
                        key = (n, canon, local_orbit_id)
                        if key not in orbit_map:
                            continue
                        Phi[v, orbit_map[key]] += 1.0

        else:
            # sampling for large graphs
            # node sampling: pick nodes v and sample subgraphs in its k-hop neighborhood
            if self.sampling not in {"node", "subgraph"}:
                raise ValueError(f"Unknown sampling mode: {self.sampling}")

            for _ in tqdm(range(self.num_samples), desc=f"Orbit features (sampling={self.sampling})"):
                v = int(rng.integers(0, N))
                neigh = _k_hop_nodes(G, v, self.k_hop)
                neigh = [u for u in neigh if u != v]
                if len(neigh) == 0:
                    continue
                n = int(rng.integers(2, self.max_graphlet_size+1))
                if len(neigh) < n-1:
                    continue
                combo = list(rng.choice(neigh, size=n-1, replace=False))
                nodes = [v, *combo]
                H = G.subgraph(nodes)
                if not nx.is_connected(H):
                    continue
                local_nodes = list(nodes)
                canon = _adj_bitstring(H, local_nodes)
                from .graphlets import _aut_group_orbits
                orbits = _aut_group_orbits(H, local_nodes)
                v_local = 0
                local_orbit_id = None
                for oid, orb in enumerate(orbits):
                    if v_local in orb:
                        local_orbit_id = oid
                        break
                if local_orbit_id is None:
                    continue
                key = (n, canon, local_orbit_id)
                if key not in orbit_map:
                    continue
                Phi[v, orbit_map[key]] += 1.0

            # normalize sampling counts to be comparable across nodes
            # (optional) here we just leave raw sample counts.

        meta = {
            "max_graphlet_size": self.max_graphlet_size,
            "k_hop": self.k_hop,
            "sampling": self.sampling,
            "num_samples": self.num_samples,
            "num_orbits": K,
            "orbit_meta": orbit_meta,
        }
        return Phi, meta
