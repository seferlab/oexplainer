from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import itertools
import networkx as nx

def _adj_bitstring(G: nx.Graph, nodes: List[int]) -> str:
    # canonical label: min adjacency upper-tri bitstring over all permutations
    n = len(nodes)
    idx = list(range(n))
    best = None
    for perm in itertools.permutations(idx):
        bits = []
        for i in range(n):
            for j in range(i+1, n):
                u = nodes[perm[i]]
                v = nodes[perm[j]]
                bits.append('1' if G.has_edge(u, v) else '0')
        s = ''.join(bits)
        if best is None or s < best:
            best = s
    return best or ""

def _aut_group_orbits(G: nx.Graph, nodes: List[int]) -> List[List[int]]:
    """Compute orbit partition of nodes via automorphisms (brute-force; OK for n<=5)."""
    n = len(nodes)
    # adjacency matrix for fast comparison
    A = [[0]*n for _ in range(n)]
    for i,u in enumerate(nodes):
        for j,v in enumerate(nodes):
            if i<j and G.has_edge(u,v):
                A[i][j]=A[j][i]=1

    auts = []
    for perm in itertools.permutations(range(n)):
        ok = True
        for i in range(n):
            for j in range(n):
                if A[i][j] != A[perm[i]][perm[j]]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            auts.append(perm)

    # orbit: i ~ j if exists automorphism mapping i->j
    parent = list(range(n))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]
            a=parent[a]
        return a
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra

    for perm in auts:
        for i in range(n):
            union(i, perm[i])

    groups: Dict[int,List[int]] = {}
    for i in range(n):
        r=find(i)
        groups.setdefault(r,[]).append(i)
    # return orbits as lists of indices (0..n-1)
    return list(groups.values())

@dataclass(frozen=True)
class GraphletSpec:
    n: int
    canon: str  # canonical adjacency bitstring
    orbits: List[List[int]]  # orbit partition over node indices 0..n-1

def enumerate_connected_graphlets(max_n: int = 4) -> List[GraphletSpec]:
    """Enumerate all non-isomorphic connected graphlets up to max_n.

    We brute-force all graphs on n nodes and keep connected ones,
    de-duplicated by canonical adjacency bitstring.
    """
    specs: List[GraphletSpec] = []
    for n in range(2, max_n+1):
        nodes = list(range(n))
        all_edges = [(i,j) for i in range(n) for j in range(i+1,n)]
        seen = set()
        for mask in range(1, 1<<len(all_edges)):
            G = nx.Graph()
            G.add_nodes_from(nodes)
            for b,(u,v) in enumerate(all_edges):
                if (mask>>b) & 1:
                    G.add_edge(u,v)
            if not nx.is_connected(G):
                continue
            canon = _adj_bitstring(G, nodes)
            if canon in seen:
                continue
            seen.add(canon)
            orbits = _aut_group_orbits(G, nodes)
            specs.append(GraphletSpec(n=n, canon=canon, orbits=orbits))
    return specs

def orbit_index_map(specs: List[GraphletSpec]) -> Tuple[Dict[Tuple[int,str,int], int], List[Dict]]:
    """Assign a global orbit index for each (n, canon, local_orbit_id)."""
    mapping: Dict[Tuple[int,str,int], int] = {}
    meta: List[Dict] = []
    idx = 0
    for spec in specs:
        for oid, orbit in enumerate(spec.orbits):
            mapping[(spec.n, spec.canon, oid)] = idx
            meta.append({
                "orbit_index": idx,
                "graphlet_n": spec.n,
                "graphlet_canon": spec.canon,
                "local_orbit_id": oid,
                "orbit_nodes": orbit,
            })
            idx += 1
    return mapping, meta
