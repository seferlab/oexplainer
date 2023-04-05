import networkx as nx
from oexplainer.explain.graphlets import enumerate_connected_graphlets, orbit_index_map

def test_enumeration_small():
    specs = enumerate_connected_graphlets(3)
    assert len(specs) > 0
    mapping, meta = orbit_index_map(specs)
    assert len(mapping) == len(meta)
    assert len(meta) > 0
