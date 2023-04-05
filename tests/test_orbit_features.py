from oexplainer.data.synthetic import ba_shapes
from oexplainer.explain.orbit_features import OrbitFeatureExtractor

def test_orbit_features_shape():
    data = ba_shapes(seed=0)
    ext = OrbitFeatureExtractor(max_graphlet_size=3, k_hop=1, sampling="subgraph", num_samples=200)
    Phi, meta = ext.fit_transform(data)
    assert Phi.shape[0] == data.x.shape[0]
    assert Phi.shape[1] == meta["num_orbits"]
