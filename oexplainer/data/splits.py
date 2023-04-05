from __future__ import annotations
import numpy as np

def make_splits(n: int, train_ratio: float = 0.6, val_ratio: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    train_mask = np.zeros(n, dtype=bool); train_mask[train_idx] = True
    val_mask = np.zeros(n, dtype=bool); val_mask[val_idx] = True
    test_mask = np.zeros(n, dtype=bool); test_mask[test_idx] = True
    return train_mask, val_mask, test_mask
