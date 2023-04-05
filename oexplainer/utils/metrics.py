from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class SplitMetrics:
    acc: float

def accuracy(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float((pred[mask] == y[mask]).mean())

def collect_metrics(logits: np.ndarray, y: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, SplitMetrics]:
    pred = logits.argmax(axis=1)
    out = {}
    for k, m in masks.items():
        out[k] = SplitMetrics(acc=accuracy(pred, y, m))
    return out
