from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List
import numpy as np

import torch

from oexplainer.data.types import GraphData

def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

@dataclass
class OExplainer:
    topk: int = 10
    ridge: float = 1e-3

    def _get_logits_and_embeddings(self, model: torch.nn.Module, data: GraphData) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        x = torch.tensor(data.x, dtype=torch.float32)
        edge_index = torch.tensor(data.edge_index, dtype=torch.long)
        device = next(model.parameters()).device
        with torch.no_grad():
            h, z = model(x.to(device), edge_index.to(device))
        return _to_numpy(h), _to_numpy(z)

    def _fit_beta(self, Phi: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Fit beta per class: z[:,m] ~ Phi @ beta_m (ridge)."""
        # Add bias term by augmenting Phi with ones
        N = Phi.shape[0]
        X = np.concatenate([Phi, np.ones((N,1), dtype=Phi.dtype)], axis=1)
        # ridge closed form: (X^T X + λI)^{-1} X^T z
        XtX = X.T @ X
        lamI = self.ridge * np.eye(XtX.shape[0], dtype=XtX.dtype)
        B = np.linalg.solve(XtX + lamI, X.T @ z)  # (K+1, C)
        return B  # last row is bias-like

    def explain(
        self,
        model: torch.nn.Module,
        data: GraphData,
        orbit_features: np.ndarray,
        orbit_meta: Dict[str, Any],
        explain_node: int,
    ) -> Dict[str, Any]:
        H, Z = self._get_logits_and_embeddings(model, data)
        Phi = orbit_features.astype(np.float32)
        B = self._fit_beta(Phi, Z)  # (K+1, C)
        beta = B[:-1, :]  # (K, C)
        bias = B[-1, :]   # (C,)

        num_classes = Z.shape[1]
        # model-level: top orbits per class by |beta|
        top_orbits = {}
        for m in range(num_classes):
            idx = np.argsort(-np.abs(beta[:, m]))[: self.topk]
            top_orbits[str(m)] = [
                {"orbit_index": int(i), "score": float(beta[i, m])} for i in idx
            ]

        # instance-level: contributions for a node for predicted class
        pred_class = int(Z[explain_node].argmax())
        contrib = Phi[explain_node, :] * beta[:, pred_class]
        top_idx = np.argsort(-np.abs(contrib))[: self.topk]
        instance = {
            "node": int(explain_node),
            "pred_class": pred_class,
            "logits": Z[explain_node].tolist(),
            "top_orbit_contributions": [
                {"orbit_index": int(i), "contribution": float(contrib[i]),
                 "phi": float(Phi[explain_node, i]), "beta": float(beta[i, pred_class])}
                for i in top_idx
            ],
        }

        report = {
            "decomposition": {
                "ridge": self.ridge,
                "beta_shape": list(beta.shape),
                "bias_shape": list(bias.shape),
            },
            "model_level": {
                "top_orbits_per_class": top_orbits,
            },
            "instance_level": instance,
            "orbit_meta_summary": {
                "num_orbits": orbit_meta.get("num_orbits"),
                "max_graphlet_size": orbit_meta.get("max_graphlet_size"),
                "k_hop": orbit_meta.get("k_hop"),
                "sampling": orbit_meta.get("sampling"),
                "num_samples": orbit_meta.get("num_samples"),
            },
            "orbit_meta": orbit_meta.get("orbit_meta", []),
        }
        return report
