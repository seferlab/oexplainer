from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam

from .gnn import NodeGNN
from oexplainer.data.types import GraphData
from oexplainer.utils.metrics import collect_metrics

def _to_torch(data: GraphData):
    x = torch.tensor(data.x, dtype=torch.float32)
    edge_index = torch.tensor(data.edge_index, dtype=torch.long)
    y = torch.tensor(data.y, dtype=torch.long)
    masks = {
        "train": torch.tensor(data.train_mask, dtype=torch.bool),
        "val": torch.tensor(data.val_mask, dtype=torch.bool),
        "test": torch.tensor(data.test_mask, dtype=torch.bool),
    }
    return x, edge_index, y, masks

def train_node_classifier(
    data: GraphData,
    model_name: str,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    outdir: Path,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = int(np.max(data.y)) + 1
    model = NodeGNN(in_dim=data.x.shape[1], hidden_dim=hidden_dim, out_dim=num_classes,
                    num_layers=num_layers, dropout=dropout, model=model_name).to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    x, edge_index, y, masks = _to_torch(data)
    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
    masks = {k: v.to(device) for k, v in masks.items()}

    best_val = -1.0
    best_state = None
    bad = 0
    history = []

    for ep in range(1, epochs+1):
        model.train()
        _, z = model(x, edge_index)
        loss = F.cross_entropy(z[masks["train"]], y[masks["train"]])
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            _, z = model(x, edge_index)
            logits = z.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            masks_np = {k: v.detach().cpu().numpy() for k, v in masks.items()}
            mets = collect_metrics(logits, y_np, masks_np)
            val_acc = mets["val"].acc

        history.append({"epoch": ep, "loss": float(loss.item()), "val_acc": float(val_acc)})

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final metrics
    model.eval()
    with torch.no_grad():
        _, z = model(x, edge_index)
        logits = z.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        masks_np = {k: v.detach().cpu().numpy() for k, v in masks.items()}
        mets = collect_metrics(logits, y_np, masks_np)

    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "model.pt")

    report = {
        "best_val_acc": float(best_val),
        "metrics": {k: {"acc": v.acc} for k, v in mets.items()},
        "history": history,
    }
    return model, report
