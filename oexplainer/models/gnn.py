from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GATConv, GINConv
except Exception as e:
    GCNConv = GATConv = GINConv = None

class NodeGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float, model: str):
        super().__init__()
        if GCNConv is None:
            raise ImportError("torch-geometric is required. Install torch-geometric>=2.5.")

        self.model = model
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        if model == "gcn":
            self.convs.append(GCNConv(in_dim, hidden_dim))
            for _ in range(num_layers-2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif model == "gat":
            heads = 4
            self.convs.append(GATConv(in_dim, hidden_dim//heads, heads=heads, dropout=dropout))
            for _ in range(num_layers-2):
                self.convs.append(GATConv(hidden_dim, hidden_dim//heads, heads=heads, dropout=dropout))
            self.convs.append(GATConv(hidden_dim, hidden_dim//heads, heads=heads, dropout=dropout))
        elif model == "gin":
            def mlp(din, dout):
                return nn.Sequential(nn.Linear(din, dout), nn.ReLU(), nn.Linear(dout, dout))
            self.convs.append(GINConv(mlp(in_dim, hidden_dim)))
            for _ in range(num_layers-1):
                self.convs.append(GINConv(mlp(hidden_dim, hidden_dim)))
        else:
            raise ValueError(f"Unknown model: {model}")

        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns embeddings H and logits Z
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        h = x
        z = self.classifier(h)
        return h, z
