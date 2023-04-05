#!/usr/bin/env python3
"""Main entry point to train a node-classification GNN and run OExplainer.

Examples:
  python run_pipeline.py --dataset ba_shapes --model gin --epochs 200 --explain_node 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

from oexplainer.utils.seed import seed_everything
from oexplainer.utils.io import ensure_dir, save_json
from oexplainer.data.loaders import load_dataset
from oexplainer.models.train import train_node_classifier
from oexplainer.explain.orbit_features import OrbitFeatureExtractor
from oexplainer.explain.oexplainer import OExplainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ba_shapes",
                   choices=["ba_shapes", "ba_community", "erdos_renyi", "custom_ppi", "custom_kegg"],
                   help="Dataset to run.")
    p.add_argument("--model", type=str, default="gin", choices=["gcn", "gat", "gin"])
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)

    # Orbit/graphlet extraction
    p.add_argument("--max_graphlet_size", type=int, default=4, choices=[2,3,4,5])
    p.add_argument("--orbit_sampling", type=str, default="none", choices=["none", "node", "subgraph"],
                   help="Sampling mode for orbit features. Use sampling for large graphs.")
    p.add_argument("--num_samples", type=int, default=2000, help="Sampling budget when orbit_sampling != none.")
    p.add_argument("--k_hop", type=int, default=2, help="Neighborhood hop limit for subgraph enumeration/sampling.")

    # Explanation
    p.add_argument("--explain_node", type=int, default=0, help="Node index to explain (instance-level).")
    p.add_argument("--topk_orbits", type=int, default=10)

    # Output
    p.add_argument("--outdir", type=str, default="runs")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    outdir = Path(args.outdir) / f"{args.dataset}_{args.model}_seed{args.seed}"
    ckpt_dir = ensure_dir(outdir / "checkpoints")
    exp_dir = ensure_dir(outdir / "explanations")

    data = load_dataset(args.dataset)

    model, train_report = train_node_classifier(
        data=data,
        model_name=args.model,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        outdir=ckpt_dir,
    )

    # Orbit features (Phi)
    orbit_extractor = OrbitFeatureExtractor(
        max_graphlet_size=args.max_graphlet_size,
        k_hop=args.k_hop,
        sampling=args.orbit_sampling,
        num_samples=args.num_samples,
    )
    Phi, orbit_meta = orbit_extractor.fit_transform(data)

    explainer = OExplainer(topk=args.topk_orbits)
    report = explainer.explain(
        model=model,
        data=data,
        orbit_features=Phi,
        orbit_meta=orbit_meta,
        explain_node=args.explain_node,
    )

    save_json(exp_dir / "train_report.json", train_report)
    save_json(exp_dir / "oexplainer_report.json", report)

    print(f"[OK] Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
