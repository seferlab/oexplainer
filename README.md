# OExplainer (Orbit-based GNN Explainer) =

The core idea is to (1) train a GNN for node classification and (2) explain its predictions using **predefined
connected graphlets** and their **node orbits** (structural roles), producing both:

- **Model-level explanations**: which orbit/graphlet patterns the trained model relies on (globally).
- **Instance-level explanations**: which orbit/graphlet patterns contribute to a specific node’s prediction.

==
> The implementation follows the decomposition ideas described in the paper: decomposing the downstream model’s
> class scores on an orbit basis and producing vertex-level orbit contributions.

---

## Quickstart

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Run a synthetic demo (BA-Shapes)
```bash
python run_pipeline.py --dataset ba_shapes --model gin --epochs 200 --explain_node 42
```

Outputs:
- trained model checkpoints in `runs/.../checkpoints/`
- explanation artifacts in `runs/.../explanations/`
- a JSON report with **top orbits per class** (model-level) and **orbit contributions for a node** (instance-level)

---

## What’s implemented

### A) GNN training (node classification)
- `GCN`, `GAT`, `GIN` via **PyTorch Geometric** (recommended).
- Train/val/test splits, early stopping, reproducibility seeds.

### B) Graphlet & orbit machinery (2–4 nodes by default)
- Enumeration of **all non-isomorphic connected graphlets** up to `--max_graphlet_size` (default 4).
- **Orbit partitioning** computed by brute-force automorphism enumeration (OK for n ≤ 5).
- Orbit feature extraction per node using:
  - exact enumeration for small graphs
  - *sampling mode* for large graphs (PPI-like) via `--orbit_sampling` and `--num_samples`.

> Note: The original paper discusses graphlets of size 2–5 (up to 72 orbits).
> Exact counting at scale typically uses ORCA-like methods. This repo includes:
> - a pure-Python **exact** counter (best for small graphs / demos),
> - a **sampling** counter (best for big graphs),
> - a plug-in point to integrate ORCA binaries for full 2–5 orbit counts.

### C) OExplainer decomposition
Given:
- node embeddings `H` from the GNN encoder,
- logits `Z = H W + b` from the downstream classifier,
- orbit-count features `Phi` (node × orbit),

we fit a linear decomposition per class:
- `Z[:, m] ≈ Phi @ beta_m`

so:
- **class–orbit score** = `beta_m[k]` (model-level importance of orbit k for class m),
- **instance contribution** for node v: `Phi[v, k] * beta_m[k]`.

This aligns with the paper’s goal to decompose **task weights / class scores** on an orbit basis and obtain
**vertex-specific orbit scores**.

---

## Command-line interface

```bash
python run_pipeline.py   --dataset ba_shapes   --model gin   --max_graphlet_size 4   --orbit_sampling none   --explain_node 42
```

Key flags:
- `--max_graphlet_size`: 2–5 (default 4 for tractability)
- `--orbit_sampling`: `none` (exact) | `node` | `subgraph`
- `--num_samples`: sampling budget for large graphs
- `--explain_node`: node id to explain (instance-level)

---

## Repository layout

- `run_pipeline.py` — main entry point
- `oexplainer/models/` — GNN encoders/classifiers
- `oexplainer/explain/` — orbit counting + decomposition explainer
- `oexplainer/data/` — dataset loaders (synthetic included; bio loaders are stubs w/ docs)
- `oexplainer/utils/` — logging, seeding, IO, metrics
- `scripts/` — helper scripts
- `tests/` — basic unit tests

---

## Extending to PPI / KEGG-like datasets

The paper uses tissue-specific PPI graphs and KEGG pathway graphs. This repo includes
stubs and documentation to plug in your own graphs:
- `oexplainer/data/ppi_loader.py`
- `oexplainer/data/kegg_loader.py`

You can load graphs as:
- `edge_index` (PyG COO format) + `x` features + `y` labels.

---

## Citation

If you use this repository, please cite the paper.
