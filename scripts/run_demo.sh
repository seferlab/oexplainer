#!/usr/bin/env bash
set -euo pipefail
python run_pipeline.py --dataset ba_shapes --model gin --epochs 50 --explain_node 0 --orbit_sampling subgraph --num_samples 1000
