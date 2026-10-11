"""
Script 02: Train PUMA-MIL ensemble across 3 random seeds (42, 123, 456).
Usage:
  python scripts/02_train_puma_mil.py --data_dir ./data/phase-1-output --out_dir ./outputs
"""

import os
import sys
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.config import Config
from src.models.puma_mil import PUMAMILLite

def main():
    parser = argparse.ArgumentParser(description="Train PUMA-MIL Model Ensemble")
    parser.add_argument("--data_dir", type=str, default="./data/phase-1-output", help="Path to Phase 1 precomputed features")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Random seeds for ensemble")
    args = parser.parse_args()

    cfg = Config()
    cfg.work_dir = args.out_dir
    cfg.seeds = args.seeds
    cfg.ensure_dirs()

    print(f"Starting PUMA-MIL training on device: {cfg.device}")
    print(f"Seeds: {cfg.seeds}")
    print(f"Window size: W={cfg.train_window}, Top-K: K={cfg.top_k}, EMA momentum: {cfg.proto_momentum}")

if __name__ == "__main__":
    main()
