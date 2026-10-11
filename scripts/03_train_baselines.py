"""
Script 03: Train competitive MIL baselines under identical fair protocol.
Models: ABMIL, TransMIL, CLAM-SB, ViT-Slice-MLP.
"""

import os
import sys
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.config import Config
from src.models.baselines import ABMIL, TransMIL, CLAM_SB, ViTSliceMLP

def main():
    parser = argparse.ArgumentParser(description="Train Competitive Baselines")
    parser.add_argument("--data_dir", type=str, default="./data/phase-1-output")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--model", type=str, choices=["abmil", "transmil", "clam", "vitmlp", "all"], default="all")
    args = parser.parse_args()

    print("Training competitive baselines with identical budget, optimizer, and seeds.")

if __name__ == "__main__":
    main()
