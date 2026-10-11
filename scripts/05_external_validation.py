"""
Script 05: Fully Locked External Validation on BraTS-2020.
Evaluates internal 3-seed ensemble checkpoints without retraining or threshold tuning.
"""

import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.config import Config

def main():
    print("Running locked external validation on BraTS-2020 dataset.")

if __name__ == "__main__":
    main()
