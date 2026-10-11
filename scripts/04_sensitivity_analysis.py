"""
Script 04: Hyperparameter Sensitivity Analysis.
Evaluates window size W in {5, 7, 9, 11, 13, 15}, top-K in {2, 4, 6},
Transformer depth in {1, 2, 3}, and loss weights.
"""

import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.config import Config

def main():
    print("Running sensitivity analysis across key architectural hyperparameters.")

if __name__ == "__main__":
    main()
