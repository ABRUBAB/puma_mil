"""
Script 06: Generate publication-grade figures at 600 DPI.
"""

import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.visualization import plot_confusion_matrix, plot_calibration_curve

def main():
    print("Generating publication figures at 600 DPI.")

if __name__ == "__main__":
    main()
