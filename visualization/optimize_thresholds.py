#!/usr/bin/env python3
"""Optimize classification thresholds from the command line.

This script provides a convenient way to optimize thresholds for instrument
detection to maximize either F1 score or balanced accuracy.

Example usage:
    python visualization/optimize_thresholds.py lightning_logs/version_X/checkpoints/best.ckpt
    python visualization/optimize_thresholds.py --metric balanced lightning_logs/version_X/checkpoints/best.ckpt
"""

import pathlib
import sys

# Add parent directory to path for imports
parent_dir = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from visualization.threshold_optimization import main

if __name__ == "__main__":
    main()
