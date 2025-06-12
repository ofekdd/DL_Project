#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml

from training.shared_backbone_train import main as train_main

def main():
    parser = argparse.ArgumentParser(description="Train SharedBackbonePANNs model")
    parser.add_argument("--config", type=str, default="configs/shared_backbone.yaml", 
                        help="Path to YAML config file")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Override max samples for faster training")
    parser.add_argument("--epochs", type=int, default=None, 
                        help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--limit_val", type=float, default=None,
                        help="Override limit_val_batches for faster validation")

    args = parser.parse_args()

    # Load configuration
    print(f"📝 Loading configuration from {args.config}")
    with open(args.config, 'r') as file:
        cfg = yaml.safe_load(file)

    # Apply command-line overrides
    if args.max_samples is not None:
        cfg['max_samples'] = args.max_samples
        print(f"⚠️ Overriding max_samples: {args.max_samples}")

    if args.epochs is not None:
        cfg['num_epochs'] = args.epochs
        print(f"⚠️ Overriding num_epochs: {args.epochs}")

    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
        print(f"⚠️ Overriding batch_size: {args.batch_size}")

    if args.limit_val is not None:
        cfg['limit_val_batches'] = args.limit_val
        print(f"⚠️ Overriding limit_val_batches: {args.limit_val}")

    # Print final configuration
    print("\n🚀 Training with configuration:")
    for key, value in cfg.items():
        print(f"   {key}: {value}")

    # Start training
    print("\n🏃 Starting training...")
    train_main(cfg)

if __name__ == "__main__":
    main()
