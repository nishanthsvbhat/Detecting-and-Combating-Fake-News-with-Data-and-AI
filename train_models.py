#!/usr/bin/env python3
"""
Quick Start Training Script
Train Word2Vec embeddings and neural models on ISOT Fake News dataset

Usage:
    python train_models.py --epochs 50 --batch_size 32 --sample_size 5000
"""

import argparse
import os
import sys

from training_pipeline import run_full_pipeline


def main():
    parser = argparse.ArgumentParser(
        description='Train fake news detection models'
    )
    parser.add_argument(
        '--true_csv',
        default='True.csv',
        help='Path to True.csv (real news)'
    )
    parser.add_argument(
        '--fake_csv',
        default='Fake.csv',
        help='Path to Fake.csv (fake news)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=None,
        help='Sample size per category (None = use all data)'
    )
    parser.add_argument(
        '--output_dir',
        default='model_artifacts',
        help='Output directory for models (default: model_artifacts)'
    )
    
    args = parser.parse_args()
    
    # Verify data files exist
    if not os.path.exists(args.true_csv):
        print(f"❌ Error: {args.true_csv} not found")
        print("Please download ISOT dataset: https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/")
        sys.exit(1)
    
    if not os.path.exists(args.fake_csv):
        print(f"❌ Error: {args.fake_csv} not found")
        sys.exit(1)
    
    print("=" * 70)
    print("FAKE NEWS DETECTION - FULL TRAINING PIPELINE")
    print("=" * 70)
    print(f"\nTraining Configuration:")
    print(f"  True news file: {args.true_csv}")
    print(f"  Fake news file: {args.fake_csv}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sample size: {args.sample_size if args.sample_size else 'All data'}")
    print(f"  Output directory: {args.output_dir}\n")
    
    try:
        # Run pipeline
        pipeline = run_full_pipeline(
            true_csv=args.true_csv,
            fake_csv=args.fake_csv,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sample_size=args.sample_size
        )
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nModels saved to: {args.output_dir}/")
        print(f"  - word2vec_model (embeddings)")
        print(f"  - ANN_best_model.pth")
        print(f"  - CNN1D_best_model.pth")
        print(f"  - BiLSTM_best_model.pth")
        print(f"  - pipeline_config.json")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
