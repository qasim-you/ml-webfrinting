import sys
import os
import argparse
import numpy as np

sys.path.append(os.getcwd())

from src.feature_engineering import FeatureBuilder
from src.classifiers import ClassifierSuite
from src.defense_buflo import BuFLOShim  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to data folder')
    parser.add_argument('--target_ip', default=None, help='Filter for this IP')
    parser.add_argument('--out', default='results', help='Output folder')
    
    args = parser.parse_args()

    # Setup
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # 1. Load Data
    print(">>> Step 1: Loading & Feature Extraction")
    fb = FeatureBuilder(target_ip=args.target_ip)
    df = fb.build_dataset(args.data)
    
    if df.empty:
        print("No stats extracted. Check your data path/PCAPs.")
        return

    # Save features just in case
    df.to_csv(os.path.join(args.out, "features.csv"), index=False)

    # 2. Prep for ML
    print("\n>>> Step 2: Preparing Datasets")
    X_train, X_test, y_train, y_test, classes = fb.get_train_test(df)
    
    print(f"Train size: {len(X_train)}")
    print(f"Test size:  {len(X_test)}")
    print(f"Classes: {classes}")

    # 3. Train & Test
    print("\n>>> Step 3: Training Models")
    suite = ClassifierSuite()
    suite.train_all(X_train, y_train)

    print("\n>>> Step 4: Evaluation")
    res = suite.test_all(X_test, y_test)

    # Save logic
    for name, metrics in res.items():
        # Save CM plot
        cm_file = os.path.join(args.out, f"cm_{name}.png")
        suite.plot_cm(metrics['cm'], classes, title=f"{name} Confusion Matrix", out_file=cm_file)

    print(f"\nDone! Results saved to {args.out}/")

if __name__ == "__main__":
    main()
