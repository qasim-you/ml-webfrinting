import sys
import os
import argparse
import pandas as pd
import numpy as np

# Add src to python path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering import FeatureEngineer
from src.classifiers import TrafficClassifier
from src.evaluation import plot_confusion_matrix

def run_website_fingerprinting(data_dir: str, output_dir: str):
    print("="*60)
    print("WEBSITE FINGERPRINTING EXTENSION")
    print(f"Data Dir: {data_dir}")
    print("="*60)
    
    # 1. Feature Extraction
    print("Extracting features from website traffic traces...")
    engineer = FeatureEngineer() # target_ip might be dynamic for websites, so we process all logic or rely on clean captures
    
    df = engineer.create_dataset_from_directory(data_dir)
    
    if df.empty:
        print("No data found. Ensure directory structure is data/website_name/*.pcap")
        return

    # 2. Prepare Data
    print(f"Dataset: {len(df)} samples from {df['label'].nunique()} websites.")
    X_train, X_test, y_train, y_test, class_names = engineer.prepare_for_training(df)
    
    # 3. Train Classifiers
    print("Training classifiers...")
    classifier = TrafficClassifier()
    classifier.train(X_train, y_train)
    
    # 4. Evaluate
    print("Evaluating models...")
    results = classifier.evaluate(X_test, y_test, class_names)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, 'website_fingerprinting_results.txt')
    
    with open(report_file, 'w') as f:
        f.write("Website Fingerprinting Results\n")
        f.write("==============================\n")
        f.write(f"Websites: {class_names}\n\n")
        for name, res in results.items():
            f.write(f"Model: {name}\n")
            f.write(f"Accuracy: {res['accuracy']:.4f}\n")
            f.write(f"Precision: {res['precision']:.4f}\n")
            f.write(f"Recall: {res['recall']:.4f}\n")
            f.write(f"F1 Score: {res['f1_score']:.4f}\n")
            f.write("-" * 20 + "\n")
            
    print(f"Results saved to {report_file}")
    
    # Plot Confusion Matrix for Random Forest (usually best)
    rf_cm = results['Random Forest']['confusion_matrix']
    plot_path = os.path.join(output_dir, 'cm_website_rf.png')
    plot_confusion_matrix(rf_cm, class_names, title='Website Fingerprinting (RF)', save_path=plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to website pcap data')
    parser.add_argument('--output_dir', type=str, default='./output_web', help='Output directory')
    args = parser.parse_args()
    
    run_website_fingerprinting(args.data_dir, args.output_dir)
