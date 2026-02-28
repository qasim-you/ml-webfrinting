import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from typing import List, Dict, Any

def plot_confusion_matrix(cm: np.ndarray, classes: List[str], 
                         title: str = 'Confusion Matrix', save_path: str = None):
    """
    Plots a confusion matrix heatmap from a pre-computed matrix.
    
    Args:
        cm: Confusion matrix (2D array).
        classes: List of class names.
        title: Title of the plot.
        save_path: Path to save the figure (optional).
    """
    # cm is passed directly
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

def plot_feature_importance(model, feature_names: List[str], top_n: int = 10, 
                          save_path: str = None):
    """
    Plots feature importance for tree-based models (Random Forest, AdaBoost).
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not provide feature importances.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Feature Importances")
    plt.bar(range(top_n), importances[indices], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_defense_tradeoff(accuracies: List[float], overheads: List[float], 
                        labels: List[str] = None, save_path: str = None):
    """
    Plots the trade-off between Accuracy and Bandwidth Overhead under defense.
    
    Args:
        accuracies: List of accuracy scores.
        overheads: List of overhead ratios (e.g., 0.1 for 10%).
        labels: Optional labels for points (e.g., defense configurations).
    """
    plt.figure(figsize=(8, 6))
    plt.plot(overheads, accuracies, marker='o', linestyle='-')
    
    if labels:
        for i, txt in enumerate(labels):
            plt.annotate(txt, (overheads[i], accuracies[i]), 
                         textcoords="offset points", xytext=(0,10), ha='center')
            
    plt.xlabel('Bandwidth Overhead Ratio')
    plt.ylabel('Classification Accuracy')
    plt.title('Defense Evaluation: Accuracy vs Overhead')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def generate_report(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]):
    """
    Prints a detailed classification report.
    """
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

if __name__ == "__main__":
    # Test stub
    y_true = np.array([0, 1, 0, 1, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 2, 1])
    classes = ['ClassA', 'ClassB', 'ClassC']
    
    # Save instead of show to prevent blocking
    plot_path = 'test_confusion_matrix.png'
    plot_confusion_matrix(y_true, y_pred, classes, save_path=plot_path)
    generate_report(y_true, y_pred, classes)
    
    # Clean up
    if os.path.exists(plot_path):
        os.remove(plot_path)

