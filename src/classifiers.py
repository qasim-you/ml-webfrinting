import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold

class ClassifierSuite:
    def __init__(self):
        self.models = {
            "NaiveBayes": GaussianNB(),
            "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.trained = {}

    def train_all(self, X_train, y_train):
        for name, clf in self.models.items():
            print(f"Training {name}...")
            clf.fit(X_train, y_train)
            self.trained[name] = clf

    def test_all(self, X_test, y_test):
        results = {}
        
        # Make sure inputs are clean 1D arrays
        y_test = np.array(y_test).flatten().astype(int)

        for name, clf in self.trained.items():
            preds = clf.predict(X_test)
            preds = np.array(preds).flatten().astype(int)
            
            # Metrics
            acc = accuracy_score(y_test, preds)
            # Weighted avg is usually best for multiclass
            p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
            
            cm = confusion_matrix(y_test, preds)

            results[name] = {
                'acc': acc,
                'p': p,
                'r': r,
                'f1': f1,
                'cm': cm
            }
            
            print(f"\nModel: {name}")
            print(f"  Accuracy: {acc:.2%}")
            print(f"  F1-Score: {f1:.2f}")
            
        return results

    def plot_cm(self, cm, classes, title="CM", out_file=None):
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.title(title)
        
        if out_file:
            plt.savefig(out_file)
            plt.close()
        # else: plt.show() # blocking

if __name__ == "__main__":
    pass
