import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Local import
try:
    from src.pcap_parser import PcapParser
except:
    from pcap_parser import PcapParser

class FeatureBuilder:
    def __init__(self, target_ip=None):
        self.parser = PcapParser(target_ip)
        self.le = LabelEncoder()
        self.scaler = StandardScaler()
        self.cols = []

    def build_dataset(self, data_path):
        """
        Reads all .pcap files in subfolders of data_path.
        Each subfolder name = label.
        """
        samples = []
        
        if not os.path.exists(data_path):
            print(f"Error: {data_path} does not exist.")
            return pd.DataFrame()

        # Iterate over class folders
        for label in os.listdir(data_path):
            folder = os.path.join(data_path, label)
            if not os.path.isdir(folder):
                continue

            print(f"Processing class: {label}...")
            files = glob.glob(os.path.join(folder, "*.pcap"))
            
            if not files: 
                print(f"  No pcaps in {label}")
                continue

            for f in files:
                try:
                    # Parse packets
                    pkts = self.parser.parse(f)
                    
                    # Get features
                    row = self.parser.extract_stats(pkts)
                    
                    if not row:
                        continue
                        
                    # Add metadata
                    row['label'] = label
                    row['filename'] = os.path.basename(f)
                    
                    samples.append(row)
                except Exception as e:
                    print(f"  Skipping {f}: {e}")

        if not samples:
            print("No data extracted.")
            return pd.DataFrame()

        df = pd.DataFrame(samples)
        
        # Basic cleanup: fill NaN with 0 (happens if std=NaN for single packet)
        df.fillna(0, inplace=True)
        
        print(f"Dataset ready: {df.shape}")
        return df

    def get_train_test(self, df, test_split=0.2):
        # Drop non-numeric cols
        ignore = ['label', 'filename']
        X_df = df.drop(columns=[c for c in ignore if c in df.columns])
        
        # Save columns for later
        self.cols = X_df.columns.tolist()

        X = X_df.values
        y = df['label'].values

        # Encode labels to ints
        y_enc = self.le.fit_transform(y)
        classes = self.le.classes_

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=test_split, stratify=y_enc, random_state=42
        )

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, classes

if __name__ == "__main__":
    pass
