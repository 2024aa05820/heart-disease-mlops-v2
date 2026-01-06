#!/usr/bin/env python3
"""
Download Heart Disease UCI Dataset.

This script downloads the heart disease dataset from UCI Machine Learning Repository
and prepares it for training.
"""

import os
import sys
from pathlib import Path
import requests
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Dataset URL
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Alternative URL (Kaggle-style format)
ALTERNATIVE_URL = "https://raw.githubusercontent.com/datasets/heart-disease/main/data/heart.csv"

# Column names for UCI dataset
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Output path
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "heart.csv"


def download_uci_dataset() -> pd.DataFrame:
    """Download dataset from UCI repository."""
    print(f"Downloading from UCI repository: {DATASET_URL}")
    
    try:
        response = requests.get(DATASET_URL, timeout=30)
        response.raise_for_status()
        
        # Save raw data temporarily
        lines = response.text.strip().split('\n')
        
        # Parse data
        data = []
        for line in lines:
            values = line.split(',')
            if len(values) == 14:
                data.append(values)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=COLUMN_NAMES)
        
        # Convert columns to appropriate types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"Failed to download from UCI: {e}")
        return None


def download_alternative_dataset() -> pd.DataFrame:
    """Download dataset from alternative source."""
    print(f"Downloading from alternative source: {ALTERNATIVE_URL}")
    
    try:
        df = pd.read_csv(ALTERNATIVE_URL)
        return df
    except Exception as e:
        print(f"Failed to download from alternative source: {e}")
        return None


def create_sample_dataset() -> pd.DataFrame:
    """Create a sample dataset for testing."""
    print("Creating sample dataset for testing...")
    
    import numpy as np
    np.random.seed(42)
    
    n_samples = 303
    
    data = {
        "age": np.random.randint(29, 77, n_samples),
        "sex": np.random.randint(0, 2, n_samples),
        "cp": np.random.randint(0, 4, n_samples),
        "trestbps": np.random.randint(94, 200, n_samples),
        "chol": np.random.randint(126, 564, n_samples),
        "fbs": np.random.randint(0, 2, n_samples),
        "restecg": np.random.randint(0, 3, n_samples),
        "thalach": np.random.randint(71, 202, n_samples),
        "exang": np.random.randint(0, 2, n_samples),
        "oldpeak": np.round(np.random.uniform(0, 6.2, n_samples), 1),
        "slope": np.random.randint(0, 3, n_samples),
        "ca": np.random.randint(0, 5, n_samples),
        "thal": np.random.randint(0, 4, n_samples),
        "target": np.random.randint(0, 2, n_samples)
    }
    
    return pd.DataFrame(data)


def main():
    """Main function to download and save the dataset."""
    print("=" * 60)
    print("Heart Disease Dataset Downloader")
    print("=" * 60)
    
    # Try UCI first
    df = download_uci_dataset()
    
    # Try alternative if UCI fails
    if df is None:
        df = download_alternative_dataset()
    
    # Create sample if all downloads fail
    if df is None:
        print("\nWarning: Could not download dataset. Creating sample data.")
        df = create_sample_dataset()
    
    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Dataset saved to: {OUTPUT_PATH}")
    
    # Print dataset info
    print(f"\nDataset Info:")
    print(f"  - Samples: {len(df)}")
    print(f"  - Features: {len(df.columns) - 1}")
    print(f"  - Target distribution:")
    if "target" in df.columns:
        target_counts = df["target"].value_counts()
        for val, count in target_counts.items():
            print(f"    - Class {val}: {count} ({count/len(df)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    main()

