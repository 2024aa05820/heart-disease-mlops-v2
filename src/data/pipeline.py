"""Data preprocessing pipeline for heart disease prediction."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib


# Feature columns for the dataset
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Numerical features that need scaling
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Categorical/binary features
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load heart disease dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    return df


def preprocess_data(
    df: pd.DataFrame,
    target_column: str = "target"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the heart disease dataset.
    
    Args:
        df: Raw DataFrame
        target_column: Name of the target column
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Handle missing values (represented as '?' in UCI dataset)
    df = df.replace('?', np.nan)
    
    # Convert columns to appropriate types
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle target variable (convert to binary: 0 = no disease, 1 = disease)
    if target_column in df.columns:
        df[target_column] = (df[target_column] > 0).astype(int)
    
    # Fill missing values with median for numerical columns
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Fill missing values with mode for categorical columns
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    
    # Separate features and target
    feature_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
    X = df[feature_cols]
    y = df[target_column] if target_column in df.columns else None
    
    return X, y


class DataPipeline:
    """
    Complete data preprocessing pipeline for heart disease prediction.
    
    This class provides:
    - Data loading and cleaning
    - Feature preprocessing (scaling, imputation)
    - Train/test splitting
    - Pipeline serialization
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the data pipeline.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.preprocessing_pipeline = None
        self._create_preprocessing_pipeline()
    
    def _create_preprocessing_pipeline(self) -> None:
        """Create the sklearn preprocessing pipeline."""
        # Numerical preprocessing: impute + scale
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing: impute only (already encoded)
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        
        # Combine transformers
        self.preprocessing_pipeline = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, NUMERICAL_FEATURES),
                ('cat', categorical_transformer, CATEGORICAL_FEATURES)
            ],
            remainder='passthrough'
        )
    
    def load_and_preprocess(
        self,
        file_path: str,
        target_column: str = "target"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and preprocess data from file.
        
        Args:
            file_path: Path to the CSV file
            target_column: Name of the target column
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = load_data(file_path)
        return preprocess_data(df, target_column)
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Fit the preprocessing pipeline and transform the data.
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional, not used for fitting)
            
        Returns:
            Transformed feature array
        """
        return self.preprocessing_pipeline.fit_transform(X)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using the fitted preprocessing pipeline.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Transformed feature array
        """
        if self.preprocessing_pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
        return self.preprocessing_pipeline.transform(X)
    
    def split_data(
        self,
        X: np.ndarray,
        y: pd.Series,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature array
            y: Target array
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
    
    def save_pipeline(self, file_path: str) -> None:
        """
        Save the preprocessing pipeline to a file.
        
        Args:
            file_path: Path to save the pipeline
        """
        if self.preprocessing_pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessing_pipeline, file_path)
    
    def load_pipeline(self, file_path: str) -> None:
        """
        Load a preprocessing pipeline from a file.
        
        Args:
            file_path: Path to the saved pipeline
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {file_path}")
        
        self.preprocessing_pipeline = joblib.load(file_path)
    
    def get_feature_names(self) -> list:
        """Get the names of features after transformation."""
        return NUMERICAL_FEATURES + CATEGORICAL_FEATURES

