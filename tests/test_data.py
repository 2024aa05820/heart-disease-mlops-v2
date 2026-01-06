"""Unit tests for data processing module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import (
    DataPipeline,
    load_data,
    preprocess_data,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)


class TestDataPipeline:
    """Test cases for DataPipeline class."""
    
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample data for testing."""
        np.random.seed(42)
        n = 50
        
        return pd.DataFrame({
            "age": np.random.randint(30, 70, n),
            "sex": np.random.randint(0, 2, n),
            "cp": np.random.randint(0, 4, n),
            "trestbps": np.random.randint(100, 180, n),
            "chol": np.random.randint(150, 400, n),
            "fbs": np.random.randint(0, 2, n),
            "restecg": np.random.randint(0, 3, n),
            "thalach": np.random.randint(100, 180, n),
            "exang": np.random.randint(0, 2, n),
            "oldpeak": np.round(np.random.uniform(0, 5, n), 1),
            "slope": np.random.randint(0, 3, n),
            "ca": np.random.randint(0, 4, n),
            "thal": np.random.randint(0, 4, n),
            "target": np.random.randint(0, 2, n)
        })
    
    @pytest.fixture
    def pipeline(self) -> DataPipeline:
        """Create a DataPipeline instance."""
        return DataPipeline(random_state=42)
    
    def test_preprocess_data_returns_features_and_target(self, sample_data):
        """Test that preprocess_data returns features and target."""
        X, y = preprocess_data(sample_data)
        
        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert len(X) == len(sample_data)
    
    def test_preprocess_data_binary_target(self, sample_data):
        """Test that target is binary (0 or 1)."""
        _, y = preprocess_data(sample_data)
        
        assert set(y.unique()).issubset({0, 1})
    
    def test_preprocess_data_handles_missing_values(self):
        """Test that missing values are handled."""
        data = pd.DataFrame({
            "age": [50, np.nan, 60],
            "sex": [1, 0, 1],
            "cp": [2, 1, np.nan],
            "trestbps": [120, 130, 140],
            "chol": [200, np.nan, 250],
            "fbs": [0, 1, 0],
            "restecg": [0, 1, 0],
            "thalach": [150, 160, np.nan],
            "exang": [0, 1, 0],
            "oldpeak": [1.0, 2.0, 1.5],
            "slope": [1, 2, 1],
            "ca": [0, 1, 0],
            "thal": [2, 3, 2],
            "target": [0, 1, 0]
        })
        
        X, y = preprocess_data(data)
        
        # Should not have any NaN values
        assert not X.isnull().any().any()
    
    def test_pipeline_fit_transform(self, sample_data, pipeline):
        """Test pipeline fit_transform."""
        X, _ = preprocess_data(sample_data)
        X_transformed = pipeline.fit_transform(X)
        
        assert X_transformed is not None
        assert len(X_transformed) == len(X)
    
    def test_pipeline_transform_after_fit(self, sample_data, pipeline):
        """Test that transform works after fit."""
        X, _ = preprocess_data(sample_data)
        
        # Fit on training data
        pipeline.fit_transform(X[:40])
        
        # Transform test data
        X_test = pipeline.transform(X[40:])
        
        assert X_test is not None
        assert len(X_test) == 10
    
    def test_pipeline_split_data(self, sample_data, pipeline):
        """Test data splitting."""
        X, y = preprocess_data(sample_data)
        X_transformed = pipeline.fit_transform(X)
        
        X_train, X_test, y_train, y_test = pipeline.split_data(
            X_transformed, y, test_size=0.2
        )
        
        assert len(X_train) == 40
        assert len(X_test) == 10
        assert len(y_train) == 40
        assert len(y_test) == 10
    
    def test_feature_columns_defined(self):
        """Test that feature columns are properly defined."""
        assert len(FEATURE_COLUMNS) == 13
        assert "age" in FEATURE_COLUMNS
        assert "target" not in FEATURE_COLUMNS
    
    def test_numerical_and_categorical_features_complete(self):
        """Test that numerical and categorical features cover all columns."""
        all_features = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
        expected_features = set(FEATURE_COLUMNS)
        
        assert all_features == expected_features


class TestDataLoading:
    """Test cases for data loading functions."""
    
    def test_load_data_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent_file.csv")
    
    def test_load_data_returns_dataframe(self, tmp_path):
        """Test that load_data returns a DataFrame."""
        # Create a temporary CSV file
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(csv_path, index=False)
        
        result = load_data(str(csv_path))
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

