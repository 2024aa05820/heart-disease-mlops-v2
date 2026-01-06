"""Unit tests for model training and prediction module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import ModelPredictor


class TestModelPredictor:
    """Test cases for ModelPredictor class."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing."""
        return {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1
        }
    
    def test_predictor_initialization(self):
        """Test that ModelPredictor initializes correctly."""
        predictor = ModelPredictor()
        
        assert predictor.model is None
        assert predictor.pipeline is None
        assert predictor.is_loaded() is False
    
    def test_predictor_is_loaded_false_without_model(self):
        """Test that is_loaded returns False without model."""
        predictor = ModelPredictor()
        assert predictor.is_loaded() is False
    
    def test_predictor_load_model_file_not_found(self):
        """Test that FileNotFoundError is raised for missing model."""
        predictor = ModelPredictor()
        
        with pytest.raises(FileNotFoundError):
            predictor.load_model("nonexistent_model.joblib")
    
    def test_predictor_load_pipeline_file_not_found(self):
        """Test that FileNotFoundError is raised for missing pipeline."""
        predictor = ModelPredictor()
        
        with pytest.raises(FileNotFoundError):
            predictor.load_pipeline("nonexistent_pipeline.joblib")
    
    def test_predictor_get_model_info_not_loaded(self):
        """Test model info when not loaded."""
        predictor = ModelPredictor()
        info = predictor.get_model_info()
        
        assert info["loaded"] is False
    
    def test_get_risk_level_low(self):
        """Test risk level calculation for low probability."""
        predictor = ModelPredictor()
        
        assert predictor._get_risk_level(0.1) == "low"
        assert predictor._get_risk_level(0.2) == "low"
        assert predictor._get_risk_level(0.29) == "low"
    
    def test_get_risk_level_medium(self):
        """Test risk level calculation for medium probability."""
        predictor = ModelPredictor()
        
        assert predictor._get_risk_level(0.3) == "medium"
        assert predictor._get_risk_level(0.45) == "medium"
        assert predictor._get_risk_level(0.59) == "medium"
    
    def test_get_risk_level_high(self):
        """Test risk level calculation for high probability."""
        predictor = ModelPredictor()
        
        assert predictor._get_risk_level(0.6) == "high"
        assert predictor._get_risk_level(0.8) == "high"
        assert predictor._get_risk_level(1.0) == "high"
    
    def test_get_risk_level_none(self):
        """Test risk level calculation for None probability."""
        predictor = ModelPredictor()
        
        assert predictor._get_risk_level(None) == "unknown"
    
    def test_predict_raises_without_model(self, sample_features):
        """Test that predict raises error without loaded model."""
        predictor = ModelPredictor()
        
        with pytest.raises(ValueError):
            predictor.predict(sample_features)


class TestModelTraining:
    """Test cases for model training integration."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 100
        
        X = np.random.randn(n, 13)
        y = np.random.randint(0, 2, n)
        
        return X, y
    
    def test_training_data_shape(self, sample_training_data):
        """Test that training data has correct shape."""
        X, y = sample_training_data
        
        assert X.shape == (100, 13)
        assert y.shape == (100,)
    
    def test_training_data_labels(self, sample_training_data):
        """Test that training labels are binary."""
        _, y = sample_training_data
        
        assert set(np.unique(y)).issubset({0, 1})

