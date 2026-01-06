"""Model prediction module."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union
import joblib


class ModelPredictor:
    """
    Model predictor for heart disease classification.
    
    Loads trained model and preprocessing pipeline to make predictions.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        pipeline_path: Optional[str] = None
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the saved model
            pipeline_path: Path to the saved preprocessing pipeline
        """
        self.model = None
        self.pipeline = None
        self.model_path = model_path
        self.pipeline_path = pipeline_path
        
        if model_path:
            self.load_model(model_path)
        if pipeline_path:
            self.load_pipeline(pipeline_path)
    
    def load_model(self, file_path: str) -> None:
        """
        Load a trained model from file.
        
        Args:
            file_path: Path to the saved model
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        self.model = joblib.load(file_path)
        self.model_path = file_path
    
    def load_pipeline(self, file_path: str) -> None:
        """
        Load a preprocessing pipeline from file.
        
        Args:
            file_path: Path to the saved pipeline
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {file_path}")
        
        self.pipeline = joblib.load(file_path)
        self.pipeline_path = file_path
    
    def is_loaded(self) -> bool:
        """Check if both model and pipeline are loaded."""
        return self.model is not None and self.pipeline is not None
    
    def predict(
        self,
        features: Union[Dict[str, Any], pd.DataFrame, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Make a prediction for input features.
        
        Args:
            features: Input features as dict, DataFrame, or array
            
        Returns:
            Dictionary with prediction, probability, and risk level
        """
        if not self.is_loaded():
            raise ValueError("Model or pipeline not loaded. Call load_model() and load_pipeline() first.")
        
        # Convert input to DataFrame if needed
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, np.ndarray):
            df = pd.DataFrame(features)
        else:
            df = features.copy()
        
        # Preprocess features
        X = self.pipeline.transform(df)
        
        # Make prediction
        prediction = int(self.model.predict(X)[0])
        
        # Get probability
        probability = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            probability = float(proba[1])  # Probability of positive class
        
        # Determine risk level
        risk_level = self._get_risk_level(probability)
        
        return {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "disease_present": prediction == 1
        }
    
    def predict_batch(
        self,
        features: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Make predictions for multiple samples.
        
        Args:
            features: Input features as DataFrame or array
            
        Returns:
            DataFrame with predictions for each sample
        """
        if not self.is_loaded():
            raise ValueError("Model or pipeline not loaded.")
        
        # Convert to DataFrame if needed
        if isinstance(features, np.ndarray):
            df = pd.DataFrame(features)
        else:
            df = features.copy()
        
        # Preprocess
        X = self.pipeline.transform(df)
        
        # Predictions
        predictions = self.model.predict(X)
        
        # Probabilities
        probabilities = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)[:, 1]
        
        # Create result DataFrame
        results = pd.DataFrame({
            "prediction": predictions,
            "probability": probabilities if probabilities is not None else [None] * len(predictions),
            "disease_present": predictions == 1
        })
        
        results["risk_level"] = results["probability"].apply(self._get_risk_level)
        
        return results
    
    def _get_risk_level(self, probability: Optional[float]) -> str:
        """
        Determine risk level based on probability.
        
        Args:
            probability: Probability of heart disease
            
        Returns:
            Risk level string
        """
        if probability is None:
            return "unknown"
        
        if probability < 0.3:
            return "low"
        elif probability < 0.6:
            return "medium"
        else:
            return "high"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"loaded": False}
        
        info = {
            "loaded": True,
            "model_type": type(self.model).__name__,
            "model_path": self.model_path,
            "pipeline_path": self.pipeline_path,
        }
        
        # Add model-specific info
        if hasattr(self.model, "n_features_in_"):
            info["n_features"] = self.model.n_features_in_
        if hasattr(self.model, "classes_"):
            info["classes"] = self.model.classes_.tolist()
        
        return info

