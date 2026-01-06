#!/usr/bin/env python3
"""
Train heart disease prediction models.

This script:
1. Loads and preprocesses the dataset
2. Trains multiple models (Logistic Regression, Random Forest)
3. Logs experiments to MLflow
4. Saves the best model and preprocessing pipeline
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import DataPipeline, FEATURE_COLUMNS
from src.models.train import ModelTrainer


def main():
    """Main training function."""
    print("=" * 60)
    print("Heart Disease Model Training")
    print("=" * 60)
    
    # Paths
    data_path = PROJECT_ROOT / "data" / "raw" / "heart.csv"
    model_path = PROJECT_ROOT / "models" / "best_model.joblib"
    pipeline_path = PROJECT_ROOT / "models" / "preprocessing_pipeline.joblib"
    
    # Check if data exists
    if not data_path.exists():
        print(f"❌ Error: Dataset not found at {data_path}")
        print("Run 'python scripts/download_data.py' first.")
        sys.exit(1)
    
    print(f"\n📂 Loading data from: {data_path}")
    
    # Initialize data pipeline
    data_pipeline = DataPipeline(random_state=42)
    
    # Load and preprocess data
    X, y = data_pipeline.load_and_preprocess(str(data_path))
    print(f"✅ Loaded {len(X)} samples with {len(X.columns)} features")
    
    # Fit and transform features
    X_transformed = data_pipeline.fit_transform(X)
    print(f"✅ Features transformed: shape = {X_transformed.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = data_pipeline.split_data(
        X_transformed, y, test_size=0.2
    )
    print(f"✅ Data split: Train = {len(X_train)}, Test = {len(X_test)}")
    
    # Save preprocessing pipeline
    data_pipeline.save_pipeline(str(pipeline_path))
    print(f"✅ Preprocessing pipeline saved to: {pipeline_path}")
    
    # Initialize model trainer
    # Using SQLite backend for reliability in Jenkins CI/CD
    print("\n🤖 Starting model training with MLflow (SQLite backend)...")
    trainer = ModelTrainer(
        experiment_name="heart-disease-classification",
        tracking_uri="sqlite:///mlflow.db",  # SQLite for reliability
        log_models_to_mlflow=False  # Disabled for Jenkins compatibility
    )
    
    # Train models
    results = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=data_pipeline.get_feature_names()
    )
    
    # Save best model
    trainer.save_best_model(str(model_path))
    
    # Print final summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    best_name, _ = trainer.get_best_model()
    print(f"\n🏆 Best Model: {best_name}")
    print(f"📦 Model saved to: {model_path}")
    print(f"📦 Pipeline saved to: {pipeline_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. View experiments: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
    print("  2. Start API: uvicorn src.api.app:app --reload")
    print("  3. Test prediction: curl http://localhost:8000/health")
    print("\nℹ️  Note: Using SQLite backend for MLflow (more reliable in CI/CD)")


if __name__ == "__main__":
    main()

