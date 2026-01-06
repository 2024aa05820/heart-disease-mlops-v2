"""Model training with MLflow experiment tracking."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:
    """
    Model trainer with MLflow experiment tracking.
    
    Supports training multiple models, comparing performance,
    and logging all results to MLflow.
    
    NOTE: Uses SQLite backend for reliability in Jenkins CI/CD.
    Model artifacts are saved via joblib (not MLflow registry) to avoid
    file locking issues in automated pipelines.
    """
    
    def __init__(
        self,
        experiment_name: str = "heart-disease-classification",
        tracking_uri: str = "sqlite:///mlflow.db",
        log_models_to_mlflow: bool = False  # Disabled by default for Jenkins compatibility
    ):
        """
        Initialize the model trainer.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (default: SQLite for reliability)
            log_models_to_mlflow: Whether to log models to MLflow registry
                                  Set to False for Jenkins CI/CD compatibility
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.log_models_to_mlflow = log_models_to_mlflow
        self.models: Dict[str, Any] = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0
        
        # Setup MLflow with SQLite backend (more reliable than file store)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        print(f"📊 MLflow tracking URI: {tracking_uri}")
        print(f"📊 MLflow experiment: {experiment_name}")
    
    def _create_models(self, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create model instances.
        
        Args:
            config: Optional configuration dictionary for models
            
        Returns:
            Dictionary of model instances
        """
        config = config or {}
        
        models = {
            "logistic_regression": LogisticRegression(
                max_iter=config.get("logistic_regression", {}).get("max_iter", 1000),
                C=config.get("logistic_regression", {}).get("C", 1.0),
                solver=config.get("logistic_regression", {}).get("solver", "lbfgs"),
                random_state=42
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=config.get("random_forest", {}).get("n_estimators", 100),
                max_depth=config.get("random_forest", {}).get("max_depth", 10),
                min_samples_split=config.get("random_forest", {}).get("min_samples_split", 5),
                min_samples_leaf=config.get("random_forest", {}).get("min_samples_leaf", 2),
                random_state=42
            )
        }
        
        return models
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1_score": f1_score(y_true, y_pred, average="weighted")
        }
        
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        
        return metrics
    
    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> plt.Figure:
        """Create confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        return fig
    
    def _plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str
    ) -> plt.Figure:
        """Create ROC curve plot."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {model_name}")
        ax.legend(loc="lower right")
        plt.tight_layout()
        return fig
    
    def _plot_feature_importance(
        self,
        model: Any,
        feature_names: list,
        model_name: str
    ) -> Optional[plt.Figure]:
        """Create feature importance plot for tree-based models."""
        if not hasattr(model, "feature_importances_"):
            return None
        
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(importance)), importance[indices])
        ax.set_xticks(range(len(importance)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
        ax.set_xlabel("Features")
        ax.set_ylabel("Importance")
        ax.set_title(f"Feature Importance - {model_name}")
        plt.tight_layout()
        return fig
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[list] = None,
        model_config: Optional[Dict] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Train multiple models and log to MLflow.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names (for plotting)
            model_config: Optional model configuration
            
        Returns:
            Dictionary of model names and their metrics
        """
        models = self._create_models(model_config)
        results = {}
        
        for name, model in models.items():
            print(f"\n{'='*50}")
            print(f"Training: {name}")
            print(f"{'='*50}")
            
            with mlflow.start_run(run_name=name):
                # Log model parameters
                params = model.get_params()
                mlflow.log_params(params)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_proba = None
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_proba)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
                mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
                mlflow.log_metric("cv_std_accuracy", cv_scores.std())
                
                # Print metrics
                print(f"\nMetrics:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
                print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                
                # Print classification report
                print(f"\nClassification Report:")
                print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))
                
                # Create and log plots
                # Confusion matrix
                fig_cm = self._plot_confusion_matrix(y_test, y_pred, name)
                mlflow.log_figure(fig_cm, f"confusion_matrix_{name}.png")
                plt.close(fig_cm)
                
                # ROC curve
                if y_proba is not None:
                    fig_roc = self._plot_roc_curve(y_test, y_proba, name)
                    mlflow.log_figure(fig_roc, f"roc_curve_{name}.png")
                    plt.close(fig_roc)
                
                # Feature importance
                if feature_names:
                    fig_fi = self._plot_feature_importance(model, feature_names, name)
                    if fig_fi:
                        mlflow.log_figure(fig_fi, f"feature_importance_{name}.png")
                        plt.close(fig_fi)
                
                # Log model to MLflow registry (optional - disabled for Jenkins)
                # NOTE: Disabled by default to avoid file locking issues in CI/CD
                if self.log_models_to_mlflow:
                    try:
                        mlflow.sklearn.log_model(model, f"model_{name}")
                        print(f"  ✅ Model logged to MLflow registry")
                    except Exception as e:
                        print(f"  ⚠️ Could not log model to MLflow: {e}")
                        print(f"  ℹ️ Model will be saved via joblib instead")
                else:
                    print(f"  ℹ️ MLflow model logging disabled (using joblib for reliability)")
                
                # Store results
                results[name] = metrics
                self.models[name] = model
                
                # Track best model (by ROC-AUC or accuracy)
                score = metrics.get("roc_auc", metrics["accuracy"])
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = model
                    self.best_model_name = name
        
        print(f"\n{'='*50}")
        print(f"Best Model: {self.best_model_name} (Score: {self.best_score:.4f})")
        print(f"{'='*50}")
        
        return results
    
    def save_best_model(self, file_path: str) -> None:
        """
        Save the best model to a file.
        
        Args:
            file_path: Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train() first.")
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, file_path)
        print(f"Best model saved to: {file_path}")
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best trained model.
        
        Returns:
            Tuple of (model_name, model_instance)
        """
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train() first.")
        return self.best_model_name, self.best_model

