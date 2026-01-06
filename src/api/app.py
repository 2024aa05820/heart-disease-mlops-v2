"""
FastAPI application for Heart Disease Prediction API.

This module provides:
- /health endpoint for health checks
- /predict endpoint for making predictions
- /metrics endpoint for Prometheus metrics
- Comprehensive logging and monitoring
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Model paths
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
PIPELINE_PATH = PROJECT_ROOT / "models" / "preprocessing_pipeline.joblib"

# Global predictor instance
predictor = None


# ============================================
# Prometheus Metrics
# ============================================
PREDICTION_COUNTER = Counter(
    "heart_disease_predictions_total",
    "Total number of predictions made",
    ["result", "risk_level"]
)

PREDICTION_LATENCY = Histogram(
    "heart_disease_prediction_latency_seconds",
    "Prediction request latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]
)

REQUEST_COUNTER = Counter(
    "heart_disease_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status"]
)

ERROR_COUNTER = Counter(
    "heart_disease_errors_total",
    "Total number of errors",
    ["error_type"]
)


# ============================================
# Pydantic Models
# ============================================
class HeartDiseaseFeatures(BaseModel):
    """Input features for heart disease prediction."""
    
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1=male, 0=female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1=yes)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    prediction: int = Field(..., description="Prediction (0=no disease, 1=disease)")
    probability: Optional[float] = Field(None, description="Probability of heart disease")
    risk_level: str = Field(..., description="Risk level (low/medium/high)")
    disease_present: bool = Field(..., description="Whether heart disease is predicted")
    timestamp: str = Field(..., description="Timestamp of prediction")


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str
    model_loaded: bool
    timestamp: str
    version: str = "1.0.0"


class FeatureSchema(BaseModel):
    """Schema for input features."""
    
    name: str
    type: str
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None


# ============================================
# Application Lifespan
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    global predictor
    
    # Startup
    logger.info("Starting Heart Disease Prediction API...")
    
    try:
        from src.models.predict import ModelPredictor
        predictor = ModelPredictor()
        
        if MODEL_PATH.exists() and PIPELINE_PATH.exists():
            predictor.load_model(str(MODEL_PATH))
            predictor.load_pipeline(str(PIPELINE_PATH))
            logger.info("Model and pipeline loaded successfully")
        else:
            logger.warning(f"Model files not found at {MODEL_PATH} or {PIPELINE_PATH}")
            logger.warning("API will start but predictions will fail until models are loaded")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down Heart Disease Prediction API...")


# ============================================
# FastAPI Application
# ============================================
app = FastAPI(
    title="Heart Disease Prediction API",
    description="""
    API for predicting heart disease risk based on patient health data.
    
    ## Features
    - Predict heart disease risk from health metrics
    - Get probability and risk level
    - Prometheus metrics for monitoring
    
    ## Model
    Trained on the UCI Heart Disease dataset using machine learning.
    """,
    version="1.0.0",
    lifespan=lifespan
)


# ============================================
# Middleware for Request Logging
# ============================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and track metrics."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate latency
    latency = time.time() - start_time
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Latency: {latency:.3f}s"
    )
    
    # Track metrics
    REQUEST_COUNTER.labels(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code
    ).inc()
    
    return response


# ============================================
# Endpoints
# ============================================
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Heart Disease Prediction API",
        "version": "1.0.0",
        "description": "API for predicting heart disease risk",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the API and whether the model is loaded.
    """
    model_loaded = predictor is not None and predictor.is_loaded()
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HeartDiseaseFeatures):
    """
    Make a heart disease prediction.
    
    Takes patient health metrics and returns:
    - Binary prediction (0=no disease, 1=disease present)
    - Probability of heart disease
    - Risk level (low/medium/high)
    """
    if predictor is None or not predictor.is_loaded():
        ERROR_COUNTER.labels(error_type="model_not_loaded").inc()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model files exist."
        )
    
    start_time = time.time()
    
    try:
        # Convert to dict for prediction
        feature_dict = features.model_dump()
        
        # Make prediction
        result = predictor.predict(feature_dict)
        
        # Track latency
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        
        # Track prediction metrics
        PREDICTION_COUNTER.labels(
            result="disease" if result["prediction"] == 1 else "no_disease",
            risk_level=result["risk_level"]
        ).inc()
        
        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            risk_level=result["risk_level"],
            disease_present=result["disease_present"],
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        ERROR_COUNTER.labels(error_type="prediction_error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus format for scraping.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/schema", response_model=list[FeatureSchema])
async def get_schema():
    """
    Get the schema for input features.
    
    Returns information about each feature including name, type, and valid range.
    """
    return [
        FeatureSchema(name="age", type="float", description="Age in years", min_value=0, max_value=120),
        FeatureSchema(name="sex", type="int", description="Sex (1=male, 0=female)", min_value=0, max_value=1),
        FeatureSchema(name="cp", type="int", description="Chest pain type", min_value=0, max_value=3),
        FeatureSchema(name="trestbps", type="float", description="Resting blood pressure (mm Hg)", min_value=50, max_value=250),
        FeatureSchema(name="chol", type="float", description="Serum cholesterol (mg/dl)", min_value=100, max_value=600),
        FeatureSchema(name="fbs", type="int", description="Fasting blood sugar > 120 mg/dl", min_value=0, max_value=1),
        FeatureSchema(name="restecg", type="int", description="Resting ECG results", min_value=0, max_value=2),
        FeatureSchema(name="thalach", type="float", description="Maximum heart rate achieved", min_value=50, max_value=250),
        FeatureSchema(name="exang", type="int", description="Exercise induced angina", min_value=0, max_value=1),
        FeatureSchema(name="oldpeak", type="float", description="ST depression", min_value=0, max_value=10),
        FeatureSchema(name="slope", type="int", description="Slope of peak ST segment", min_value=0, max_value=2),
        FeatureSchema(name="ca", type="int", description="Number of major vessels (0-4)", min_value=0, max_value=4),
        FeatureSchema(name="thal", type="int", description="Thalassemia", min_value=0, max_value=3),
    ]


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if predictor is None:
        return {"loaded": False, "message": "Model not initialized"}
    
    return predictor.get_model_info()


# ============================================
# Error Handlers
# ============================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    ERROR_COUNTER.labels(error_type="unhandled").inc()
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

