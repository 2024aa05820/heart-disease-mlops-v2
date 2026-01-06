"""Unit tests for FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test cases for health endpoint."""
    
    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_returns_status(self, client):
        """Test that health endpoint returns status field."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]
    
    def test_health_returns_model_loaded(self, client):
        """Test that health endpoint returns model_loaded field."""
        response = client.get("/health")
        data = response.json()
        
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)
    
    def test_health_returns_timestamp(self, client):
        """Test that health endpoint returns timestamp."""
        response = client.get("/health")
        data = response.json()
        
        assert "timestamp" in data


class TestRootEndpoint:
    """Test cases for root endpoint."""
    
    def test_root_returns_200(self, client):
        """Test that root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_returns_api_info(self, client):
        """Test that root endpoint returns API info."""
        response = client.get("/")
        data = response.json()
        
        assert "name" in data
        assert "version" in data


class TestSchemaEndpoint:
    """Test cases for schema endpoint."""
    
    def test_schema_returns_200(self, client):
        """Test that schema endpoint returns 200."""
        response = client.get("/schema")
        assert response.status_code == 200
    
    def test_schema_returns_list(self, client):
        """Test that schema endpoint returns a list."""
        response = client.get("/schema")
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) == 13  # 13 features
    
    def test_schema_contains_required_fields(self, client):
        """Test that each schema item has required fields."""
        response = client.get("/schema")
        data = response.json()
        
        for item in data:
            assert "name" in item
            assert "type" in item
            assert "description" in item


class TestPredictEndpoint:
    """Test cases for predict endpoint."""
    
    @pytest.fixture
    def valid_input(self):
        """Valid input for prediction."""
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
    
    def test_predict_accepts_valid_input(self, client, valid_input):
        """Test that predict accepts valid input."""
        response = client.post("/predict", json=valid_input)
        
        # May return 503 if model not loaded, which is acceptable
        assert response.status_code in [200, 503]
    
    def test_predict_rejects_missing_fields(self, client):
        """Test that predict rejects input with missing fields."""
        incomplete_input = {"age": 63, "sex": 1}
        
        response = client.post("/predict", json=incomplete_input)
        assert response.status_code == 422  # Validation error
    
    def test_predict_rejects_invalid_age(self, client, valid_input):
        """Test that predict rejects invalid age."""
        valid_input["age"] = -5
        
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 422
    
    def test_predict_rejects_invalid_sex(self, client, valid_input):
        """Test that predict rejects invalid sex value."""
        valid_input["sex"] = 5
        
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Test cases for metrics endpoint."""
    
    def test_metrics_returns_200(self, client):
        """Test that metrics endpoint returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200
    
    def test_metrics_returns_prometheus_format(self, client):
        """Test that metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        
        # Prometheus metrics should contain certain patterns
        content = response.text
        assert "heart_disease" in content or "python" in content

