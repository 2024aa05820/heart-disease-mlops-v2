# Heart Disease Prediction - MLOps Project

[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-Jenkins-blue)](https://www.jenkins.io/)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-blue)](https://kubernetes.io/)

A production-ready machine learning solution for predicting heart disease risk, built with modern MLOps best practices.

## 🎯 Project Overview

**Assignment:** MLOps (S1-25_AIMLCZG523) - End-to-End ML Model Development, CI/CD, and Production Deployment

**Problem Statement:** Build a machine learning classifier to predict the risk of heart disease based on patient health data, and deploy the solution as a cloud-ready, monitored API.

**Dataset:** UCI Heart Disease Dataset (303 samples, 14 features)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Jenkins CI/CD Pipeline                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │   Lint   │→│   Test   │→│  Train   │→│  Docker  │→│  Deploy  │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster (Minikube)                        │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Heart Disease API (FastAPI)                                      │ │
│  │  - /health     Health check                                       │ │
│  │  - /predict    Make predictions                                   │ │
│  │  - /metrics    Prometheus metrics                                 │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
heart-disease-mlops-v2/
├── data/
│   ├── raw/                    # Raw dataset
│   └── processed/              # Processed data
├── deploy/
│   └── k8s/                    # Kubernetes manifests
│       ├── deployment.yaml
│       ├── service.yaml
│       └── ingress.yaml
├── models/                     # Trained models
├── mlruns/                     # MLflow experiments
├── notebooks/                  # Jupyter notebooks (EDA)
├── reports/
│   └── screenshots/            # Documentation screenshots
├── scripts/
│   ├── download_data.py        # Dataset download
│   ├── train.py                # Model training
│   ├── rocky-setup.sh          # Rocky Linux setup
│   └── configure-jenkins-minikube.sh
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI application
│   ├── config/
│   │   └── config.yaml         # Configuration
│   ├── data/
│   │   └── pipeline.py         # Data preprocessing
│   └── models/
│       ├── train.py            # Model trainer
│       └── predict.py          # Model predictor
├── tests/
│   ├── test_api.py             # API tests
│   ├── test_data.py            # Data tests
│   └── test_model.py           # Model tests
├── Dockerfile                  # Container definition
├── Jenkinsfile                 # CI/CD pipeline
├── Makefile                    # Build automation
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Option 1: Local Development

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/heart-disease-mlops.git
cd heart-disease-mlops

# Setup environment
make setup

# Activate virtual environment
source venv/bin/activate

# Train models
make train

# Start API
make serve

# Test API
curl http://localhost:8000/health
```

### Option 2: Rocky Linux Production Deployment

```bash
# 1. Run automated setup (installs everything)
sudo ./scripts/rocky-setup.sh

# 2. Log out and back in (for docker group)
exit

# 3. Start Minikube
minikube start --driver=docker --cpus=2 --memory=4096

# 4. Configure Jenkins
sudo ./scripts/configure-jenkins-minikube.sh

# 5. Access Jenkins and create pipeline
http://<server-ip>:8080
```

## 📋 Assignment Tasks

### 1. Data Acquisition & EDA (5 marks)
- ✅ Download script: `scripts/download_data.py`
- ✅ Data cleaning and preprocessing: `src/data/pipeline.py`
- ✅ EDA notebook: `notebooks/01_eda.ipynb`

### 2. Feature Engineering & Model Development (8 marks)
- ✅ Feature preprocessing (scaling, encoding): `src/data/pipeline.py`
- ✅ Two classification models (Logistic Regression, Random Forest)
- ✅ Cross-validation and metrics evaluation
- ✅ Model training: `src/models/train.py`

### 3. Experiment Tracking (5 marks)
- ✅ MLflow integration for all experiments
- ✅ Logging parameters, metrics, artifacts
- ✅ View experiments: `mlflow ui --port 5000`

### 4. Model Packaging & Reproducibility (7 marks)
- ✅ Model saved in joblib format
- ✅ Complete `requirements.txt`
- ✅ Preprocessing pipeline saved separately

### 5. CI/CD Pipeline & Automated Testing (8 marks)
- ✅ Unit tests with pytest: `tests/`
- ✅ Jenkins pipeline: `Jenkinsfile`
- ✅ Linting (ruff, black)
- ✅ Automated testing

### 6. Model Containerization (5 marks)
- ✅ Docker container: `Dockerfile`
- ✅ FastAPI with `/predict` endpoint
- ✅ JSON input/output with confidence

### 7. Production Deployment (7 marks)
- ✅ Kubernetes manifests: `deploy/k8s/`
- ✅ Minikube deployment
- ✅ NodePort service (30080)

### 8. Monitoring & Logging (3 marks)
- ✅ Request logging in API
- ✅ Prometheus metrics: `/metrics` endpoint
- ✅ Grafana-ready metrics

### 9. Documentation & Reporting (2 marks)
- ✅ Complete README
- ✅ Setup instructions
- ✅ Architecture diagram

## 🧪 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Make prediction |
| `/metrics` | GET | Prometheus metrics |
| `/schema` | GET | Feature schema |
| `/docs` | GET | Swagger documentation |

### Prediction Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

Response:
```json
{
  "prediction": 1,
  "probability": 0.85,
  "risk_level": "high",
  "disease_present": true,
  "timestamp": "2024-01-06T12:00:00"
}
```

## 🔧 Jenkins CI/CD Pipeline

The Jenkinsfile provides end-to-end automation:

1. **Checkout** - Clone code from GitHub
2. **Setup** - Create Python environment
3. **Lint** - Run ruff and black
4. **Test** - Run pytest
5. **Download** - Fetch UCI dataset
6. **Train** - Train models with MLflow
7. **Build** - Create Docker image
8. **Test** - Verify Docker container
9. **Load** - Push to Minikube
10. **Deploy** - Apply Kubernetes manifests
11. **Verify** - Check deployment health
12. **MLflow** - Start experiment UI

## 📊 MLflow Experiment Tracking

```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Access at: http://localhost:5000
```

Logged metrics:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Cross-validation scores
- Confusion matrix
- ROC curve
- Feature importance

## 🐳 Docker

```bash
# Build image
make docker-build

# Run container
make docker-run

# Test container
curl http://localhost:8000/health
```

## ☸️ Kubernetes

```bash
# Deploy
make deploy

# Check status
make k8s-status

# View logs
make k8s-logs

# Get service URL
minikube service heart-disease-api-service --url
```

## 📈 Monitoring

The API exposes Prometheus metrics at `/metrics`:

- `heart_disease_predictions_total` - Total predictions
- `heart_disease_prediction_latency_seconds` - Latency histogram
- `heart_disease_requests_total` - Total requests
- `heart_disease_errors_total` - Total errors

## 👤 Author

- **Course:** MLOps (S1-25_AIMLCZG523)
- **Institution:** BITS Pilani

## 📄 License

This project is for educational purposes (BITS Pilani MLOps Assignment).

