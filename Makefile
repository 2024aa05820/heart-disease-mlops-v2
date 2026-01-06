.PHONY: help install lint format test train serve docker-build docker-run deploy clean

# Default Python
PYTHON := python3

help:
	@echo "Heart Disease MLOps Project"
	@echo "==========================="
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make setup        - Complete setup (venv + install + download)"
	@echo "  make train        - Train models with MLflow"
	@echo "  make serve        - Start API server"
	@echo ""
	@echo "📦 Setup:"
	@echo "  make venv         - Create virtual environment"
	@echo "  make install      - Install dependencies"
	@echo "  make download     - Download dataset"
	@echo ""
	@echo "🧪 Testing & Quality:"
	@echo "  make lint         - Run linting (ruff)"
	@echo "  make format       - Format code (black)"
	@echo "  make test         - Run pytest"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make docker-test  - Test Docker container"
	@echo ""
	@echo "☸️  Kubernetes:"
	@echo "  make deploy       - Deploy to Kubernetes"
	@echo "  make k8s-status   - Check deployment status"
	@echo "  make k8s-logs     - View pod logs"
	@echo ""
	@echo "📊 MLflow:"
	@echo "  make mlflow-ui    - Start MLflow UI (SQLite backend)"
	@echo ""
	@echo "📈 Monitoring:"
	@echo "  make monitoring-up     - Start Prometheus + Grafana (Docker)"
	@echo "  make monitoring-down   - Stop monitoring stack"
	@echo "  make monitoring-k8s    - Deploy monitoring to Kubernetes"
	@echo ""
	@echo "🔧 Rocky Linux:"
	@echo "  make rocky-setup  - Install all prerequisites"
	@echo "  make rocky-start  - Start Minikube"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  make clean        - Remove generated files"

# ==================== SETUP ====================
venv:
	$(PYTHON) -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

setup: venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	. venv/bin/activate && python scripts/download_data.py
	@echo "✅ Setup complete! Activate venv: source venv/bin/activate"

download:
	$(PYTHON) scripts/download_data.py

# ==================== TRAINING ====================
train:
	$(PYTHON) scripts/train.py

# ==================== API ====================
serve:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# ==================== TESTING ====================
lint:
	ruff check src/ tests/ scripts/

format:
	black src/ tests/ scripts/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# ==================== DOCKER ====================
docker-build:
	docker build -t heart-disease-api:latest .

docker-run:
	docker run -d -p 8000:8000 --name heart-api heart-disease-api:latest
	@echo "Container started. API at http://localhost:8000"

docker-test:
	@echo "Testing Docker container..."
	docker run -d -p 8001:8000 --name heart-api-test heart-disease-api:latest
	@sleep 10
	curl http://localhost:8001/health
	docker stop heart-api-test
	docker rm heart-api-test

docker-stop:
	docker stop heart-api && docker rm heart-api

# ==================== KUBERNETES ====================
deploy:
	kubectl apply -f deploy/k8s/
	@echo "Deployed! Check status: make k8s-status"

undeploy:
	kubectl delete -f deploy/k8s/

k8s-status:
	kubectl get pods,svc,ingress -l app=heart-disease-api

k8s-logs:
	kubectl logs -f -l app=heart-disease-api

k8s-restart:
	kubectl rollout restart deployment/heart-disease-api

# ==================== MLFLOW ====================
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

# ==================== MONITORING ====================
monitoring-up:
	docker-compose -f docker-compose.monitoring.yml up -d
	@echo "✅ Monitoring stack started!"
	@echo "   Prometheus: http://localhost:9090"
	@echo "   Grafana:    http://localhost:3000 (admin/admin123)"

monitoring-down:
	docker-compose -f docker-compose.monitoring.yml down

monitoring-k8s:
	kubectl apply -f deploy/monitoring/k8s-monitoring.yaml
	@echo "✅ Monitoring deployed to Kubernetes!"
	@MINIKUBE_IP=$$(minikube ip); \
	echo "   Prometheus: http://$$MINIKUBE_IP:30090"; \
	echo "   Grafana:    http://$$MINIKUBE_IP:30030 (admin/admin123)"

monitoring-k8s-down:
	kubectl delete -f deploy/monitoring/k8s-monitoring.yaml

# ==================== ROCKY LINUX ====================
rocky-setup:
	sudo ./scripts/rocky-setup.sh

rocky-start:
	minikube start --driver=docker --cpus=2 --memory=4096

rocky-status:
	@echo "Docker:"
	@sudo systemctl status docker --no-pager | head -3
	@echo ""
	@echo "Jenkins:"
	@sudo systemctl status jenkins --no-pager | head -3
	@echo ""
	@echo "Minikube:"
	@minikube status || echo "Not running"

configure-jenkins:
	sudo ./scripts/configure-jenkins-minikube.sh

urls:
	@echo "Service URLs:"
	@MINIKUBE_IP=$$(minikube ip 2>/dev/null || echo "N/A"); \
	SERVER_IP=$$(hostname -I | awk '{print $$1}'); \
	echo "  API:     http://$$MINIKUBE_IP:30080"; \
	echo "  Swagger: http://$$MINIKUBE_IP:30080/docs"; \
	echo "  MLflow:  http://$$SERVER_IP:5001"; \
	echo "  Jenkins: http://$$SERVER_IP:8080"

# ==================== CLEANUP ====================
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ dist/ build/ *.egg-info/

clean-all: clean
	rm -rf venv mlruns/ models/*.joblib

