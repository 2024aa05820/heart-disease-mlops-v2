// Heart Disease MLOps - Jenkins CI/CD Pipeline
// =============================================
// This pipeline provides end-to-end automation:
// 1. Code checkout and linting
// 2. Unit testing
// 3. Model training with MLflow
// 4. Docker image build and test
// 5. Kubernetes deployment
// 6. MLflow UI startup

pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'heart-disease-api'
        IMAGE_TAG = "${env.BUILD_NUMBER}"
        MINIKUBE_HOME = '/var/lib/jenkins/.minikube'
        PATH = "/usr/local/bin:${env.PATH}"
    }
    
    stages {
        // ============================================
        // Stage 1: Checkout Code
        // ============================================
        stage('Checkout') {
            steps {
                echo '📥 Checking out code from GitHub...'
                checkout scm
                sh 'git log -1 --pretty=format:"%h - %an: %s"'
            }
        }
        
        // ============================================
        // Stage 2: Setup Python Environment
        // ============================================
        stage('Setup Python Environment') {
            steps {
                echo '🐍 Setting up Python environment...'
                sh '''
                    python3 -m venv venv || true
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }
        
        // ============================================
        // Stage 3: Code Linting
        // ============================================
        stage('Lint Code') {
            steps {
                echo '🔍 Running code linters...'
                sh '''
                    . venv/bin/activate
                    pip install ruff black
                    echo "Running Ruff..."
                    ruff check src/ tests/ scripts/ || true
                    echo "Checking Black formatting..."
                    black --check src/ tests/ scripts/ || true
                '''
            }
        }
        
        // ============================================
        // Stage 4: Run Unit Tests
        // ============================================
        stage('Run Tests') {
            steps {
                echo '🧪 Running unit tests...'
                sh '''
                    . venv/bin/activate
                    pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing || true
                '''
            }
            post {
                always {
                    // Archive test results
                    junit allowEmptyResults: true, testResults: '**/test-results.xml'
                }
            }
        }
        
        // ============================================
        // Stage 5: Download Dataset
        // ============================================
        stage('Download Dataset') {
            steps {
                echo '📊 Downloading dataset...'
                sh '''
                    . venv/bin/activate
                    python scripts/download_data.py
                    
                    echo "Dataset downloaded:"
                    ls -la data/raw/
                '''
            }
        }
        
        // ============================================
        // Stage 6: Train Models
        // ============================================
        stage('Train Models') {
            steps {
                echo '🤖 Training ML models with MLflow tracking...'
                sh '''
                    set -e
                    
                    echo "📂 Current directory: $(pwd)"
                    echo "📂 Models directory before training:"
                    ls -la models/ || echo "models/ directory empty"
                    
                    . venv/bin/activate
                    
                    echo "🚀 Starting model training..."
                    python scripts/train.py
                    
                    echo ""
                    echo "📂 Models directory after training:"
                    ls -la models/
                    
                    # Verify models were created
                    if [ ! -f "models/best_model.joblib" ]; then
                        echo "❌ ERROR: best_model.joblib not found!"
                        exit 1
                    fi
                    
                    if [ ! -f "models/preprocessing_pipeline.joblib" ]; then
                        echo "❌ ERROR: preprocessing_pipeline.joblib not found!"
                        exit 1
                    fi
                    
                    echo "✅ Model files verified:"
                    ls -lh models/*.joblib
                '''
            }
        }
        
        // ============================================
        // Stage 7: Build Docker Image
        // ============================================
        stage('Build Docker Image') {
            steps {
                echo '🐳 Building Docker image...'
                sh """
                    echo "🔍 Verifying model files before Docker build..."
                    if [ ! -f "models/best_model.joblib" ] || [ ! -f "models/preprocessing_pipeline.joblib" ]; then
                        echo "❌ ERROR: Model files not found!"
                        ls -la models/
                        exit 1
                    fi
                    
                    echo "✅ Model files verified:"
                    ls -lh models/*.joblib
                    
                    # Use Minikube's Docker daemon
                    if command -v minikube &> /dev/null; then
                        echo "✅ Using Minikube's Docker daemon"
                        eval \$(minikube docker-env) || echo "⚠️  Using local Docker"
                    fi
                    
                    # Build image
                    docker build -t ${DOCKER_IMAGE}:${IMAGE_TAG} .
                    docker tag ${DOCKER_IMAGE}:${IMAGE_TAG} ${DOCKER_IMAGE}:latest
                    
                    echo "📦 Docker images:"
                    docker images | grep ${DOCKER_IMAGE}
                    
                    echo "🔍 Verifying models in Docker image..."
                    docker run --rm ${DOCKER_IMAGE}:${IMAGE_TAG} ls -lh /app/models/
                """
            }
        }
        
        // ============================================
        // Stage 8: Test Docker Image
        // ============================================
        stage('Test Docker Image') {
            steps {
                echo '🧪 Testing Docker image...'
                sh """
                    # Use Minikube's Docker daemon
                    if command -v minikube &> /dev/null; then
                        eval \$(minikube docker-env) || true
                    fi
                    
                    # Clean up old test containers
                    echo "🧹 Cleaning up old test containers..."
                    docker ps -a | grep test-api- | awk '{print \$1}' | xargs -r docker rm -f 2>/dev/null || true
                    docker ps --filter "publish=8001" -q | xargs -r docker stop 2>/dev/null || true
                    docker ps -a --filter "publish=8001" -q | xargs -r docker rm -f 2>/dev/null || true
                    
                    sleep 3
                    
                    # Start test container
                    echo "🚀 Starting container test-api-${BUILD_NUMBER}..."
                    docker run -d --name test-api-${BUILD_NUMBER} -p 8001:8000 ${DOCKER_IMAGE}:${IMAGE_TAG}
                    
                    echo "⏳ Waiting for container to start..."
                    sleep 10
                    
                    # Show logs
                    echo "📋 Container logs:"
                    docker logs test-api-${BUILD_NUMBER}
                    
                    # Check if container is running
                    if ! docker ps | grep -q test-api-${BUILD_NUMBER}; then
                        echo "❌ Container is not running!"
                        docker logs test-api-${BUILD_NUMBER}
                        docker rm -f test-api-${BUILD_NUMBER}
                        exit 1
                    fi
                    
                    # Test health endpoint
                    echo "🏥 Testing health endpoint..."
                    for i in 1 2 3 4 5; do
                        echo "Attempt \$i/5..."
                        HTTP_CODE=\$(curl -s -o /tmp/health_response.txt -w "%{http_code}" http://localhost:8001/health)
                        
                        if [ "\$HTTP_CODE" = "200" ]; then
                            echo "✅ Health check passed!"
                            cat /tmp/health_response.txt
                            break
                        fi
                        
                        if [ \$i -eq 5 ]; then
                            echo "❌ Health check failed after 5 attempts"
                            docker logs test-api-${BUILD_NUMBER}
                            docker rm -f test-api-${BUILD_NUMBER}
                            exit 1
                        fi
                        sleep 5
                    done
                    
                    # Cleanup
                    echo "🧹 Cleaning up test container..."
                    docker stop test-api-${BUILD_NUMBER}
                    docker rm test-api-${BUILD_NUMBER}
                    echo "✅ Docker image test completed!"
                """
            }
        }
        
        // ============================================
        // Stage 9: Load Image to Minikube
        // ============================================
        stage('Load Image to Minikube') {
            steps {
                echo '📦 Verifying image in Minikube...'
                sh """
                    if command -v minikube &> /dev/null; then
                        echo "✅ Verifying image in Minikube Docker daemon"
                        eval \$(minikube docker-env) || exit 0
                        
                        if docker images | grep -q ${DOCKER_IMAGE}; then
                            echo "✅ Image found in Minikube"
                            docker images | grep ${DOCKER_IMAGE}
                        else
                            echo "⚠️  Loading image to Minikube..."
                            minikube image load ${DOCKER_IMAGE}:latest
                        fi
                    else
                        echo "⚠️  Minikube not available"
                    fi
                """
            }
        }
        
        // ============================================
        // Stage 10: Deploy to Kubernetes
        // ============================================
        stage('Deploy to Kubernetes') {
            steps {
                echo '🚀 Deploying to Kubernetes...'
                sh '''
                    # Apply Kubernetes manifests
                    kubectl apply -f deploy/k8s/
                    
                    # Wait for deployment
                    kubectl wait --for=condition=available --timeout=300s deployment/heart-disease-api || true
                    
                    # Restart deployment with new image
                    kubectl rollout restart deployment/heart-disease-api
                    kubectl rollout status deployment/heart-disease-api
                    
                    echo "✅ Deployment complete!"
                '''
            }
        }
        
        // ============================================
        // Stage 11: Verify Deployment
        // ============================================
        stage('Verify Deployment') {
            steps {
                echo '✅ Verifying deployment...'
                sh '''
                    echo "📦 Pods:"
                    kubectl get pods -l app=heart-disease-api
                    
                    echo ""
                    echo "🌐 Services:"
                    kubectl get services
                    
                    # Get service URL
                    SERVICE_URL=$(minikube service heart-disease-api-service --url)
                    echo ""
                    echo "🔗 Service URL: $SERVICE_URL"
                    
                    # Wait for pods to be ready
                    sleep 15
                    
                    # Test health endpoint
                    echo ""
                    echo "🏥 Testing health endpoint..."
                    curl -f $SERVICE_URL/health || echo "Health check pending..."
                '''
            }
        }
        
        // ============================================
        // Stage 12: Start MLflow UI
        // ============================================
        stage('Start MLflow UI') {
            steps {
                echo '📊 Starting MLflow UI with SQLite backend...'
                sh '''
                    . venv/bin/activate
                    
                    # Kill any existing MLflow processes
                    pkill -f "mlflow ui" || true
                    sleep 2
                    
                    # Start MLflow UI with SQLite backend (more reliable)
                    # This matches the tracking_uri used in training
                    nohup mlflow ui \
                        --backend-store-uri sqlite:///mlflow.db \
                        --host 0.0.0.0 \
                        --port 5001 \
                        > mlflow.log 2>&1 &
                    
                    sleep 3
                    
                    if pgrep -f "mlflow ui" > /dev/null; then
                        echo "✅ MLflow UI started on port 5001 (SQLite backend)"
                    else
                        echo "⚠️ MLflow UI may not have started - check mlflow.log"
                        cat mlflow.log || true
                    fi
                '''
            }
        }
    }
    
    // ============================================
    // Post-build Actions
    // ============================================
    post {
        success {
            echo '✅ Pipeline completed successfully!'
            sh '''
                echo "========================================="
                echo "📊 Deployment Summary"
                echo "========================================="
                echo "Build Number: ${BUILD_NUMBER}"
                echo "Docker Image: ${DOCKER_IMAGE}:${IMAGE_TAG}"
                echo ""
                echo "🔗 Service URLs:"
                SERVICE_URL=$(minikube service heart-disease-api-service --url 2>/dev/null || echo "Not available")
                SERVER_IP=$(hostname -I | awk '{print $1}')
                echo "  API:     $SERVICE_URL"
                echo "  Swagger: $SERVICE_URL/docs"
                echo "  MLflow:  http://$SERVER_IP:5001"
                echo "  Jenkins: http://$SERVER_IP:8080"
                echo ""
                echo "📦 Kubernetes Resources:"
                kubectl get pods -l app=heart-disease-api
                echo "========================================="
            '''
        }
        failure {
            echo '❌ Pipeline failed!'
            sh '''
                echo "Check logs for errors:"
                echo "- Jenkins console output"
                echo "- Docker logs: docker logs <container-id>"
                echo "- Kubernetes logs: kubectl logs <pod-name>"
            '''
        }
        always {
            echo '🧹 Cleaning up...'
            sh '''
                # Use Minikube's Docker daemon
                if command -v minikube &> /dev/null; then
                    eval $(minikube docker-env) || true
                fi
                
                # Clean up test containers
                docker ps -a | grep test-api- | awk '{print $1}' | xargs -r docker rm -f 2>/dev/null || true
                
                # Clean up old images (keep last 5)
                docker images ${DOCKER_IMAGE} --format "{{.ID}}" | tail -n +6 | xargs -r docker rmi || true
            '''
        }
    }
}

