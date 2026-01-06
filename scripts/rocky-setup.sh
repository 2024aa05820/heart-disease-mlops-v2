#!/bin/bash
#
# Rocky Linux Setup Script for Heart Disease MLOps
# =================================================
# Installs: Java 17, Docker, kubectl, Minikube, Jenkins, Python 3
#
# Usage: sudo ./scripts/rocky-setup.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Heart Disease MLOps - Rocky Linux Setup  ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root: sudo $0${NC}"
    exit 1
fi

ACTUAL_USER=${SUDO_USER:-$USER}
echo -e "${BLUE}Installing for user: $ACTUAL_USER${NC}"

# Install EPEL repository (NO system upgrade)
echo -e "${YELLOW}📦 Installing EPEL repository...${NC}"
# NOTE: Skipping 'dnf update -y' to avoid Linux system upgrades
dnf install -y epel-release
echo -e "${GREEN}✅ EPEL repository installed${NC}"

# Install Java 17
echo -e "${YELLOW}📦 Installing Java 17...${NC}"
dnf install -y java-17-openjdk java-17-openjdk-devel
echo -e "${GREEN}✅ Java installed${NC}"
java -version

# Install Docker
echo -e "${YELLOW}📦 Installing Docker...${NC}"
dnf install -y yum-utils
yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

systemctl start docker
systemctl enable docker
usermod -aG docker $ACTUAL_USER

echo -e "${GREEN}✅ Docker installed${NC}"
docker --version

# Install kubectl
echo -e "${YELLOW}📦 Installing kubectl...${NC}"
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm -f kubectl
echo -e "${GREEN}✅ kubectl installed${NC}"

# Install Minikube
echo -e "${YELLOW}📦 Installing Minikube...${NC}"
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
install minikube-linux-amd64 /usr/local/bin/minikube
rm -f minikube-linux-amd64
echo -e "${GREEN}✅ Minikube installed${NC}"

# Install Jenkins
echo -e "${YELLOW}📦 Installing Jenkins...${NC}"
wget -O /etc/yum.repos.d/jenkins.repo https://pkg.jenkins.io/redhat-stable/jenkins.repo
rpm --import https://pkg.jenkins.io/redhat-stable/jenkins.io-2023.key
dnf install -y jenkins

# Configure Jenkins
usermod -aG docker jenkins
mkdir -p /var/lib/jenkins/.minikube
mkdir -p /var/lib/jenkins/.kube
chown -R jenkins:jenkins /var/lib/jenkins/.minikube
chown -R jenkins:jenkins /var/lib/jenkins/.kube

systemctl daemon-reload
systemctl start jenkins
systemctl enable jenkins
echo -e "${GREEN}✅ Jenkins installed${NC}"

# Install additional tools
echo -e "${YELLOW}📦 Installing additional tools...${NC}"
dnf install -y git curl wget jq unzip python3 python3-pip
echo -e "${GREEN}✅ Tools installed${NC}"

# Configure firewall
echo -e "${YELLOW}🔥 Configuring firewall...${NC}"
firewall-cmd --permanent --add-port=8080/tcp  # Jenkins
firewall-cmd --permanent --add-port=5001/tcp  # MLflow
firewall-cmd --permanent --add-port=30080/tcp # API
firewall-cmd --reload
echo -e "${GREEN}✅ Firewall configured${NC}"

# Get server IP
SERVER_IP=$(hostname -I | awk '{print $1}')

# Wait for Jenkins
echo -e "${YELLOW}⏳ Waiting for Jenkins...${NC}"
sleep 10

# Summary
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  ✅ INSTALLATION COMPLETE!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "${BLUE}📋 Installed:${NC}"
echo "  ✅ Java 17"
echo "  ✅ Docker"
echo "  ✅ kubectl"
echo "  ✅ Minikube"
echo "  ✅ Jenkins"
echo "  ✅ Python 3"
echo ""
echo -e "${YELLOW}⚠️  NEXT STEPS:${NC}"
echo ""
echo "1. ${YELLOW}Log out and log back in${NC} (for docker group)"
echo ""
echo "2. ${YELLOW}Start Minikube:${NC}"
echo "   ${BLUE}minikube start --driver=docker --cpus=2 --memory=4096${NC}"
echo ""
echo "3. ${YELLOW}Configure Jenkins Minikube access:${NC}"
echo "   ${BLUE}sudo ./scripts/configure-jenkins-minikube.sh${NC}"
echo ""
echo "4. ${YELLOW}Access Jenkins:${NC}"
echo "   URL: ${BLUE}http://${SERVER_IP}:8080${NC}"
echo "   Password:"
cat /var/lib/jenkins/secrets/initialAdminPassword 2>/dev/null || echo "   (run: sudo cat /var/lib/jenkins/secrets/initialAdminPassword)"
echo ""
echo "5. ${YELLOW}Configure Jenkins:${NC}"
echo "   - Install suggested plugins"
echo "   - Create admin user"
echo "   - Add GitHub credentials (ID: github-token)"
echo "   - Create pipeline job with Jenkinsfile"
echo ""
echo -e "${GREEN}============================================${NC}"

