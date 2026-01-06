#!/bin/bash
#
# Configure Jenkins to Access Minikube
# =====================================
# Run AFTER Minikube is started
# Usage: sudo ./scripts/configure-jenkins-minikube.sh
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Configure Jenkins Minikube Access        ${NC}"
echo -e "${BLUE}============================================${NC}"

if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root: sudo $0${NC}"
    exit 1
fi

ACTUAL_USER=${SUDO_USER:-$USER}
USER_HOME=$(eval echo ~$ACTUAL_USER)

# Check Minikube is running
echo -e "${YELLOW}1. Checking Minikube...${NC}"
if ! sudo -u $ACTUAL_USER minikube status &> /dev/null; then
    echo -e "${RED}❌ Minikube not running!${NC}"
    echo -e "${YELLOW}Start it first: minikube start --driver=docker${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Minikube is running${NC}"

# Copy configs to Jenkins
echo -e "${YELLOW}2. Copying configs to Jenkins...${NC}"
mkdir -p /var/lib/jenkins/.minikube
mkdir -p /var/lib/jenkins/.kube

cp -r $USER_HOME/.minikube/* /var/lib/jenkins/.minikube/ 2>/dev/null || true
cp -r $USER_HOME/.kube/* /var/lib/jenkins/.kube/ 2>/dev/null || true

chown -R jenkins:jenkins /var/lib/jenkins/.minikube
chown -R jenkins:jenkins /var/lib/jenkins/.kube
chmod -R 755 /var/lib/jenkins/.minikube
chmod -R 755 /var/lib/jenkins/.kube
echo -e "${GREEN}✅ Configs copied${NC}"

# Set environment variables
echo -e "${YELLOW}3. Setting environment variables...${NC}"
MINIKUBE_PROFILE=$(sudo -u $ACTUAL_USER minikube profile 2>/dev/null || echo "minikube")

mkdir -p /etc/systemd/system/jenkins.service.d/
cat > /etc/systemd/system/jenkins.service.d/override.conf << EOF
[Service]
Environment="MINIKUBE_HOME=/var/lib/jenkins/.minikube"
Environment="KUBECONFIG=/var/lib/jenkins/.kube/config"
Environment="DOCKER_HOST=unix:///var/run/docker.sock"
Environment="MINIKUBE_ACTIVE_DOCKERD=$MINIKUBE_PROFILE"
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
EOF
echo -e "${GREEN}✅ Environment configured${NC}"

# Fix Docker socket
echo -e "${YELLOW}4. Fixing Docker socket...${NC}"
chmod 666 /var/run/docker.sock 2>/dev/null || true
echo -e "${GREEN}✅ Docker socket configured${NC}"

# Restart Jenkins
echo -e "${YELLOW}5. Restarting Jenkins...${NC}"
systemctl daemon-reload
systemctl restart jenkins
sleep 5
echo -e "${GREEN}✅ Jenkins restarted${NC}"

# Verify
echo -e "${YELLOW}6. Verifying...${NC}"
if sudo -u jenkins docker ps &> /dev/null; then
    echo -e "${GREEN}✅ Jenkins can access Docker${NC}"
else
    echo -e "${RED}❌ Jenkins cannot access Docker${NC}"
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  ✅ Configuration Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Next: Trigger a Jenkins build"

