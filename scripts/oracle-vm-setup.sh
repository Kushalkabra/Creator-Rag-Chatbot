#!/bin/bash
# Run once on a fresh Ubuntu 22.04/24.04 Oracle Ampere VM (as ubuntu user with sudo).
set -euo pipefail

echo "==> Installing Docker and tools..."
sudo apt-get update
sudo apt-get install -y git curl ca-certificates
sudo apt-get install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

echo ""
echo "==> Done. Log out and SSH back in so docker group applies."
echo "Then:"
echo "  git clone https://github.com/Kushalkabra/Creator-Rag-Chatbot.git"
echo "  cd Creator-Rag-Chatbot"
echo "  cp .env.oracle.example .env"
echo "  cp backend/.env.example backend/.env   # add GROQ + YOUTUBE keys"
echo "  # edit .env — set YOUR_PUBLIC_IP in all three variables"
echo "  docker compose up -d --build"
echo ""
echo "Open http://YOUR_PUBLIC_IP:3000"
