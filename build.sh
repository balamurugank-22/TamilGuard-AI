#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Installing Node.js dependencies and building frontend..."
cd frontend
npm install --include=optional
npm run build
cd ..

echo "Installing Python dependencies with CPU-only PyTorch..."
pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
