#!/bin/bash

# Install system dependencies if needed
apt-get update && apt-get install -y libgl1-mesa-glx

# Upgrade pip
pip install --upgrade pip

# Install Python packages
pip install -r requirements.txt

# Create directories
mkdir -p uploads static templates

echo "Build complete"