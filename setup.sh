#!/bin/bash

# Exit on error
set -e

echo "Setting up environment for Language Models Use Trigonometry to Do Addition..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment 'helix' with Python 3.10..."
conda create -n helix python=3.10 -y

# Activate environment
echo "Activating helix environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate helix

# Install PyTorch with CUDA if available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    echo "CUDA not detected, installing CPU-only PyTorch..."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Create necessary directories if they don't exist
echo "Creating project directory structure..."
mkdir -p configs/
mkdir -p src/{data,models,analysis,visualization}
mkdir -p tests/
mkdir -p notebooks/

# Initialize git repository if not already initialized
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
    echo "__pycache__/" > .gitignore
    echo "*.pyc" >> .gitignore
    echo "*.pyo" >> .gitignore
    echo "*.pyd" >> .gitignore
    echo ".Python" >> .gitignore
    echo "env/" >> .gitignore
    echo "venv/" >> .gitignore
    echo ".env" >> .gitignore
    echo ".venv" >> .gitignore
    echo "wandb/" >> .gitignore
    echo ".pytest_cache/" >> .gitignore
    echo ".mypy_cache/" >> .gitignore
fi

echo "Setup complete! Activate the environment with: conda activate helix" 