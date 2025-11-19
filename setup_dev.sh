#!/bin/bash

echo "🚀 Setting up ProvenanceTrace development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install the package in development mode from packages directory
echo "Installing domynclaimalign in development mode..."
pip install -e packages/

echo "✅ Setup complete! Activate the environment with:"
echo "source .venv/bin/activate"