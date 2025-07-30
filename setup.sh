#!/bin/bash

# Define your Python version and environment name
PYTHON_VERSION=3.12
ENV_NAME=venv

# Check if desired Python version is installed
if ! command -v python$PYTHON_VERSION &> /dev/null
then
    echo "Python version $PYTHON_VERSION could not be found, please install it first"
    exit
fi

# Create a new virtualenv
echo "Creating new virtual environment..."
uv venv

# Activate the virtualenv
echo "Activating the virtual environment..."
source .venv/bin/activate

# Install the packages from requirements.lock
echo "Installing packages..."
if [ ! -f "requirements.lock" ] || [ "requirements.txt" -nt "requirements.lock" ]; then
    echo "Compiling requirements.lock..."
    uv pip compile requirements.txt -o requirements.lock --index-url https://artifactory.global.square/artifactory/api/pypi/block-pypi/simple
fi
uv pip install -r requirements.lock --index-url https://artifactory.global.square/artifactory/api/pypi/block-pypi/simple

echo "Setup complete!"
