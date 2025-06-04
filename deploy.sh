#!/usr/bin/env bash
set -e  # Exit on error

# Ensure we have the necessary packages
echo "Installing required deployment packages..."
pip install --upgrade pip build twine setuptools wheel

# Clean previous build artifacts
echo "Cleaning previous build artifacts..."
rm -rf ./build/ ./dist/ ./pysad.egg-info/

# Build the package
echo "Building package..."
python3 -m build

# Upload to PyPI
echo "Uploading to PyPI..."
python3 -m twine upload dist/*

echo "Deployment completed successfully!"
