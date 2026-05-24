#!/bin/bash

echo "==================================="
echo "Deepfake Live Camera Setup for macOS"
echo "==================================="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: This script is for macOS only"
    exit 1
fi

# Check if Python 3.10 is installed
if ! command -v python3.10 &> /dev/null; then
    echo "Python 3.10 not found. Installing via Homebrew..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install Homebrew first:"
        echo "https://brew.sh"
        exit 1
    fi
    
    # Install Python 3.10
    brew install python@3.10
    brew install python-tk@3.10
fi

echo "Python 3.10 found: $(python3.10 --version)"

# Create virtual environment
echo "Creating virtual environment..."
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements (dlib is optional — only needed for lip sync)
echo "Installing requirements..."
pip install -r requirements.txt || {
    echo "Full install failed (often dlib). Installing core dependencies..."
    grep -v '^dlib' requirements.txt | pip install -r /dev/stdin
    echo ""
    echo "Note: dlib failed to install. Lip sync will be unavailable."
    echo "To enable lip sync later: brew install cmake && pip install dlib==19.24.1"
}

# Create models directory
mkdir -p models

echo ""
echo "==================================="
echo "Setup completed successfully!"
echo "==================================="
echo ""
echo "To run the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run: python3.10 run.py --execution-provider coreml"
echo ""
echo "Or simply run: ./start.sh" 