#!/bin/bash

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    if ! python3.10 -c "import cv2" &>/dev/null; then
        echo "Dependencies missing in venv. Running setup..."
        ./setup.sh
        source venv/bin/activate
    fi
else
    echo "Virtual environment not found. Running setup first..."
    ./setup.sh
    source venv/bin/activate
fi

# Run the application with CoreML
echo "Starting Deepfake Live Camera..."
python3.10 run.py --execution-provider coreml 