#!/usr/bin/env python3
"""Test script to debug face swapping issues."""

import sys
import cv2
import numpy as np
from modules.core import FaceSwapper

def test_face_swap():
    print("=== Face Swap Test ===\n")
    
    # Initialize face swapper
    print("1. Initializing face swapper...")
    face_swapper = FaceSwapper(execution_provider='cpu')  # Use CPU for testing
    
    # Check if model is loaded
    if face_swapper.face_swapper is None:
        print("\n❌ Face swap model is not loaded!")
        print("Please download the model as instructed above.")
        return False
    else:
        print("✅ Face swap model loaded successfully!")
    
    # Test face detection
    print("\n2. Testing face detection...")
    
    # Create a simple test image (you can replace with actual image path)
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    try:
        faces = face_swapper.face_app.get(test_image)
        print(f"✅ Face detection working. Found {len(faces)} faces.")
    except Exception as e:
        print(f"❌ Face detection error: {e}")
        return False
    
    print("\n3. Checking InsightFace models...")
    import os
    insightface_path = os.path.expanduser("~/.insightface/models")
    if os.path.exists(insightface_path):
        print(f"InsightFace models directory: {insightface_path}")
        for root, dirs, files in os.walk(insightface_path):
            for file in files:
                print(f"  - {os.path.join(root, file)}")
    else:
        print("InsightFace models directory not found.")
    
    return True

if __name__ == "__main__":
    test_face_swap() 