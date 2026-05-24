"""
Utility functions for Deepfake Live Camera.
"""

import os
import sys
import platform
import subprocess
import urllib.request
from pathlib import Path
import gdown

def check_requirements() -> bool:
    """
    Check if all requirements are met.
    
    Returns:
        True if all requirements are met
    """
    try:
        # Check Python version
        if sys.version_info[:2] != (3, 10):
            print(f"Error: Python 3.10 is required, but you're using Python {sys.version}")
            print("Please install Python 3.10 and run with python3.10 command")
            return False
        
        # Check platform
        if platform.system() != "Darwin":
            print("Warning: This application is optimized for macOS")
        
        # Check if running on Apple Silicon
        if platform.machine() not in ["arm64", "aarch64"]:
            print("Warning: This application is optimized for Apple Silicon (M1/M2/M3)")
        
        # Check required packages
        required_packages = [
            "cv2",
            "numpy",
            "insightface",
            "onnxruntime",
            "customtkinter",
            "PIL"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"Missing packages: {', '.join(missing_packages)}")
            print("Please install requirements: pip install -r requirements.txt")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error checking requirements: {e}")
        return False

def download_models() -> bool:
    """
    Download required models if not present.
    
    Returns:
        True if models are available
    """
    try:
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Check for face swapper model
        swapper_model_path = models_dir / "inswapper_128.onnx"
        
        if not swapper_model_path.exists():
            print("Downloading face swapper model...")
            
            # Google Drive file ID for inswapper_128.onnx
            file_id = "1HvZ4MAtzlY74Dk4ASGIS9L6Rg5oZdqvu"
            url = f"https://drive.google.com/uc?id={file_id}"
            
            try:
                # Use gdown to download from Google Drive
                gdown.download(url, str(swapper_model_path), quiet=False)
                if _is_valid_model_file(swapper_model_path):
                    print("Face swapper model downloaded successfully")
                else:
                    raise RuntimeError("Downloaded model file is missing or too small")
            except Exception as e:
                print(f"Error downloading face swapper model: {e}")
                if swapper_model_path.exists():
                    swapper_model_path.unlink()
                
                # Alternative download method
                print("Trying alternative download method...")
                alternative_url = "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx"
                
                try:
                    urllib.request.urlretrieve(alternative_url, str(swapper_model_path))
                    if _is_valid_model_file(swapper_model_path):
                        print("Face swapper model downloaded successfully (alternative method)")
                    else:
                        raise RuntimeError("Downloaded model file is missing or too small")
                except Exception as e2:
                    if swapper_model_path.exists():
                        swapper_model_path.unlink()
                    print(f"Alternative download also failed: {e2}")
                    print("\nPlease download the model manually:")
                    print(f"1. Download inswapper_128.onnx from: {alternative_url}")
                    print(f"2. Place it in: {swapper_model_path}")
                    return False
        
        # Check for InsightFace models (will be downloaded automatically by InsightFace)
        insightface_dir = Path.home() / ".insightface"
        
        if not insightface_dir.exists():
            print("InsightFace models will be downloaded on first use...")
        
        return True
        
    except Exception as e:
        print(f"Error in model download: {e}")
        return False

def _is_valid_model_file(path: Path) -> bool:
    """Return whether a downloaded ONNX model looks complete enough to use."""
    return path.exists() and path.stat().st_size > 100 * 1024 * 1024

def get_available_cameras() -> list:
    """
    Get list of available camera devices.
    
    Returns:
        List of tuples (index, name)
    """
    cameras = []
    
    # Check up to 5 camera indices
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append((i, f"Camera {i}"))
            cap.release()
    
    return cameras

def optimize_for_macos():
    """
    Apply macOS-specific optimizations.
    """
    try:
        # Set environment variables for better performance
        os.environ['OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION'] = '1'
        os.environ['QT_MAC_WANTS_LAYER'] = '1'
        
        # Disable App Nap for better performance
        if platform.system() == "Darwin":
            try:
                subprocess.run(
                    ["defaults", "write", "NSGlobalDomain", "NSAppSleepDisabled", "-bool", "YES"],
                    check=True
                )
            except:
                pass
                
    except Exception as e:
        print(f"Warning: Could not apply macOS optimizations: {e}")

# Import OpenCV to ensure it's available
try:
    import cv2
except ImportError:
    print("Error: OpenCV not found. Please install: pip install opencv-python")
