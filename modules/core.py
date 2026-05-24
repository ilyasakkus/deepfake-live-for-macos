"""
Core face swapping functionality using InsightFace and ONNX Runtime.
"""

import os
import cv2
import numpy as np
from typing import Optional, List, Tuple
import insightface
from insightface.app import FaceAnalysis

class FaceSwapper:
    """Face swapping engine using InsightFace."""
    
    def __init__(self, execution_provider: str = 'coreml'):
        """
        Initialize face swapper.
        
        Args:
            execution_provider: Execution provider for ONNX Runtime
        """
        self.execution_provider = execution_provider
        self.providers = self._get_providers()
        
        # Initialize face analysis
        self.face_app = FaceAnalysis(
            name='buffalo_l',
            providers=self.providers
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Initialize face swapper model
        self.face_swapper = None
        self._init_swapper()
        
        # Source face embedding
        self.source_face = None
        
    def _get_providers(self) -> List[str]:
        """Get ONNX Runtime providers based on execution provider."""
        if self.execution_provider == 'coreml':
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                if 'CoreMLExecutionProvider' in available_providers:
                    return ['CoreMLExecutionProvider', 'CPUExecutionProvider']

                print("Warning: CoreMLExecutionProvider is not available.")
                print(f"Available ONNX Runtime providers: {available_providers}")
                print("Falling back to CPUExecutionProvider.")
            except Exception as e:
                print(f"Warning: Could not inspect ONNX Runtime providers: {e}")
                print("Falling back to CPUExecutionProvider.")

        return ['CPUExecutionProvider']
    
    def _init_swapper(self):
        """Initialize face swapper model."""
        # First try local model
        model_path = os.path.join('models', 'inswapper_128.onnx')
        
        if os.path.exists(model_path):
            print(f"Loading face swapper model from {model_path}")
            self.face_swapper = insightface.model_zoo.get_model(
                model_path,
                providers=self.providers
            )
        else:
            print(f"Warning: Face swapper model not found at {model_path}")
            print("Trying to download from model zoo...")
            
            try:
                # Try to download from InsightFace model zoo
                from insightface.model_zoo import model_zoo
                model_urls = model_zoo.get_model_list()
                
                # Look for swapper model in the zoo
                for model_name in ['inswapper_128', 'inswapper']:
                    try:
                        self.face_swapper = insightface.model_zoo.get_model(
                            model_name,
                            providers=self.providers
                        )
                        print(f"Successfully loaded {model_name} from model zoo")
                        break
                    except:
                        continue
                        
                if self.face_swapper is None:
                    print("\n⚠️  IMPORTANT: Face swap model not found!")
                    print("Please download the model manually:")
                    print("1. Download 'inswapper_128.onnx' from:")
                    print("   https://drive.google.com/file/d/1HvZ4MAtzlY74Dk4ASGIS9L6Rg5oZdqvu/view")
                    print("2. Place it in the 'models' folder")
                    print(f"3. Full path should be: {os.path.abspath(model_path)}")
                    
            except Exception as e:
                print(f"Error loading from model zoo: {e}")
    
    def set_source_face(self, image_path: str) -> bool:
        """
        Set source face from image.
        
        Args:
            image_path: Path to source face image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Cannot read image from {image_path}")
                return False
            
            # Detect faces
            faces = self.face_app.get(img)
            if not faces:
                print("Error: No face detected in source image")
                return False
            
            # Use first detected face
            self.source_face = faces[0]
            print(f"Source face loaded successfully from {image_path}")
            print(f"Face bbox: {self.source_face.bbox}")
            print(f"Face confidence: {self.source_face.det_score}")
            return True
            
        except Exception as e:
            print(f"Error setting source face: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for face swapping.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame
        """
        if self.source_face is None:
            print("No source face set")
            return frame
            
        if self.face_swapper is None:
            # Draw warning on frame
            cv2.putText(
                frame,
                "Model not loaded! Check console for instructions",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            return frame
        
        try:
            # Detect faces in target frame
            target_faces = self.face_app.get(frame)
            
            if not target_faces:
                return frame
            
            # Process each detected face
            result = frame.copy()
            for target_face in target_faces:
                # Swap face
                result = self.face_swapper.get(
                    result,
                    target_face,
                    self.source_face,
                    paste_back=True
                )
            
            return result
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Draw error on frame
            cv2.putText(
                frame,
                f"Error: {str(e)[:50]}",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            return frame
    
    def enhance_face(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance face quality (placeholder for future implementation).
        
        Args:
            frame: Input frame
            
        Returns:
            Enhanced frame
        """
        # TODO: Implement face enhancement using GFPGAN or similar
        return frame
