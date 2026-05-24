"""
Core face swapping functionality using InsightFace and ONNX Runtime.
"""

import os
import platform
import subprocess
import tempfile
import cv2
import numpy as np
from typing import Optional, List, Tuple
import insightface
from insightface.app import FaceAnalysis
from PIL import Image, ImageOps

class FaceSwapper:
    """Face swapping engine using InsightFace."""
    
    def __init__(self, execution_provider: str = 'coreml'):
        """
        Initialize face swapper.
        
        Args:
            execution_provider: Execution provider for ONNX Runtime
        """
        self.execution_provider = execution_provider
        self.inference_providers = self._get_providers()
        # Face detection is more reliable on CPU; CoreML is kept for the swapper model.
        self.detection_providers = ['CPUExecutionProvider']
        
        # Initialize face analysis
        self.face_app = FaceAnalysis(
            name='buffalo_l',
            providers=self.detection_providers
        )
        self._prepare_face_app(det_size=(640, 640), det_thresh=0.5)
        
        # Initialize face swapper model
        self.face_swapper = None
        self.model_error = None
        self._init_swapper()
        
        # Source face embedding
        self.source_face = None
        self.last_error = None
        
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

    def _prepare_face_app(self, det_size: Tuple[int, int], det_thresh: float = 0.5):
        """Prepare FaceAnalysis while tolerating InsightFace API differences."""
        try:
            self.face_app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
        except TypeError:
            self.face_app.prepare(ctx_id=0, det_size=det_size)
            det_model = getattr(self.face_app, "det_model", None)
            if det_model is not None and hasattr(det_model, "det_thresh"):
                det_model.det_thresh = det_thresh
    
    def _init_swapper(self):
        """Initialize face swapper model."""
        # First try local model
        model_path = os.path.join('models', 'inswapper_128.onnx')
        self.model_error = None
        
        if os.path.exists(model_path):
            print(f"Loading face swapper model from {model_path}")
            self.face_swapper = insightface.model_zoo.get_model(
                model_path,
                providers=self.inference_providers
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
                            providers=self.inference_providers
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
                    self.model_error = f"Missing face swap model: {model_path}"
                    
            except Exception as e:
                self.model_error = f"Error loading face swap model: {e}"
                print(self.model_error)

    def is_model_loaded(self) -> bool:
        """Return whether the face swap model is ready."""
        return self.face_swapper is not None

    def get_model_error(self) -> str:
        """Return a user-facing model loading error."""
        if self.model_error:
            return self.model_error
        return "Missing face swap model: models/inswapper_128.onnx"
    
    def set_source_face(self, image_path: str) -> bool:
        """
        Set source face from image.
        
        Args:
            image_path: Path to source face image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.last_error = None
            img = self._read_image(image_path)
            
            # Detect faces
            faces = self._detect_faces_for_source(img)
            if not faces:
                return self._fail(
                    "No face detected in source image. Try a clear, front-facing portrait with good lighting."
                )
            
            # Use the largest detected face in case the photo contains multiple people.
            self.source_face = max(faces, key=self._face_area)
            print(f"Source face loaded successfully from {image_path}")
            print(f"Face bbox: {self.source_face.bbox}")
            print(f"Face confidence: {self.source_face.det_score}")
            return True
            
        except Exception as e:
            self.last_error = f"Error setting source face: {e}"
            print(self.last_error)
            return False

    def _read_image(self, image_path: str) -> np.ndarray:
        """Read a source image robustly across path encodings and image modes."""
        ext = os.path.splitext(image_path)[1].lower()
        if ext in {'.heic', '.heif'}:
            converted = self._convert_heic_to_bgr(image_path)
            if converted is not None:
                return converted

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is not None:
            return self._apply_exif_orientation(image_path, img)

        try:
            image_data = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if img is not None:
                return self._apply_exif_orientation(image_path, img)
        except Exception as e:
            print(f"Warning: OpenCV imdecode failed for {image_path}: {e}")

        try:
            with Image.open(image_path) as pil_image:
                pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(
                f"Cannot read image ({ext or 'unknown format'}). "
                "Use JPG/PNG, or export iPhone photos as JPEG."
            ) from e

    def _apply_exif_orientation(self, image_path: str, img: np.ndarray) -> np.ndarray:
        """Fix rotation for images loaded by OpenCV without EXIF handling."""
        try:
            with Image.open(image_path) as pil_image:
                exif = pil_image.getexif()
                if not exif:
                    return img
                orientation = exif.get(274)
                if orientation == 3:
                    return cv2.rotate(img, cv2.ROTATE_180)
                if orientation == 6:
                    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                if orientation == 8:
                    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except Exception:
            pass
        return img

    def _convert_heic_to_bgr(self, image_path: str) -> Optional[np.ndarray]:
        """Convert HEIC/HEIF to BGR using macOS sips when available."""
        if platform.system() != 'Darwin':
            return None

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
            result = subprocess.run(
                ['sips', '-s', 'format', 'jpeg', image_path, '--out', tmp_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"Warning: HEIC conversion failed: {result.stderr.strip()}")
                return None
            return cv2.imread(tmp_path, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Warning: HEIC conversion error: {e}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _detect_faces_for_source(self, img: np.ndarray):
        """Detect source faces with several retries for difficult photos."""
        original_thresh = 0.5
        original_det_size = (640, 640)

        try:
            attempts = [
                (original_thresh, original_det_size, img),
                (0.35, original_det_size, img),
                (0.35, (960, 960), img),
                (0.25, (1280, 1280), img),
            ]

            height, width = img.shape[:2]
            longest_side = max(width, height)
            if longest_side < 900:
                scale = 900 / longest_side
                upscaled = cv2.resize(
                    img,
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_CUBIC,
                )
                attempts.append((0.35, original_det_size, upscaled))
                attempts.append((0.25, (1280, 1280), upscaled))

            for det_thresh, det_size, frame in attempts:
                self._prepare_face_app(det_size=det_size, det_thresh=det_thresh)

                faces = self.face_app.get(frame)
                if faces:
                    return faces

            return []
        finally:
            self._prepare_face_app(det_size=original_det_size, det_thresh=original_thresh)

    def _face_area(self, face) -> float:
        """Calculate face bbox area for picking the primary face."""
        x1, y1, x2, y2 = face.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _fail(self, message: str) -> bool:
        """Store and print the latest source-image error."""
        self.last_error = message
        print(f"Error: {message}")
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
            self._draw_model_warning(frame)
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

    def _draw_model_warning(self, frame: np.ndarray):
        """Draw a concise missing-model warning on the preview frame."""
        lines = [
            "Face swap model missing",
            "Add models/inswapper_128.onnx",
        ]
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (30, 80 + index * 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )
    
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
