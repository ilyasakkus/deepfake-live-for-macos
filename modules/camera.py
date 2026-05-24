"""
Camera processing module for real-time face swapping.
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Callable

# Try to import lip sync module
try:
    from .lip_sync import LipSyncProcessor
    LIP_SYNC_AVAILABLE = True
except ImportError:
    LIP_SYNC_AVAILABLE = False
    print("Lip sync module not available. Running without lip sync support.")

class CameraProcessor:
    """Handles camera input and face swapping in real-time."""

    PERFORMANCE_MODES = {
        "fast": {"skip_frames": 3, "resolution_scale": 0.4},
        "balanced": {"skip_frames": 2, "resolution_scale": 0.5},
        "quality": {"skip_frames": 1, "resolution_scale": 0.8},
    }
    
    def __init__(self, camera_id: int = 0, face_swapper=None):
        """
        Initialize camera processor.
        
        Args:
            camera_id: Camera device ID
            face_swapper: FaceSwapper instance
        """
        self.camera_id = camera_id
        self.face_swapper = face_swapper
        
        # Camera capture
        self.cap = None
        self.is_running = False
        
        # Threading
        self.capture_thread = None
        self.process_thread = None
        
        # Frame queues
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)
        
        # Callbacks
        self.frame_callback = None
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Performance settings
        self.skip_frames = 2  # Process every Nth frame
        self.frame_counter = 0
        self.resolution_scale = 0.5  # Scale down for processing
        self.performance_mode = "balanced"
        
        # Lip sync
        if LIP_SYNC_AVAILABLE:
            self.lip_sync = LipSyncProcessor()
            self.enable_lip_sync = True
        else:
            self.lip_sync = None
            self.enable_lip_sync = False
        
        # Store original frame for lip sync
        self.original_frame = None
        
    def set_source_image(self, image_path: str) -> bool:
        """
        Set source image for face swapping.
        
        Args:
            image_path: Path to source image
            
        Returns:
            True if successful
        """
        if not self.face_swapper:
            self._source_error = "Face swap engine is not initialized. Restart the app."
            return False
        self._source_error = None
        return self.face_swapper.set_source_face(image_path)

    def get_source_error(self) -> str:
        """Return the latest source image error from the face swapper."""
        if getattr(self, "_source_error", None):
            return self._source_error
        if self.face_swapper and getattr(self.face_swapper, "last_error", None):
            return self.face_swapper.last_error
        return (
            "No face detected. Use a clear front-facing portrait (JPG/PNG). "
            "If this is an iPhone HEIC photo, convert it to JPG or PNG first."
        )

    def is_model_loaded(self) -> bool:
        """Return whether the face swap model is loaded."""
        return bool(self.face_swapper and self.face_swapper.is_model_loaded())

    def get_model_error(self) -> str:
        """Return the latest model loading error."""
        if self.face_swapper:
            return self.face_swapper.get_model_error()
        return "Face swap engine is not initialized. Restart the app."
    
    def set_frame_callback(self, callback: Callable):
        """Set callback for processed frames."""
        self.frame_callback = callback
    
    def start(self) -> bool:
        """Start camera capture and processing."""
        if self.is_running:
            return True
        
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_id}")
            self.cap.release()
            self.cap = None
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # macOS specific optimizations
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.is_running = True
        
        # Start threads
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            name="camera-capture",
            daemon=True
        )
        self.process_thread = threading.Thread(
            target=self._process_loop,
            name="camera-process",
            daemon=True
        )
        
        self.capture_thread.start()
        self.process_thread.start()
        
        return True
    
    def stop(self):
        """Stop camera capture and processing."""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.process_thread:
            self.process_thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
    
    def _capture_loop(self):
        """Capture frames from camera."""
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self._put_latest(self.input_queue, frame)
            else:
                time.sleep(0.01)
    
    def _process_loop(self):
        """Process frames for face swapping."""
        last_processed_frame = None
        
        while self.is_running:
            try:
                frame = self.input_queue.get(timeout=0.1)
                # Frame skipping for performance
                self.frame_counter += 1
                
                if self.frame_counter % self.skip_frames == 0:
                    # Resize frame for processing
                    height, width = frame.shape[:2]
                    process_width = max(1, int(width * self.resolution_scale))
                    process_height = max(1, int(height * self.resolution_scale))
                    small_frame = cv2.resize(
                        frame, 
                        (process_width, process_height)
                    )
                    
                    # Process frame
                    if self.face_swapper:
                        processed_small = self.face_swapper.process_frame(small_frame)
                        # Resize back to original
                        processed_frame = cv2.resize(processed_small, (width, height))
                        
                        # Apply lip sync if enabled
                        if self._should_apply_lip_sync():
                            self.original_frame = frame.copy()
                            processed_frame = self.lip_sync.process_frame(
                                self.original_frame, 
                                processed_frame
                            )
                    else:
                        processed_frame = frame
                    
                    last_processed_frame = processed_frame
                else:
                    # Use last processed frame
                    processed_frame = last_processed_frame if last_processed_frame is not None else frame
                
                # Update FPS
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed > 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()
                
                output_frame = processed_frame.copy()
                self._draw_overlay(output_frame)
                
                # Send to callback
                if self.frame_callback:
                    self.frame_callback(output_frame)
                
                # Store in output queue
                self._put_latest(self.output_queue, output_frame)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in process loop: {e}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest processed frame."""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.fps
    
    def set_performance_mode(self, mode: str):
        """
        Set performance mode.
        
        Args:
            mode: 'fast', 'balanced', or 'quality'
        """
        if mode not in self.PERFORMANCE_MODES:
            raise ValueError(f"Unknown performance mode: {mode}")

        settings = self.PERFORMANCE_MODES[mode]
        self.performance_mode = mode
        self.skip_frames = settings["skip_frames"]
        self.resolution_scale = settings["resolution_scale"]
    
    def toggle_lip_sync(self):
        """Toggle lip sync on/off."""
        if self.is_lip_sync_available():
            self.enable_lip_sync = not self.enable_lip_sync
            return self.enable_lip_sync
        self.enable_lip_sync = False
        return False

    def is_lip_sync_available(self) -> bool:
        """Return whether lip sync can run with the current model setup."""
        return bool(LIP_SYNC_AVAILABLE and self.lip_sync and self.lip_sync.enabled)

    def _should_apply_lip_sync(self) -> bool:
        """Return whether lip sync should run for this frame."""
        return self.enable_lip_sync and self.is_lip_sync_available()

    def _put_latest(self, target_queue: queue.Queue, frame: np.ndarray):
        """Store a frame, dropping stale queued frames to keep latency low."""
        if target_queue.full():
            try:
                target_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            target_queue.put_nowait(frame)
        except queue.Full:
            pass

    def _draw_overlay(self, frame: np.ndarray):
        """Draw lightweight runtime telemetry on the frame."""
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Mode: {self.performance_mode.title()}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
