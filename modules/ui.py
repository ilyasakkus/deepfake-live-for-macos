"""
GUI module for Deepfake Live Camera application.
"""

import queue
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class DeepfakeApp:
    """Main GUI application for Deepfake Live Camera."""
    
    def __init__(self, face_swapper, camera_processor):
        """Initialize the application."""
        self.face_swapper = face_swapper
        self.camera_processor = camera_processor
        
        # Main window
        self.root = ctk.CTk()
        self.root.title("Deepfake Live Camera")
        self.root.geometry("1200x800")
        
        # Variables
        self.source_image_path = None
        self.is_live = False
        self.is_closing = False
        self.preview_label = None
        self.preview_queue = queue.Queue(maxsize=1)
        self.fps_update_job = None
        
        # Setup UI
        self._setup_ui()
        
        # Set frame callback
        self.camera_processor.set_frame_callback(self._queue_preview_frame)
        self.root.after(15, self._drain_preview_queue)
        
    def _setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Controls
        left_panel = ctk.CTkFrame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Title
        title_label = ctk.CTkLabel(
            left_panel, 
            text="Deepfake Live Camera", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=20)
        
        # Source image section
        source_frame = ctk.CTkFrame(left_panel)
        source_frame.pack(fill=tk.X, padx=20, pady=10)
        
        source_label = ctk.CTkLabel(
            source_frame, 
            text="Source Face Image", 
            font=ctk.CTkFont(size=16)
        )
        source_label.pack(pady=5)
        
        # Source image preview
        self.source_preview = ctk.CTkLabel(
            source_frame, 
            text="No image selected", 
            width=200, 
            height=200
        )
        self.source_preview.pack(pady=10)
        
        # Select image button
        select_btn = ctk.CTkButton(
            source_frame, 
            text="Select Image", 
            command=self._select_source_image,
            width=200
        )
        select_btn.pack(pady=5)
        
        # Control buttons
        control_frame = ctk.CTkFrame(left_panel)
        control_frame.pack(fill=tk.X, padx=20, pady=20)
        
        self.live_btn = ctk.CTkButton(
            control_frame, 
            text="Start Live", 
            command=self._toggle_live,
            width=200,
            height=40,
            font=ctk.CTkFont(size=16)
        )
        self.live_btn.pack(pady=10)
        
        # Performance settings
        performance_label = ctk.CTkLabel(
            control_frame, 
            text="Performance Mode", 
            font=ctk.CTkFont(size=14)
        )
        performance_label.pack(pady=5)
        
        self.performance_mode = ctk.StringVar(value="fast")
        
        performance_frame = ctk.CTkFrame(control_frame)
        performance_frame.pack(pady=5)
        
        modes = [("Fast", "fast"), ("Balanced", "balanced"), ("Quality", "quality")]
        for text, mode in modes:
            btn = ctk.CTkRadioButton(
                performance_frame,
                text=text,
                variable=self.performance_mode,
                value=mode,
                command=self._update_performance_mode
            )
            btn.pack(side=tk.LEFT, padx=5)
        
        # Lip sync toggle
        self.lip_sync_btn = ctk.CTkButton(
            control_frame,
            text="Lip Sync: ON" if self.camera_processor.is_lip_sync_available() else "Lip Sync: N/A",
            command=self._toggle_lip_sync,
            width=200,
            height=30
        )
        self.lip_sync_btn.pack(pady=10)
        
        # FPS display
        self.fps_label = ctk.CTkLabel(
            control_frame, 
            text="FPS: 0.0", 
            font=ctk.CTkFont(size=14)
        )
        self.fps_label.pack(pady=5)
        
        # Instructions
        instructions_frame = ctk.CTkFrame(left_panel)
        instructions_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        instructions_label = ctk.CTkLabel(
            instructions_frame, 
            text="Instructions:", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        instructions_label.pack(pady=5)
        
        instructions_text = """1. Select a source face image
2. Choose performance mode
3. Click 'Start Live' to begin
4. Use OBS Studio to capture:
   - Add 'Window Capture' source
   - Select this preview window
5. Click 'Stop Live' to end

Performance Modes:
- Fast: Lower quality, higher FPS
- Balanced: Good quality and FPS
- Quality: Best quality, lower FPS

Features:
- Lip Sync: Transfers mouth movements
- Auto optimization for M2 chip"""
        
        instructions_content = ctk.CTkLabel(
            instructions_frame, 
            text=instructions_text,
            font=ctk.CTkFont(size=12),
            justify=tk.LEFT
        )
        instructions_content.pack(pady=10, padx=10)
        
        # Right panel - Preview
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        preview_label = ctk.CTkLabel(
            right_panel, 
            text="Camera Preview", 
            font=ctk.CTkFont(size=18)
        )
        preview_label.pack(pady=10)
        
        # Preview area
        self.preview_label = ctk.CTkLabel(
            right_panel, 
            text="Camera preview will appear here\n\nClick 'Start Live' to begin", 
            width=800, 
            height=600
        )
        self.preview_label.pack(pady=10, padx=10)
        
    def _select_source_image(self):
        """Handle source image selection."""
        file_path = filedialog.askopenfilename(
            title="Select Source Face Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.heic *.heif"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("HEIC (iPhone)", "*.heic *.heif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and resize image for preview
                image = Image.open(file_path)
                image.thumbnail((200, 200), Image.Resampling.LANCZOS)
                photo = ctk.CTkImage(light_image=image, dark_image=image, size=(200, 200))
                
                # Set source face
                if self.camera_processor.set_source_image(file_path):
                    self.source_image_path = file_path
                    self.source_preview.configure(image=photo, text="")
                    messagebox.showinfo("Success", "Source face loaded successfully!")
                else:
                    self.source_image_path = None
                    messagebox.showerror("Error", self.camera_processor.get_source_error())
                    
            except Exception as e:
                self.source_image_path = None
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def _toggle_live(self):
        """Toggle live camera mode."""
        if not self.is_live:
            if not self.source_image_path:
                messagebox.showwarning("Warning", "Please select a source face image first")
                return

            if not self.camera_processor.is_model_loaded():
                messagebox.showerror(
                    "Missing Model",
                    self.camera_processor.get_model_error()
                )
                return
            
            # Apply performance mode before starting
            self._update_performance_mode()
            
            # Start camera
            if self.camera_processor.start():
                self.is_live = True
                self.live_btn.configure(text="Stop Live")
                self._schedule_fps_update()
            else:
                messagebox.showerror("Error", "Failed to start camera")
        else:
            # Stop camera
            self.camera_processor.stop()
            self.is_live = False
            self._cancel_fps_update()
            self.live_btn.configure(text="Start Live")
            self.preview_label.configure(
                image=None, 
                text="Camera preview will appear here\n\nClick 'Start Live' to begin"
            )
    
    def _update_performance_mode(self):
        """Update performance mode."""
        mode = self.performance_mode.get()
        self.camera_processor.set_performance_mode(mode)
        print(f"Performance mode set to: {mode}")
    
    def _toggle_lip_sync(self):
        """Toggle lip sync feature."""
        enabled = self.camera_processor.toggle_lip_sync()
        if self.camera_processor.is_lip_sync_available():
            self.lip_sync_btn.configure(text=f"Lip Sync: {'ON' if enabled else 'OFF'}")
        else:
            self.lip_sync_btn.configure(text="Lip Sync: N/A")
    
    def _queue_preview_frame(self, frame):
        """Receive a processed frame from the camera thread."""
        if self.preview_queue.full():
            try:
                self.preview_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self.preview_queue.put_nowait(frame)
        except queue.Full:
            pass

    def _drain_preview_queue(self):
        """Render the latest queued preview frame on Tk's main loop."""
        if self.is_closing:
            return

        if not self.is_live:
            self.root.after(15, self._drain_preview_queue)
            return

        frame = None
        while True:
            try:
                frame = self.preview_queue.get_nowait()
            except queue.Empty:
                break

        if frame is not None:
            self._update_preview(frame)

        self.root.after(15, self._drain_preview_queue)

    def _update_preview(self, frame):
        """Update preview with processed frame."""
        if not self.is_live:
            return
        
        try:
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for display
            height, width = frame_rgb.shape[:2]
            max_width = 800
            max_height = 600
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update label
            self.preview_label.configure(image=photo, text="")
            self.preview_label.image = photo  # Keep reference
            
        except Exception as e:
            print(f"Error updating preview: {e}")
    
    def _update_fps(self):
        """Update FPS display."""
        if not self.is_live:
            self.fps_update_job = None
            return

        fps = self.camera_processor.get_fps()
        self.fps_label.configure(text=f"FPS: {fps:.1f}")
        self.fps_update_job = self.root.after(500, self._update_fps)

    def _schedule_fps_update(self):
        """Start the FPS refresh loop."""
        if self.fps_update_job is None:
            self._update_fps()

    def _cancel_fps_update(self):
        """Stop the FPS refresh loop."""
        if self.fps_update_job is not None:
            self.root.after_cancel(self.fps_update_job)
            self.fps_update_job = None
    
    def run(self):
        """Run the application."""
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Start main loop
        self.root.mainloop()
    
    def _on_closing(self):
        """Handle window closing."""
        self.is_closing = True
        if self.is_live:
            self.camera_processor.stop()
            self._cancel_fps_update()
        self.root.destroy()
