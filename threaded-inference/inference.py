import cv2
import threading
from collections import deque
import torch
import numpy as np
from model import Model
import signal
import sys
import time
from typing import Dict


class FPSCounter:
    def __init__(self, window_size: int = 60):
        """Initialize FPS counter with moving average window."""
        self.window_size = window_size
        self.times: Dict[str, deque] = {}
        self.fps: Dict[str, float] = {}
        self.locks: Dict[str, threading.Lock] = {}

    def init_stream(self, name: str):
        """Initialize a new FPS stream."""
        self.times[name] = deque(maxlen=self.window_size)
        self.fps[name] = 0.0
        self.locks[name] = threading.Lock()

    def update(self, name: str):
        """Update FPS for named stream."""
        with self.locks[name]:
            self.times[name].append(time.time())
            if len(self.times[name]) >= 2:
                fps = len(self.times[name]) / (
                    self.times[name][-1] - self.times[name][0]
                )
                self.fps[name] = round(fps, 1)

    def get_fps(self, name: str) -> float:
        """Get current FPS for named stream."""
        with self.locks[name]:
            return self.fps[name]


class DepthEstimator:
    def __init__(
        self,
        model_path: str = "models/Depth-Model.pth",
        frame_width: int = 320,
        frame_height: int = 240,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the depth estimator."""
        self.device = device
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Initialize frame and depth buffers
        self.current_frame = None
        self.current_depth = None
        self.frame_lock = threading.Lock()
        self.depth_lock = threading.Lock()
        self.running = True

        # Initialize FPS counter
        self.fps_counter = FPSCounter()
        self.fps_counter.init_stream("capture")
        self.fps_counter.init_stream("process")
        self.fps_counter.init_stream("display")

        # Load model
        try:
            self.model = Model().to(device)
            self.model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True)
            )
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a frame for model input."""
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        frame = np.transpose(frame, (2, 0, 1))  # Convert to (C, H, W)
        return frame / 255.0  # Normalize

    def capture_images(self):
        """Capture images from webcam."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        if not cap.isOpened():
            raise RuntimeError("Failed to open webcam")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue

            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)

            # Update current frame
            with self.frame_lock:
                self.current_frame = processed_frame

            # Update FPS
            self.fps_counter.update("capture")

        cap.release()

    def process_depth(self):
        """Process frames to generate depth maps."""
        while self.running:
            # Get latest frame
            with self.frame_lock:
                if self.current_frame is None:
                    continue
                current_frame = self.current_frame

            # Prepare batch
            batch = (
                torch.tensor(current_frame, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            # Inference
            with torch.no_grad():
                depth_pred = self.model(batch)

            # Process depth map
            depth_map = depth_pred[0].cpu().numpy().squeeze()
            depth_map = (depth_map - depth_map.min()) / (
                depth_map.max() - depth_map.min()
            )
            depth_map = cv2.resize(depth_map, (self.frame_width, self.frame_height))

            # Convert to colormap
            depth_colored = cv2.applyColorMap(
                (depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET
            )

            # Update current depth
            with self.depth_lock:
                self.current_depth = depth_colored

            # Update FPS
            self.fps_counter.update("process")

    def display_output(self):
        """Display the depth maps using OpenCV."""
        cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)

        while self.running:
            with self.depth_lock:
                if self.current_depth is None:
                    continue
                depth_display = self.current_depth.copy()

            # Add FPS information
            capture_fps = self.fps_counter.get_fps("capture")
            process_fps = self.fps_counter.get_fps("process")
            display_fps = self.fps_counter.get_fps("display")

            fps_text = f"Capture: {capture_fps} FPS | Process: {process_fps} FPS | Display: {display_fps} FPS"
            print(fps_text)

            # Display the frame
            cv2.imshow("Depth Map", depth_display)

            # Update FPS
            self.fps_counter.update("display")

            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False
                break

        cv2.destroyAllWindows()

    def run(self):
        """Run the depth estimation pipeline."""
        # Create and start threads
        capture_thread = threading.Thread(target=self.capture_images)
        process_thread = threading.Thread(target=self.process_depth)
        display_thread = threading.Thread(target=self.display_output)

        capture_thread.start()
        process_thread.start()
        display_thread.start()

        # Setup signal handler for clean shutdown
        def signal_handler(sig, frame):
            self.running = False
            cv2.destroyAllWindows()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Wait for threads
        capture_thread.join()
        process_thread.join()
        display_thread.join()

    def __del__(self):
        """Cleanup."""
        self.running = False
        cv2.destroyAllWindows()


# Usage
if __name__ == "__main__":
    estimator = DepthEstimator()
    estimator.run()
