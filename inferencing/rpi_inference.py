import torch
import cv2
import matplotlib.pyplot as plt
from model import Model
import argparse
import os
from time import time
import numpy as np

# Argument parser
parser = argparse.ArgumentParser(description="Depth Model Inference")
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help='Input source: "camera" or path to an image file',
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path to the folder where outputs will be saved",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Path to the saved model",
)
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POOL_SIZE = 7
STRIDE = 1

print("Loading model...")
# Load model
depth_model = Model().to(DEVICE)
depth_model.load_state_dict(
    torch.load(args.model, weights_only=False, map_location="cpu")
)

print("Model loaded successfully.")

print("Preparing input...")
# Get input image
if args.input.lower() == "camera":
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture image from camera.")
    print("Captured image from camera.")
elif os.path.isfile(args.input):
    frame = cv2.imread(args.input)
    if frame is None:
        raise RuntimeError(f"Failed to load image from {args.input}")
    print(f"Loaded image from {args.input}.")
else:
    raise ValueError('Invalid input. Use "camera" or provide a valid image path.')

print("Resizing and preprocessing image...")
# Resize and preprocess image
resized_frame = cv2.resize(frame, (320, 240))
resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
image_tensor = (
    torch.from_numpy(resized_frame_rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
)

depth_map_path = args.input.replace("colors", "depth")
ground_truth = cv2.resize(cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE), (320, 240))

print("Running inference...")
# Inference
with torch.no_grad():
    start = time()
    output_0 = depth_model(image_tensor)
    end = time()

print(f"Inference completed in {end - start:.2f} seconds.")

output_0_np = output_0.cpu().squeeze().numpy()

print("Processing output...")
h, w = output_0_np.shape
new_h = (h - POOL_SIZE) // STRIDE + 1
new_w = (w - POOL_SIZE) // STRIDE + 1

pooled_image = np.zeros((new_h, new_w), dtype=output_0_np.dtype)

for i in range(0, h - POOL_SIZE + 1, STRIDE):
    for j in range(0, w - POOL_SIZE + 1, STRIDE):
        window = output_0_np[i : i + POOL_SIZE, j : j + POOL_SIZE]
        pooled_image[i // STRIDE, j // STRIDE] = np.max(window)

print("Creating and saving visualization...")
# Visualization
fig, axs = plt.subplots(2, 2, figsize=(18, 6))

axs[0, 0].imshow(resized_frame_rgb)
axs[0, 0].set_title("Original Image")
axs[0, 0].axis("off")

axs[0, 1].imshow(output_0_np, cmap="viridis")
axs[0, 1].set_title("Depth Map")
axs[0, 1].axis("off")

axs[1, 0].imshow(pooled_image, cmap="viridis")
axs[1, 0].set_title("Pooled Depth Map")
axs[1, 0].axis("off")

axs[1, 1].imshow(ground_truth, cmap="viridis")
axs[1, 1].set_title("Ground Truth")
axs[1, 1].axis("off")

# Save the figure
plt.tight_layout()
plt.savefig(os.path.join(args.output, "output_comparison.png"))
plt.show()

print(f"Visualization saved to {os.path.join(args.output, 'output_comparison.png')}.")
