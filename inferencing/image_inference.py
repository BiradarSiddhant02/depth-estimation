# This file is part of depth-estimation.
#
# Portions of this code are derived from the [Original Repository Name] project,
# which is licensed under the MIT License.
#
# Copyright (c) 2024 Siddhant Biradar.
#
# See the LICENSE.md file in the root of the repository for more details.

import torch
import cv2
import matplotlib.pyplot as plt # type: ignore
from model import TransferLearning, CustomUNET1, CustomUNET2
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
    help="Path to the folder where the model is saved",
)
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POOL_SIZE = 7
STRIDE = 1
CMAP = "viridis"

tokens = args.model.split("/")
MODEL_CLASS = tokens[-2]
MODEL_GEN = tokens[-1]
print(MODEL_CLASS)

# Load models
if MODEL_CLASS == "Transfer-Learning":
    depth_model_0 = TransferLearning().to(DEVICE)
    depth_model_0.load_state_dict(torch.load(f"{args.model}/0.pth", weights_only=True))

    depth_model_1 = TransferLearning().to(DEVICE)
    depth_model_1.load_state_dict(torch.load(f"{args.model}/1.pth", weights_only=True))

    depth_model_2 = TransferLearning().to(DEVICE)
    depth_model_2.load_state_dict(torch.load(f"{args.model}/2.pth", weights_only=True))

    depth_model_3 = TransferLearning().to(DEVICE)
    depth_model_3.load_state_dict(torch.load(f"{args.model}/3.pth", weights_only=True))

elif MODEL_CLASS == "custom-UNET" and MODEL_GEN == "gen-1":
    depth_model_0 = CustomUNET1().to(DEVICE)
    depth_model_0.load_state_dict(torch.load(f"{args.model}/0.pth", weights_only=True))

    depth_model_1 = CustomUNET1().to(DEVICE)
    depth_model_1.load_state_dict(torch.load(f"{args.model}/1.pth", weights_only=True))

    depth_model_2 = CustomUNET1().to(DEVICE)
    depth_model_2.load_state_dict(torch.load(f"{args.model}/2.pth", weights_only=True))

    depth_model_3 = CustomUNET1().to(DEVICE)
    depth_model_3.load_state_dict(torch.load(f"{args.model}/3.pth", weights_only=True))  

elif MODEL_CLASS == "custom-UNET" and MODEL_GEN == "gen-2":
    depth_model_0 = CustomUNET2().to(DEVICE)
    depth_model_0.load_state_dict(torch.load(f"{args.model}/0.pth", weights_only=True))

    depth_model_1 = CustomUNET2().to(DEVICE)
    depth_model_1.load_state_dict(torch.load(f"{args.model}/1.pth", weights_only=True))

    depth_model_2 = CustomUNET2().to(DEVICE)
    depth_model_2.load_state_dict(torch.load(f"{args.model}/2.pth", weights_only=True))

    depth_model_3 = CustomUNET2().to(DEVICE)
    depth_model_3.load_state_dict(torch.load(f"{args.model}/3.pth", weights_only=True))    

# Get input image
if args.input.lower() == "camera":
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture image from camera.")
elif os.path.isfile(args.input):
    frame = cv2.imread(args.input)
    if frame is None:
        raise RuntimeError(f"Failed to load image from {args.input}")
else:
    raise ValueError('Invalid input. Use "camera" or provide a valid image path.')

# Resize and preprocess image
resized_frame = cv2.resize(frame, (320, 240))
resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
image_tensor = (
    torch.from_numpy(resized_frame_rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
)

ground_truth = np.ones((240, 320), dtype=np.float32)

if args.input != "camera":
    depth_map_path = args.input.replace("colors", "depth")
    ground_truth = cv2.resize(cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE), (320, 240))

# Inference
with torch.no_grad():
    start = time()
    output_0 = depth_model_0(image_tensor)
    output_1 = depth_model_1(image_tensor)
    output_2 = depth_model_2(image_tensor)
    output_3 = depth_model_3(image_tensor)
    end = time()

output_0_np = output_0.cpu().squeeze().numpy()
output_1_np = output_1.cpu().squeeze().numpy()
output_2_np = output_2.cpu().squeeze().numpy()
output_3_np = output_3.cpu().squeeze().numpy()

combined_output = (output_0_np + output_1_np + output_2_np + output_3_np) / 255.0

h, w = combined_output.shape
new_h = (h - POOL_SIZE) // STRIDE + 1
new_w = (w - POOL_SIZE) // STRIDE + 1

pooled_image = np.zeros((new_h, new_w), dtype=combined_output.dtype)

for i in range(0, h - POOL_SIZE + 1, STRIDE):
    for j in range(0, w - POOL_SIZE + 1, STRIDE):
        window = combined_output[i : i + POOL_SIZE, j : j + POOL_SIZE]
        pooled_image[i // STRIDE, j // STRIDE] = np.max(window)

fig, axs = plt.subplots(2, 4, figsize=(25, 10))

axs[0, 0].imshow(resized_frame_rgb)
axs[0, 0].set_title("Original Image")
axs[0, 0].axis("off")

axs[0, 1].imshow(ground_truth, cmap=CMAP)
axs[0, 1].set_title("Original Depth")
axs[0, 1].axis("off")

axs[0, 2].imshow(1 - output_0_np, cmap=CMAP)
axs[0, 2].set_title("Output 0")
axs[0, 2].axis("off")

axs[0, 3].imshow(1 - output_1_np, cmap=CMAP)
axs[0, 3].set_title("Output 1")
axs[0, 3].axis("off")

axs[1, 0].imshow(1 - output_2_np, cmap=CMAP)
axs[1, 0].set_title("Output 2")
axs[1, 0].axis("off")

axs[1, 1].imshow(1 - output_3_np, cmap=CMAP)
axs[1, 1].set_title("Output 3")
axs[1, 1].axis("off")

axs[1, 2].imshow(1 - combined_output, cmap=CMAP)
axs[1, 2].set_title("Combined Output")
axs[1, 2].axis("off")

axs[1, 3].imshow(1 - pooled_image, cmap=CMAP)
axs[1, 3].set_title("Pooled Output")
axs[1, 3].axis("off")

# Save the figure
plt.savefig(os.path.join(args.output, "output_comparison.png"))

# Save individual outputs
output_filenames = ["output_0.png", "output_1.png", "output_2.png", "output_3.png"]
outputs_np = [output_0_np, output_1_np, output_2_np, output_3_np]

for filename, output_np in zip(output_filenames, outputs_np):
    plt.imsave(os.path.join(args.output, filename), output_np, cmap="viridis")

plt.show()
print((end - start) / 4)
