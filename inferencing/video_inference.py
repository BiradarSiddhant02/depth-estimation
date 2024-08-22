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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import time
from model import Model

# Argument parser for model path
parser = argparse.ArgumentParser(
    description="Real-time video inference with depth model."
)
parser.add_argument("model_path", type=str, help="Path to the depth model.")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load a single model from the provided path
model = Model().to(DEVICE)
model.load_state_dict(torch.load(args.model_path, weights_only=True))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Failed to open camera")

fig, axes = plt.subplots(1, 3, figsize=(20, 10))
axes = axes.flatten()

start_time = time.time()
frame_count = 0


def update(frame_idx):
    global start_time, frame_count

    ret, frame = cap.read()
    if not ret:
        return

    resized_frame = cv2.resize(frame, (320, 240))
    resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    image_tensor = (
        torch.from_numpy(resized_frame_rgb)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        output = model(image_tensor)

    output_np = output.cpu().squeeze().numpy()

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    axes[0].imshow(resized_frame_rgb)
    axes[0].set_title("Original Frame")
    axes[0].axis("off")

    axes[1].imshow(output_np, cmap="turbo")
    axes[1].set_title("Model Output")
    axes[1].axis("off")

    # Plot FPS
    axes[2].clear()
    axes[2].axis("off")
    axes[2].text(0.5, 0.5, f"FPS: {fps:.2f}", fontsize=12, ha="center", va="center")

    return axes


ani = animation.FuncAnimation(
    fig, update, interval=100, blit=False, cache_frame_data=False
)

plt.show()

cap.release()
cv2.destroyAllWindows()
