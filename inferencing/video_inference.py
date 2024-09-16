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
import argparse
from time import time
from model import Model
import matplotlib.cm as cm

parser = argparse.ArgumentParser(
    description="Real-time video inference with depth model."
)
parser.add_argument("--model", type=str, help="Path to the depth model.")
parser.add_argument(
    "--input",
    type=str,
    help="Source of input video. 'camera' for webcam, path for stored video",
)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POOL_SIZE = 7
STRIDE = 1

model = Model().to(DEVICE)
model.load_state_dict(torch.load(args.model, map_location=DEVICE))

if args.input == "camera":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(args.input)

if not cap.isOpened():
    raise RuntimeError("Failed to open video source")

while True:
    ret, frame = cap.read()
    if not ret:
        break

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
        start = time()
        output_frame = model(image_tensor)
        end = time()

    output_frame_np = output_frame.cpu().squeeze().numpy()

    h, w = output_frame_np.shape
    new_h = (h - POOL_SIZE) // STRIDE + 1
    new_w = (w - POOL_SIZE) // STRIDE + 1

    pooled_image = np.zeros((new_h, new_w), dtype=output_frame_np.dtype)

    for i in range(0, h - POOL_SIZE + 1, STRIDE):
        for j in range(0, w - POOL_SIZE + 1, STRIDE):
            window = output_frame_np[i : i + POOL_SIZE, j : j + POOL_SIZE]
            pooled_image[i // STRIDE, j // STRIDE] = np.max(window)

    pooled_image_normalized = (pooled_image - pooled_image.min()) / (
        pooled_image.max() - pooled_image.min()
    )
    color_mapped = cm.autumn(pooled_image_normalized)
    color_mapped = (color_mapped[:, :, :3] * 255).astype(np.uint8)

    color_output_resized = cv2.resize(color_mapped, (320, 240))
    combined_frame = np.hstack((resized_frame, color_output_resized))

    cv2.imshow("Input and Depth Estimation", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
