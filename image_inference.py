import torch
import cv2
import matplotlib.pyplot as plt
from model import Model
import argparse
import os
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Depth Model Inference")
parser.add_argument("--input", type=str, required=True, help='Input source: "camera" or path to an image file')
parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder where outputs will be saved")
args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

depth_model_0 = Model().to(DEVICE)
depth_model_0.load_state_dict(torch.load("models/0.pth", weights_only=True))

depth_model_1 = Model().to(DEVICE)
depth_model_1.load_state_dict(torch.load("models/1.pth", weights_only=True))

depth_model_2 = Model().to(DEVICE)
depth_model_2.load_state_dict(torch.load("models/2.pth", weights_only=True))

depth_model_3 = Model().to(DEVICE)
depth_model_3.load_state_dict(torch.load("models/3.pth", weights_only=True))

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

resized_frame = cv2.resize(frame, (320, 240))
resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
image_tensor = torch.from_numpy(resized_frame_rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output_0 = depth_model_0(image_tensor)
    output_1 = depth_model_1(image_tensor)
    output_2 = depth_model_2(image_tensor)
    output_3 = depth_model_3(image_tensor)

output_0_np = output_0.cpu().squeeze().numpy()
output_1_np = output_1.cpu().squeeze().numpy()
output_2_np = output_2.cpu().squeeze().numpy()
output_3_np = output_3.cpu().squeeze().numpy()

combined_output = (output_0_np + output_1_np + output_2_np + output_3_np) / 255.

combined_output_tensor = torch.from_numpy(combined_output).unsqueeze(0).unsqueeze(0)
max_pooled_output = F.max_pool2d(combined_output_tensor, kernel_size=7, stride=1, padding=2)
max_pooled_output_np = max_pooled_output.squeeze().numpy()

fig, axs = plt.subplots(2, 3, figsize=(18, 12))

axs[0, 0].imshow(resized_frame_rgb)
axs[0, 0].set_title("Original Image")
axs[0, 0].axis("off")

axs[0, 1].imshow(output_0_np, cmap="viridis")
axs[0, 1].set_title("Output 0")
axs[0, 1].axis("off")

axs[0, 2].imshow(output_1_np, cmap="viridis")
axs[0, 2].set_title("Output 1")
axs[0, 2].axis("off")

axs[1, 0].imshow(output_2_np, cmap="viridis")
axs[1, 0].set_title("Output 2")
axs[1, 0].axis("off")

axs[1, 1].imshow(output_3_np, cmap="viridis")
axs[1, 1].set_title("Output 3")
axs[1, 1].axis("off")

axs[1, 2].imshow(max_pooled_output_np, cmap="viridis")
axs[1, 2].set_title("Max-Pooled Output")
axs[1, 2].axis("off")

plt.savefig(os.path.join(args.output_folder, "output_comparison.png"))

output_filenames = ["output_0.png", "output_1.png", "output_2.png", "output_3.png", "max_pooled_output.png"]
outputs_np = [output_0_np, output_1_np, output_2_np, output_3_np, max_pooled_output_np]

for filename, output_np in zip(output_filenames, outputs_np):
    plt.imsave(os.path.join(args.output_folder, filename), output_np, cmap="viridis")

plt.show()
