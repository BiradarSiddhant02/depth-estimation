import onnxruntime as ort
import numpy as np
import cv2
import argparse
import os
from time import time
import matplotlib.pyplot as plt

# Load the ONNX model
try:
    session = ort.InferenceSession("models/0.onnx")
    print("Session created")
except Exception as e:
    raise RuntimeError(f"Failed to create ONNX Runtime session: {e}")

# Set up argument parser
parser = argparse.ArgumentParser(description="Depth Model Inference")
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to the input image file",
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path to the folder where outputs will be saved",
)
parser.add_argument(
    "--model".capitalize,
    type=str,
    required=True,
    help="Path to model"
)

args = parser.parse_args()
print(f"Input = {args.input}")
print(f"Output = {args.output}")

# Create output directory if it does not exist
if not os.path.exists(args.output):
    os.makedirs(args.output)


def preprocess_image(image_path):
    # Read and preprocess the image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Image not found or unable to read: {image_path}")
    print("Image read...")

    # Resize, convert, normalize, and reshape the image
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("Image resized...")

    frame = frame.astype(np.float32) / 255.0  # Convert to float and normalize to [0, 1]
    frame = np.transpose(frame, (2, 0, 1))  # Transpose from HWC to CHW
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    print("Image reshaped...")

    return frame


def save_output(output, output_folder):
    output_path = os.path.join(output_folder, "output.png")
    output = np.squeeze(output)  # Remove batch dimension
    output = (output * 255).astype(np.uint8)  # Convert back to uint8 for saving
    plt.imshow(output, cmap="viridis")
    plt.axis("off")  # Hide axes for cleaner image
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Output saved to {output_path}")


# Handle image file input
frame = preprocess_image(args.input)

# Run inference
try:
    print("Starting inference")
    start = time()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    results = session.run([output_name], {input_name: frame})
    output = results[0]
    print("Inference done")
    end = time()
    print(f"Inference time: {end - start} seconds")
except Exception as e:
    raise RuntimeError(f"Inference failed: {e}")

# Save the output
save_output(output, args.output)
