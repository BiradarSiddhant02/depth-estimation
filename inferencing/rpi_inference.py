import onnxruntime as ort
import numpy as np
import cv2
import argparse
import os

# Load the ONNX model
session = ort.InferenceSession("models/0.onnx")

# Set up argument parser
parser = argparse.ArgumentParser(description="Depth Model Inference")
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help='Path to the input image file',
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path to the folder where outputs will be saved",
)
args = parser.parse_args()

# Create output directory if it does not exist
if not os.path.exists(args.output):
    os.makedirs(args.output)

def preprocess_image(image_path):
    # Read and preprocess the image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Image not found or unable to read: {image_path}")

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Normalize and preprocess the image
    frame = frame.astype(np.float32) / 255.0  # Convert to float and normalize to [0, 1]
    frame = np.transpose(frame, (2, 0, 1))    # Transpose from HWC to CHW
    frame = np.expand_dims(frame, axis=0)     # Add batch dimension

    return frame

def save_output(output, output_folder):
    output_path = os.path.join(output_folder, 'output.png')
    output = np.transpose(output, (1, 2, 0))  # Convert CHW to HWC if needed
    output = (output * 255).astype(np.uint8)  # Convert back to uint8 for saving
    cv2.imwrite(output_path, output)
    print(f"Output saved to {output_path}")

# Handle image file input
frame = preprocess_image(args.input)

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
results = session.run([output_name], {input_name: frame})
output = results[0][0]

# Save the output
save_output(output, args.output)
