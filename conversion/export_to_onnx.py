from torch import randn, load, onnx
from model import Model
import argparse

# Argument parser to handle command-line arguments
parser = argparse.ArgumentParser(description="Pytorch to ONNX converter")
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Path to Pytorch Model",
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path to save the ONNX model",
)
args = parser.parse_args()

# Load the model and its weights
model = Model()
try:
    model.load_state_dict(load(args.model))
except FileNotFoundError:
    print(f"Model file not found at {args.model}. Please check the path.")
except RuntimeError as e:
    print(f"Error loading model state dict: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
else:
    print("Model loaded successfully.")
model.eval()

# Prepare dummy input matching the model's expected input dimensions
dummy_input = randn(1, 3, 480, 640)

# Export the model to ONNX format with the specified save location
onnx.export(
    model,
    dummy_input,
    args.output,  # Save location is now flexible
    export_params=True,
    opset_version=11
)

print(f"Model has been exported to ONNX at {args.output} with the raw weights and structure.")
