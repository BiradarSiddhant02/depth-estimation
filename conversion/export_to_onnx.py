from torch import randn, load, onnx
from model import Model
import argparse

parser = argparse.ArgumentParser(description="Pytorch to ONNX converter")
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Path to Pytorch Model",
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

# Export the model to ONNX format without any optimizations or changes
onnx.export(
    model,
    dummy_input,
    "models/0.onnx",
    export_params=True,  # Export the model's parameters
    opset_version=11,  # Specify the ONNX version
    do_constant_folding=False,  # Disable constant folding to ensure raw weights are preserved
    input_names=["inputs"],  # Specify the input variable name
    output_names=["outputs"],  # Specify the output variable name
    keep_initializers_as_inputs=True,  # Keep initializers (weights) as inputs, maintaining raw structure
    verbose=False,  # Reduce verbosity to avoid logging unnecessary details
)

print("Model has been exported to ONNX with the raw weights and structure.")
