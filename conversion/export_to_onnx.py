import torch
from model import Model
import onnx

# Load the model and its weights
model = Model()
model.load_state_dict(torch.load("models/0.pth"))
model.eval()

# Prepare dummy input matching the model's expected input dimensions
dummy_input = torch.randn(1, 3, 480, 640)

# Export the model to ONNX format with optimizations for RPi
torch.onnx.export(
    model,
    dummy_input,
    "models/0.onnx",
    export_params=True,      # Export the model's parameters
    opset_version=11,        # Specify the ONNX version
    do_constant_folding=True, # Optimize constant expressions
    input_names=['inputs'],  # Specify the input variable name
    output_names=['outputs'], # Specify the output variable name
    dynamic_axes={'inputs': {0: 'batch_size'}, 'outputs': {0: 'batch_size'}} # Handle variable batch sizes
)

print("Model has been converted to ONNX and saved as 'models/0.onnx'")
