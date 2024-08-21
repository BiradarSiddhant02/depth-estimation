import torch
from model import Model
import onnx

model = Model()
model.load_state_dict(torch.load("models/0.pth", weights_only=True))
model.eval()

dummy_input = torch.randn(1, 3, 480, 640)

torch.onnx.export(
    model,
    dummy_input,
    "models/0.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['inputs'],
    output_names=['outputs']
)
