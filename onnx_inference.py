import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the ONNX model
session = ort.InferenceSession("models/0.onnx")

# Capture an image from the webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Failed to capture image")

# Resize and convert the frame to RGB
frame = cv2.resize(frame, (640, 480))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Normalize and preprocess the image
frame = frame.astype(np.float32) / 255.0  # Convert to float and normalize to [0, 1]
frame = np.transpose(frame, (2, 0, 1))    # Transpose from HWC to CHW
frame = np.expand_dims(frame, axis=0)     # Add batch dimension

# Define input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
results = session.run([output_name], {input_name: frame})

# Process the results
output = results[0][0]  # Remove batch dimension
output = np.transpose(output, (1, 2, 0))  # Convert CHW to HWC if needed

# Display the image
plt.imshow(output)
plt.title('Model Output')
plt.axis('off')
plt.show()