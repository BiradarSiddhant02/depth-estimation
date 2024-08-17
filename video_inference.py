import torch
import cv2
import numpy as np
from model import Model

depth_model_0 = Model().to(DEVICE)
depth_model_0.load_state_dict(torch.load("models/0.pth"))

depth_model_1 = Model().to(DEVICE)
depth_model_1.load_state_dict(torch.load("models/1.pth"))

depth_model_2 = Model().to(DEVICE)
depth_model_2.load_state_dict(torch.load("models/2.pth"))

depth_model_3 = Model().to(DEVICE)
depth_model_3.load_state_dict(torch.load("models/3.pth"))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Failed to open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    output_0_resized = cv2.resize(output_0_np, (320, 240))
    output_1_resized = cv2.resize(output_1_np, (320, 240))
    output_2_resized = cv2.resize(output_2_np, (320, 240))
    output_3_resized = cv2.resize(output_3_np, (320, 240))

    output_0_bgr = cv2.cvtColor(output_0_resized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    output_1_bgr = cv2.cvtColor(output_1_resized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    output_2_bgr = cv2.cvtColor(output_2_resized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    output_3_bgr = cv2.cvtColor(output_3_resized.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    combined_output = np.hstack((resized_frame, output_0_bgr, output_1_bgr, output_2_bgr, output_3_bgr))

    cv2.imshow('Video Inference', combined_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
