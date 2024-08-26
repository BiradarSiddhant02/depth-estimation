import torch
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from model import Model
import argparse
import os
from time import time
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from dataloader import (
    DepthDataset,
    Augmentation,
    ToTensor,
)

# Argument parser
parser = argparse.ArgumentParser(description="Depth Model Inference")
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Path to the folder where the model is saved",
)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
depth_model_0 = Model().to(DEVICE)
depth_model_0.load_state_dict(torch.load(f"{args.model}/0.pth", map_location=DEVICE, weights_only=True))

depth_model_1 = Model().to(DEVICE)
depth_model_1.load_state_dict(torch.load(f"{args.model}/1.pth", map_location=DEVICE, weights_only=True))

depth_model_2 = Model().to(DEVICE)
depth_model_2.load_state_dict(torch.load(f"{args.model}/2.pth", map_location=DEVICE, weights_only=True))

depth_model_3 = Model().to(DEVICE)
depth_model_3.load_state_dict(torch.load(f"{args.model}/3.pth", map_location=DEVICE, weights_only=True))

print("Models loaded")

test_csv = pd.read_csv("inputs/nyu_data/data/nyu2_test.csv")
test_csv = test_csv.values.tolist()
test_csv = shuffle(test_csv, random_state=2)

eval_set = DepthDataset(
    csv=test_csv,
    root_dir="inputs/nyu_data",
    transform=transforms.Compose([Augmentation(0.5), ToTensor()]),
)

batch_size = 1
test_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=False)

print("Dataset created")

def RMS(output, truth):
    output = output.squeeze().cpu().numpy()
    truth = truth.squeeze().cpu().numpy()

    # Compute the squared difference
    squared_diff = (output - truth) ** 2

    # Calculate the mean squared error
    mse = np.mean(squared_diff)

    # Calculate the root mean squared error
    rms = np.sqrt(mse)

    return rms

# Iterate over the DataLoader
with torch.no_grad():
    for batch in test_loader:
        image = batch['image'].to(DEVICE)
        depth = batch['depth'].to(DEVICE)

        depth_n = 1000. / depth

        output_0 = depth_model_0(image)
        output_1 = depth_model_1(image)
        output_2 = depth_model_2(image)
        output_3 = depth_model_3(image)

        model_0_rms = RMS(output_0, depth_n)      
        model_1_rms = RMS(output_1, depth_n)      
        model_2_rms = RMS(output_2, depth_n)      
        model_3_rms = RMS(output_3, depth_n)

        print(model_0_rms, model_1_rms, model_2_rms, model_3_rms)
