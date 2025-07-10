import os
import sys
from simon.simple_model.unet import SimpleUNet
print(f"Python version: {sys.version}", flush=True)
from data_loader import load_all_images
import numpy as np
import torch
from utils import *
from performance_analysis import calculate_and_log_performance
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

print_available_memory()

STORAGE_PATH = "/scratch/tmp/tstraus2/"
CREATOR_NAME = "arne"
DATA_PATH = "/scratch/tmp/tstraus2/Brick_Data_Train/"

if __name__ == "__main__":
    print_available_memory()
    print("Loading data...", flush=True)
    all_images, all_image_paths, all_labels, all_labels_paths = load_all_images(DATA_PATH, load_into_ram=True, as_numpy=True)
    
    print(type(all_images))
    print(all_images.shape)


    # Set all NaN values to 0 in all_images
    all_images = np.nan_to_num(all_images, nan=0.0)
    # Find pixels where all input channels are 0
    input_zero_mask = np.all(all_images == 0, axis=3)  # shape: (num_images, 256, 256)

    # Find pixels where label > 0
    label_positive_mask = all_labels > 0  # shape: (num_images, 256, 256)

    # Find pixels where all input channels are 0 and label > 0
    problematic_pixels = np.logical_and(input_zero_mask, label_positive_mask)

    # Count total number of such pixels
    num_problematic_pixels = np.sum(problematic_pixels)

    print(f"Number of pixels with all-zero input channels and label > 0: {num_problematic_pixels}", flush=True)

    # Optionally, print indices of such pixels for inspection
    if num_problematic_pixels > 0:

        indices = np.argwhere(problematic_pixels)
        unique_image_indices = np.unique(indices[:, 0])
        print(f"Unique image indices with problematic pixels: {unique_image_indices}", flush=True)
        print(f"Indices of problematic pixels (image_idx, row, col): {indices}", flush=True)






