import os
import torch
import matplotlib.pyplot as plt
from itertools import combinations
import pickle
import numpy as np

# Directories
LOGS_DIR = "vislogs"
PLOTS_DIR = "visplots"

# Function to compute Peak Signal-to-Noise Ratio (PSNR)
def compute_psnr(original, reconstructed):
    mse = np.mean((np.array(original) - np.array(reconstructed)) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match
    max_pixel = 1.0  # Assuming normalized pixel values in [0, 1]
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# Helper function to check if a value is plottable (i.e., list of numbers)
def is_plottable(value):
    return isinstance(value, list) and all(isinstance(v, (int, float)) for v in value)
def plot(image, title, filename):
    plt.figure(figsize=(10, 6))
    plt.imshow(image[0], cmap= "gray", vmin = 0, vmax = 1)
    plt.title(title)
    # Clean filename
    plot_filename = f"{filename[:-3]}.png"
    plot_path = os.path.join(PLOTS_DIR, plot_filename[:-3])
    plt.savefig(plot_path)
    plt.close()

train_idx = 3
val_idx = 5
# Loop through each .pt log file
for filename in os.listdir(LOGS_DIR):
    if filename.endswith(".pt"):
        log_path = os.path.join(LOGS_DIR, filename)
        log_data = torch.load(log_path, map_location='cpu', weights_only = False, pickle_module=pickle)
        # Extract plottable entries
        numeric_entries = {k: v for k, v in log_data.items() if is_plottable(v) and len(v) > 1}
        
        image_train = log_data["train"]
        image_val = log_data["val"]
        checkpoint_cost = log_data["cost"]
        original_train = image_train[0]
        original_val = image_val[0]
        print(checkpoint_cost)
        for idx, (img_train, img_val) in enumerate(zip(image_train[2:], image_val[2:])):
            psnr_train = compute_psnr(original_train, img_train)
            psnr_val = compute_psnr(original_val, img_val)
            plot(img_train[train_idx], f"PSNR: {psnr_train.mean()}, cost {checkpoint_cost[idx]}", f"train{checkpoint_cost[idx]}"+filename)
            plot(img_val[val_idx], f"PSNR: {psnr_val}, cost {checkpoint_cost[idx]}", f"val{checkpoint_cost[idx]}"+filename)

print(f"Plots saved to '{PLOTS_DIR}/'")