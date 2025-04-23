#%%
from priors import ICNNPrior
import torch
from deepinv.physics import Denoising, GaussianNoise
from training_methods import bilevel_training
from deepinv.optim import L2
from dataset import get_dataset
from torchvision.transforms import RandomCrop, CenterCrop, Compose, Resize
import random
import numpy as np
import json
import argparse
import random
import sys, os

from torchvision.transforms import (
    RandomCrop,
    RandomVerticalFlip,
    Compose,
    RandomHorizontalFlip,
    CenterCrop,
    RandomApply,
    RandomRotation,
)


# set random seed
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

directory = os.getcwd()
parser = argparse.ArgumentParser(description="Bilevel Optimization Training Script")
parser.add_argument('--config', type=str, default=directory +'/config.json',
                    help='Path to the JSON configuration file (default: config.json)')
args = parser.parse_args()

# --- Load Configuration ---
try:
    with open(args.config, 'r') as f:
        config = json.load(f)
    print(f"Loaded configuration from: {args.config}")
except FileNotFoundError:
    print(f"Error: Configuration file not found at {args.config}")
    sys.exit(1)
except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.config}")
        sys.exit(1)


if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "Denoising"

# problem dependent parameters
noise_level = 25/255
physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))
data_fidelity = L2(sigma=1.0)
lmbd = 1.0

transform = Compose(
    [
        RandomCrop(64),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomApply([RandomRotation((90, 90))], p=0.5),
    ]
)
train_dataset = get_dataset("BSDS500_gray", test=False, transform=transform)
val_dataset = get_dataset("BSDS500_gray", test=False, transform=CenterCrop((256)))
# splitting in training and validation set
test_ratio = 0.05
test_len = int(len(train_dataset) * test_ratio)
train_len = len(train_dataset) - test_len
train_set = torch.utils.data.Subset(train_dataset, range(train_len))
val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(train_dataset)))

print(f"test size: {test_len}, train size: {train_len}")
# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=8, shuffle=True, drop_last=True, num_workers=0
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=4, shuffle=False, drop_last=True, num_workers=0
)

# define regularizer
regularizer = ICNNPrior(in_channels=1,channels=32,device=device)

regularizer, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    epochs=config["training"]['epochs'],
    NAG_step_size=1e-1,
    NAG_max_iter=300,
    NAG_tol_train=config["optimizer"]['eps0'],
    NAG_tol_val=1e-6,
    lr=config["optimizer"]['alpha'],
    lr_decay=0.99,
    device=device,
    verbose=True,
)

torch.save(regularizer.state_dict(), f"weights/ICNN_bilevel.pt")