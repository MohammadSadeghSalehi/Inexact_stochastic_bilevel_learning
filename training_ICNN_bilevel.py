#%%
from priors import ICNNPrior, WCRR
import torch
from deepinv.physics import Denoising, GaussianNoise, Inpainting, Blur
from deepinv.physics.generator import MotionBlurGenerator

import deepinv
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
import torch.nn as nn
from torchvision.transforms import Grayscale
import matplotlib.pyplot as plt

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



transform = Compose(
    [
        # Grayscale(num_output_channels=1,),
        CenterCrop((64,64)),
        # RandomHorizontalFlip(p=0.5),
        # RandomVerticalFlip(p=0.5),
        # RandomApply([RandomRotation((90, 90))], p=0.5),
    ]
)
train_dataset = get_dataset("BSDS500", test=False, transform=transform)
val_dataset = get_dataset("BSDS500", test=True, transform=Compose([CenterCrop((192,192)),
                                                                #    Grayscale(num_output_channels=1,),
                                                                  ]))
# splitting in training and validation set
test_len = int(8)
train_len = 128
train_set = torch.utils.data.Subset(train_dataset, range(train_len))
val_set = torch.utils.data.Subset(val_dataset, range(10, 10 + test_len))

print(f"test size: {test_len}, train size: {train_len}")
# create dataloaders
sampler = torch.utils.data.RandomSampler(train_set, replacement=True)
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=16, sampler=sampler, drop_last=True, num_workers=8
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=8, shuffle=False, drop_last=True, num_workers=8
)

# problem dependent parameters
noise_level = 25/255  # noise level for the Gaussian noise model
psf_size = 5
# physics = Inpainting(mask=0.7, tensor_size= (1, 96, 96), noise_model=GaussianNoise(sigma=noise_level), device=device).to(device)
# physics = deepinv.physics.Tomography(
#     angles=180,
#     img_width=64,
#     circle=False,
#     device=device,
#     # parallel_beam=True,
#     noise_model=GaussianNoise(sigma=noise_level),
# )
filter_torch = deepinv.physics.blur.gaussian_blur(sigma=(1, 1))
n_channels = 3  # 3 for color images, 1 for gray-scale images

def gaussian_kernel(size: int, mean: float, std: float):
    """Creates a 2D Gaussian Kernel for convolution."""
    d = torch.arange(size).float() - mean
    gaussian_1d = torch.exp(-(d ** 2) / (2 * std ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    
    # Create a 2D Gaussian kernel by computing the outer product of two 1D kernels
    gaussian_2d = torch.outer(gaussian_1d, gaussian_1d)
    
    # Normalize the 2D kernel to ensure the sum equals 1
    gaussian_2d = gaussian_2d / gaussian_2d.sum()
    
    return gaussian_2d

# Create a Gaussian filter with the specified size and standard deviation
filter_size = psf_size
filter_std = 1.0  # Standard deviation for the Gaussian filter
filter_2d = gaussian_kernel(filter_size, mean=filter_size // 2, std=filter_std)
kernel_4d = filter_2d.unsqueeze(0).unsqueeze(0)
kernel_4d = kernel_4d.repeat(1, 3, 1, 1)
print("Kernel shape:", kernel_4d.shape)
# The BlurFFT instance from physics enables to compute efficently backward operators with Fourier transform.

# class Blur(nn.Module):
#     def __init__(self, filter, padding=2, device="cpu", in_channels=3):
#         super(Blur, self).__init__()
#         self.filter = filter.to(device)
#         self.padding = padding
#         self.device = device
#         self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=self.padding, bias=False, padding_mode="zeros",groups= 3)
#         self.convT = nn.ConvTranspose2d(
#             in_channels, in_channels, kernel_size=5, padding=self.padding, bias=False, padding_mode="zeros", groups= 3
#         )
#         # freeze the weights of the convolutional layer
#         self.conv.requires_grad_(False)
#         self.conv.weight.data = self.filter
#         self.conv.to(device)
#         self.convT.weight.data = self.filter
#         self.convT.to(device)
#     def forward(self, x):
#         return self.conv(x)
#     def A(self, x):
#         self.forward(x)
#     def A_adjoint(self, x):
#         return self.convT(x)
    
# physics = deepinv.physics.Blur(
#     filter=deepinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=90.0),
#     padding= "replicate",
#     device=device,
#     noise_model=deepinv.physics.GaussianNoise(sigma=noise_level),
# )
# override the A_dagger method of physics
# physics.A_dagger = physics.A_adjoint
# deepinv.physics.Blur(
#     filter=kernel_4d,
#     padding= "replicate",
#     device=device,
#     noise_model=deepinv.physics.GaussianNoise(sigma=noise_level),
# )

physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))
#Denoising(noise_model=GaussianNoise(sigma=noise_level))
data_fidelity = L2(sigma=1.0)
lmbd = 1.0

# define regularizer
# regularizer = ICNNPrior(in_channels=3,channels=20,device=device)
regularizer = WCRR(
    sigma=noise_level,
    weak_convexity=0.0,    
).to(device)
print("numebr of parameters in regularizer: ", sum(p.numel() for p in regularizer.parameters()), "\n")

# alpha_values = [ 1e-2, 1e-4]
# eps_values = [ 1e-1, 1e-3]
# p_values = [  0.25 , 1.0, 0.5, 0.0]
# q_values = [0.0, 1.0, 0.5, 0.25] 

setting_pairs = [

    
    # (1e-2, 1e-0, 0, 2),
    # (1e-3, 1e-0, 0, 2),

    # (1e-2, 1e-0, 0.5, 2),
    # (1e-2, 1e-0, 0, 0.5),
    # (1e-3, 1e-0, 0, 0.5),
    # (1e-3, 1e-0, 0.5, 0.5),
    # (1e-2, 1e-0, 0.5, 0.5),

    (1e-3, 1e-4, 0.5, 0.0),
    # (1e-2, 1e-4, 0.0, 0.5),
    # (1e-3, 1e-4, 0.5, 0.5),
    # (1e-2, 1e-4, 0.5, 0.5),
    # (1e-2, 1e-2, 0.5, 0.25),
    # (1e-2, 1e-2, 0.5, 0.5),

    # (1e-2, 1e-4, 0, 0),
    # (1e-2, 1e-4, 0, 0.25),
    # (1e-2, 1e-4, 0.5, 0),

    # (1e-3, 1e0, 0.5, 0.5),
    # (1e-3, 1e-4, 0.5, 0),
]

# for alpha in alpha_values:
#     for eps in eps_values:
#         for p in p_values:       
#             for q in q_values:
                # if (alpha == 1e-1) or (eps == 1e-0 and (p ==0.0 or p == 0.25)):
                #     print("Skipping combination")
                #     continue
for setting in setting_pairs:
                alpha = setting[0]
                eps = setting[1]
                q = setting[2]
                p = setting[3]
                
                regularizer = WCRR(
                    sigma=noise_level,
                    weak_convexity=0.0,    
                ).to(device)
                # regularizer = ICNNPrior(in_channels=3,channels=30,device=device)

                print(f"Training with alpha={alpha}, eps={eps}, p={p}, q={q}")
                config["optimizer"]["lower_tol_decay_p"] = p
                config["optimizer"]["lr_decay_q"] = q
                config["optimizer"]["alpha"] = alpha
                config["optimizer"]["eps0"] = eps
                config["optimizer"]["cg_tol"] = eps
                with open("config.json", "w") as f:
                    json.dump(config, f, indent=4)
                print(f"config lr decay: {config['optimizer']["lr_decay_q"]}, lower tol decay: {config['optimizer']["lower_tol_decay_p"]}, alpha: {config['optimizer']["alpha"]}, eps0: {config['optimizer']["eps0"]}")
                regularizer_learned, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
                    regularizer,
                    physics,
                    data_fidelity,
                    lmbd,
                    train_dataloader,
                    val_dataloader,
                    epochs=2000,
                    NAG_step_size=1e-1,
                    NAG_max_iter=300,
                    NAG_tol_train= eps,
                    NAG_tol_val=1e-4,
                    minres_tol = eps,
                    lr= alpha,
                    lr_decay=0.99,
                    device=device,
                    verbose=True,
                    p_val = p,
                    q_val = q,
)

torch.save(regularizer.state_dict(), f"weights/ICNN_bilevel.pt")