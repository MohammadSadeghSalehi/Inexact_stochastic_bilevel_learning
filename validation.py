from priors import ICNNPrior, WCRR
import torch
from deepinv.physics import Denoising, GaussianNoise, Inpainting, Blur
from deepinv.physics.generator import MotionBlurGenerator
from evaluation import reconstruct_nmAPG

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
from deepinv.loss.metric import PSNR
from torchvision.transforms import Compose, FiveCrop, Grayscale, ToTensor, Resize
from skimage import data, exposure
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from torchvision.transforms import (
    RandomCrop,
    RandomVerticalFlip,
    Compose,
    RandomHorizontalFlip,
    CenterCrop,
    RandomApply,
    RandomRotation,
)

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "cpu"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "Denoising"


val_dataset = get_dataset("BSDS500", test=True
                          , transform=Compose([#Resize((256, 256)),
                              CenterCrop((256, 256)),
                                                                   Grayscale(num_output_channels=1,),
                                                                  ]))
# splitting in training and validation set
test_len = int(1)
train_len = 128
val_set = torch.utils.data.Subset(val_dataset,range(180, 180+ test_len))

print(f"test size: {test_len}, train size: {train_len}")
# create dataloaders
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=test_len, shuffle=False, drop_last=True, num_workers=8
)

# problem dependent parameters
noise_level = 5/255  # noise level for the Gaussian noise model
physics = Inpainting(mask=0.6, tensor_size= (1, 256, 256), noise_model=GaussianNoise(sigma=noise_level), device=device).to(device)
physics.A_dagger = physics.A_adjoint
data_fidelity = L2(sigma=1.0)
lmbd = 1.0
regularizer = WCRR(
                    sigma=noise_level,
                    weak_convexity=0.0,    
                ).to(device)
LOGS_DIR = "ckp/"
psnr = PSNR()

for file in os.listdir(LOGS_DIR):
    weight_dict_list = torch.load(os.path.join(LOGS_DIR, file), map_location=device)["weights"][-5:]
    print(f"Processing {file} with {len(weight_dict_list)} weights")
    visual_val = []
    # plot all the images in the validation set
    for i, batch in enumerate(val_dataloader):
        for j in range(batch.shape[0]):
            print(f"Image {i * batch.shape[0] + j + 1}/{len(val_dataloader) * batch.shape[0]}")
            plt.imshow(physics(batch.to(device))[j, 0].cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.show()
            visual_val.append(exposure.equalize_adapthist(batch[j, 0].cpu().numpy(), clip_limit=0.01))
            visual_val.append(exposure.equalize_adapthist(physics(batch.to(device))[j, 0].cpu().numpy().clip(0, 1), clip_limit=0.01))
    for weights in weight_dict_list:
        regularizer.load_state_dict(weights)
        # reconstructing the image
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                batch =  torch.from_numpy(exposure.equalize_adapthist(batch.cpu().numpy(), clip_limit=0.01)).to(device)
                physics.A_dagger = physics.A_adjoint
                x0 = physics.A_dagger(batch)
                x_hat,_ = reconstruct_nmAPG(
                        batch.to(device), physics, data_fidelity, regularizer, lmbd,
                        1000, 1000, 1e-4,
                        verbose=True,
                    )
                visual_val.append(x_hat[0, 0].cpu().numpy())
                print(f"PSNR: {psnr(x_hat,batch)}")
                # save the image
                plt.imshow(x_hat[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                plt.show()
    # save the images
    print(f"Saving {file[2:-2]}.pt")
    torch.save(visual_val, f"{file[:-2]}.pt")
