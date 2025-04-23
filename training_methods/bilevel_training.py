import torch
import numpy as np
from tqdm import tqdm
from deepinv.loss.metric import PSNR
# from deepinv.optim.utils import minres
from evaluation import reconstruct_nmAPG
import copy
import os
import sys
import json
import argparse
from minres import *
import random

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

logs_dict = {
    "loss_batch": [],
    "loss_epoch": [],
    "psnr_batch": [],
    "psnr_epoch": [],
    "cost_batch": [],
    "cost_epoch": [],
    "cg_batch": [],
    "cg_epoch": [],
    'lower_batch' : [],
    "lower_epoch": [],
    "p": config["optimizer"]['lower_tol_decay_p'],
    "q": config["optimizer"]['lr_decay_q'],
    "step": config["optimizer"]['alpha'],
    "eps" : config["optimizer"]['eps0'],
    "psnr_test": [],
    "loss_test": [],   
}

budget = config["training"]["budget"]
def total_iter(batch, epoch, n_batches):
    return batch + epoch * n_batches
# Manual Schedules for accuracy and step size
def _update_lr_tol(optimizer, current_total_iter, opt_config):
    """Updates learning rate and tolerance based on decay settings."""
    setting = opt_config["optimizer"]['decay_setting']
    alpha0 = opt_config["optimizer"]['alpha']
    eps0 = opt_config["optimizer"]['eps0']
    cg_tol0 = opt_config["optimizer"]['cg_tol'] # Base CG tolerance
    p = opt_config["optimizer"]['lower_tol_decay_p']
    q = opt_config["optimizer"]['lr_decay_q']
    k = float(current_total_iter) # Use float for calculations

    new_lr = alpha0
    new_lower_tol = eps0
    new_cg_tol = cg_tol0


    log_k = torch.log(torch.tensor(k)).item() + 1e-9 # Add epsilon for log(1)=0 case


    if setting == "poly-poly":
        new_lr = alpha0 * (k ** (-q))
        new_lower_tol = eps0 * (k ** (-p))
        new_cg_tol = cg_tol0 * (k ** (-p))
    elif setting == "poly-log":
        new_lr = alpha0 * (k ** (-q))
        if k > 1: new_lower_tol = eps0 * (log_k ** (-p))
        if k > 1: new_cg_tol = cg_tol0 * (log_k ** (-p))
    elif setting == "log-log":
            if k > 1: new_lr = alpha0 * (log_k ** (-q))
            if k > 1: new_lower_tol = eps0 * (log_k ** (-p))
            if k > 1: new_cg_tol = cg_tol0 * (log_k ** (-p))
    elif setting == "constant":
        pass # Keep initial values
    else:
        print(f"Warning: Unknown decay setting '{setting}'. Using constant values.")

    # Assumes optimizer structure allows direct LR update or via param_groups
    if hasattr(optimizer, 'param_groups'): # Standard PyTorch optimizers
            optimizer.param_groups[0]['lr'] = new_lr
    elif hasattr(optimizer, 'lr'): # Custom optimizers like MAID or Bilevel wrapper might have direct lr attribute
            optimizer.lr = new_lr
            # If Bilevel_Optimizer wraps a standard one, update that too
            if hasattr(optimizer, 'optimizer') and hasattr(optimizer.optimizer, 'param_groups'):
                optimizer.optimizer.param_groups[0]['lr'] = new_lr


    lower_tol = max(new_lower_tol, 1e-9) # Avoid zero tolerance
    cg_tol = max(new_cg_tol, 1e-9) # Avoid zero tolerance

    return new_lr, lower_tol, cg_tol, optimizer
def bilevel_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    epochs=100,
    NAG_step_size=1e-3,
    NAG_max_iter=300,
    NAG_tol_train=1e-4,
    NAG_tol_val=1e-4,
    minres_max_iter=1000,
    minres_tol=1e-6,
    lr=0.005,
    lr_decay=0.99,
    device="cuda" if torch.cuda.is_available() else "mps",
    verbose=False,
    upper_loss=lambda x, y: torch.sum(((x - y) ** 2).view(x.shape[0], -1), -1),
):

    def hessian_vector_product(x, v, data_fidelity, y, regularizer, lmbd, physics):
        x = x.requires_grad_(True)
        grad = data_fidelity.grad(x, y, physics) + lmbd * regularizer.grad(x)
        dot = torch.dot(grad.view(-1), v.view(-1))
        hvp = torch.autograd.grad(dot, x, create_graph=False)[0].detach()
        return hvp.detach()

    def jac_vector_product(x, v, data_fidelity, y, regularizer, lmbd, physics):
        grad_lower_level = lambda x: data_fidelity.grad(x, y, physics) + lmbd * regularizer.grad(x)
        for param in regularizer.parameters():
            dot = torch.dot(grad_lower_level(x).view(-1), v.view(-1))
            param.grad = -torch.autograd.grad(dot, param, create_graph=False)[0].detach()
        return regularizer
    lr = config["optimizer"]['alpha']
    optimizer = torch.optim.SGD(regularizer.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    psnr = PSNR()

    loss_train = []
    loss_val = []
    psnr_train = []
    psnr_val = []
    best_val_psnr = -float("inf")
    best_regularizer_state = copy.deepcopy(regularizer.state_dict())
    total_cg = 0
    total_lower = 0
    # initalise logs with noise
    logs_dict["cg_epoch"].append(0)
    logs_dict["lower_epoch"].append(0)
    logs_dict["cost_epoch"].append(0)
    loss_init = 0
    psnr_init = 0
    
    loss_test_init = 0
    psnr_test_init = 0
    for gt in train_dataloader:
        noisy = physics(gt)
        loss_fn = lambda x_in: upper_loss(gt, x_in).mean()
        loss_init += loss_fn(noisy).item()
        psnr_init += psnr(noisy, gt).mean().item()
    logs_dict["loss_epoch"].append(loss_init/len(train_dataloader))
    logs_dict["psnr_epoch"].append(psnr_init/len(train_dataloader))
    
    for gt_test in val_dataloader:
        noisy = physics(gt_test)
        loss_fn = lambda x_in: upper_loss(gt_test, x_in).mean()
        loss_test_init += loss_fn(noisy).item()
        psnr_test_init += psnr(noisy, gt_test).mean().item()
        
    logs_dict["loss_test"].append(loss_test_init / len(val_dataloader))
    logs_dict["psnr_test"].append(psnr_test_init / len(val_dataloader))
    budget_flag = False
    for epoch in range(1, epochs):
        loss_epoch = []
        psnr_epoch = []
        # ---- Training ----
        regularizer.train()
        train_loss_epoch = 0
        train_psnr_epoch = 0
        batch_idx = 0         
        for x in tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs} - Train"):
            batch_idx +=1
            iterations = total_iter(batch_idx, epoch-1, len(train_dataloader))
            lr, NAG_tol_train, minres_tol, optimizer = _update_lr_tol(optimizer,iterations, config)
            x = x.to(device).to(torch.float32)
            y = physics(x)
            x_init = y

            x_recon, iterations_lower = reconstruct_nmAPG(
                y, physics, data_fidelity, regularizer, lmbd,
                NAG_step_size, NAG_max_iter, NAG_tol_train,
                verbose=verbose, x_init=x_init,
            )
            total_lower += iterations_lower
            logs_dict["lower_batch"].append(total_lower)
            optimizer.zero_grad()
            loss_fn = lambda x_in: upper_loss(x, x_in).mean()
            train_loss_epoch += loss_fn(x_recon).item()
            train_psnr_epoch += psnr(x_recon, x).mean().item()
            
            loss_epoch.append(loss_fn(x_recon).item())
            psnr_epoch.append(psnr(x_recon, x).mean().item())
            logs_dict["loss_batch"].append(loss_epoch[-1])
            logs_dict["psnr_batch"].append(psnr_epoch[-1])
            x_recon = x_recon.requires_grad_(True)
            grad_loss = torch.autograd.grad(loss_fn(x_recon), x_recon, create_graph=False)[0].detach()

            q, cg_iter = minres(
                lambda input: hessian_vector_product(x_recon.detach(), input, data_fidelity, y, regularizer, lmbd, physics),
                grad_loss, max_iter=minres_max_iter, tol=minres_tol,
            )
            total_cg += cg_iter
            logs_dict["cg_batch"].append(total_cg)
            logs_dict["cost_batch"].append(total_cg+total_lower)
            regularizer = jac_vector_product(x_recon, q, data_fidelity, y, regularizer, lmbd, physics)

            for param, reg in zip(optimizer.param_groups[0]["params"], regularizer.parameters()):
                if param.grad is not None:
                    param.grad = reg.grad

            optimizer.step()
            
            
        logs_dict["cg_epoch"].append(total_cg)
        logs_dict["lower_epoch"].append(total_lower)
        logs_dict["cost_epoch"].append(logs_dict["cost_batch"][-1])
       
        # scheduler.step()
        mean_train_loss = train_loss_epoch / len(train_dataloader)
        mean_train_psnr = train_psnr_epoch / len(train_dataloader)
        loss_train.append(mean_train_loss)
        psnr_train.append(mean_train_psnr)
        logs_dict["loss_epoch"].append(loss_train[-1])
        logs_dict["psnr_epoch"].append(psnr_train[-1])
        print(f"[Epoch {epoch+1}] Train Loss: {mean_train_loss:.2E}, PSNR: {mean_train_psnr:.2f}")

        # ---- Validation ----
        regularizer.eval()
        with torch.no_grad():
            val_loss_epoch = 0
            val_psnr_epoch = 0
            for x_val in tqdm(val_dataloader, desc=f"Epoch {epoch}/{epochs} - Val"):
                x_val = x_val.to(device).to(torch.float32)
                y_val = physics(x_val)

                x_recon_val, _ = reconstruct_nmAPG(
                    y_val, physics, data_fidelity, regularizer, lmbd,
                    NAG_step_size, NAG_max_iter, NAG_tol_val,
                    verbose=verbose, x_init=y_val,
                )

                val_loss_epoch += upper_loss(x_val, x_recon_val).mean().item()
                val_psnr_epoch += psnr(x_recon_val, x_val).mean().item()

            mean_val_loss = val_loss_epoch / len(val_dataloader)
            mean_val_psnr = val_psnr_epoch / len(val_dataloader)
            loss_val.append(mean_val_loss)
            psnr_val.append(mean_val_psnr)
            logs_dict["loss_test"].append(loss_val[-1])
            logs_dict["psnr_test"].append(psnr_val[-1])

            print(f"[Epoch {epoch+1}] Val Loss: {mean_val_loss:.2E}, PSNR: {mean_val_psnr:.2f}")

            # ---- Save best regularizer based on validation PSNR ----
            if mean_val_psnr > best_val_psnr:
                best_val_psnr = mean_val_psnr
                best_regularizer_state = copy.deepcopy(regularizer.state_dict())
        p = config["optimizer"]['lower_tol_decay_p']
        q = config["optimizer"]['lr_decay_q']
        eps = config["optimizer"]['eps0']
        alpha = config["optimizer"]['alpha']
        torch.save(logs_dict, os.path.join(directory, f"logs_{alpha}_{eps}_{p}_{q}_.pt"))
        if logs_dict["cost_epoch"][-1] >= budget:
                budget_flag = True
                break
        if budget_flag:
            print("Budget exceeded")
            break
    # Load best regularizer
    regularizer.load_state_dict(best_regularizer_state)

    return regularizer, loss_train, loss_val, psnr_train, psnr_val

