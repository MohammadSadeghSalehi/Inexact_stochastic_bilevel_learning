import torch
import numpy as np
from tqdm import tqdm
from deepinv.loss.metric import PSNR
from evaluation import reconstruct_nmAPG, minres
from minres import *
import copy

def total_iter(batch, epoch, n_batches):
    return batch + epoch * n_batches
# Manual Schedules for accuracy and step size
def _update_lr_tol(optimizer, current_total_iter, alpha0, eps0, delta0):
    """Updates learning rate and tolerance based on decay settings."""
    cg_tol0 = delta0
    p = 0.0
    q = 0.0
    k = float(current_total_iter)
    
    new_lr = alpha0
    new_lower_tol = eps0
    new_cg_tol = cg_tol0
    new_lr = alpha0 * (k ** (-q))
    new_lower_tol = eps0 * (k ** (-p))
    new_cg_tol = cg_tol0 * (k ** (-p))

    if hasattr(optimizer, 'param_groups'): 
            optimizer.param_groups[0]['lr'] = new_lr
    elif hasattr(optimizer, 'lr'): 
            optimizer.lr = new_lr
            if hasattr(optimizer, 'optimizer') and hasattr(optimizer.optimizer, 'param_groups'):
                optimizer.optimizer.param_groups[0]['lr'] = new_lr

    lower_tol = max(new_lower_tol, 1e-9) # Avoid zero tolerance
    cg_tol = max(new_cg_tol, 1e-9) # Avoid zero tolerance

    return new_lr, lower_tol, cg_tol, optimizer

def bilevel_training_ISGD(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    epochs=100,
    NAG_step_size=1e-1,
    NAG_max_iter=1000,
    NAG_tol_train=1e-4,
    NAG_tol_val=1e-4,
    minres_max_iter=1000,
    minres_tol=1e-6,
    lr=0.005,
    lr_decay=0.99,
    device="cuda" if torch.cuda.is_available() else "cpu",
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

    optimizer = torch.optim.Adam(regularizer.parameters(), lr=lr)
    psnr = PSNR()

    loss_train = []
    loss_val = []
    psnr_train = []
    psnr_val = []

    logs = {"train_loss":[],
            "psnr" : [],
            "val_loss" : [],
            "val_psnr" : [],
            "eps": [],
            "lr": []
            }
    best_val_psnr = -float("inf")
    best_regularizer_state = copy.deepcopy(regularizer.state_dict())

    alpha0 = lr
    eps0 = NAG_tol_train
    delta0 = minres_tol
    for epoch in range(epochs):
        # ---- Training ----
        regularizer.train()
        train_loss_epoch = 0
        train_psnr_epoch = 0
        batch_idx = 0
        for x in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
            batch_idx+=1
            iterations = total_iter(batch_idx, epoch, len(train_dataloader))
            lr, NAG_tol_train, minres_tol, optimizer = _update_lr_tol(optimizer,iterations, alpha0, eps0, delta0)
            logs["eps"].append(NAG_tol_train)
            logs["lr"].append(optimizer.param_groups[0]["lr"])
            x = x.to(device).to(torch.float32)
            y = physics(x)
            x_init = y

            x_recon, lower_iter = reconstruct_nmAPG(
                y, physics, data_fidelity, regularizer, lmbd,
                NAG_step_size, NAG_max_iter, NAG_tol_train,
                verbose=verbose, x_init=x_init,
            )

            optimizer.zero_grad()
            loss_fn = lambda x_in: upper_loss(x, x_in).mean()
            train_loss_epoch += loss_fn(x_recon).item()
            train_psnr_epoch += psnr(x_recon, x).mean().item()

            x_recon = x_recon.requires_grad_(True)
            grad_loss = torch.autograd.grad(loss_fn(x_recon), x_recon, create_graph=False)[0].detach()

            q , _= minres(
                lambda input: hessian_vector_product(x_recon.detach(), input, data_fidelity, y, regularizer, lmbd, physics),
                grad_loss, max_iter=minres_max_iter, tol=minres_tol,
            )

            regularizer = jac_vector_product(x_recon, q, data_fidelity, y, regularizer, lmbd, physics)

            for param, reg in zip(optimizer.param_groups[0]["params"], regularizer.parameters()):
                if param.grad is not None:
                    param.grad = reg.grad

            optimizer.step()

        mean_train_loss = train_loss_epoch / len(train_dataloader)
        mean_train_psnr = train_psnr_epoch / len(train_dataloader)
        loss_train.append(mean_train_loss)
        psnr_train.append(mean_train_psnr)
        logs["train_loss"].append(mean_train_loss)
        logs["psnr"].append(mean_train_psnr)
        print(f"[Epoch {epoch+1}] Train Loss: {mean_train_loss:.2E}, PSNR: {mean_train_psnr:.2f}")

        # ---- Validation ----
        regularizer.eval()
        with torch.no_grad():
            val_loss_epoch = 0
            val_psnr_epoch = 0
            for x_val in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Val"):
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
            logs["val_loss"].append(mean_val_loss)
            logs["val_psnr"].append(mean_val_psnr)
            print(f"[Epoch {epoch+1}] Val Loss: {mean_val_loss:.2E}, PSNR: {mean_val_psnr:.2f}")

            # ---- Save best regularizer based on validation PSNR ----
            torch.save(logs, f"weights/logs_ISGD_{eps0}_{alpha0}.pt")

            if mean_val_psnr > best_val_psnr:
                best_val_psnr = mean_val_psnr
                best_regularizer_state = copy.deepcopy(regularizer.state_dict())
                torch.save(regularizer.state_dict(), f"weights/ISGD_{eps0}_{alpha0}.pt")

    # Load best regularizer
    regularizer.load_state_dict(best_regularizer_state)

    return regularizer, loss_train, loss_val, psnr_train, psnr_val

