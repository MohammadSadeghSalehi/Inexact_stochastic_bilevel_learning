import os
import torch
import numpy as np
from tqdm import tqdm
from deepinv.loss.metric import PSNR, SSIM
from evaluation import reconstruct_nmAPG, reconstruct_nmAPG2, minres
import copy

def total_iter(batch, epoch, n_batches):
    return batch + epoch * n_batches

# Manual Schedules for accuracy and step size
def _update_lr_tol(optimizer, current_total_iter, alpha0, eps0, delta0, p=0.0, q=0.0):
    """Updates learning rate and tolerance based on the polynomial decay
    schedules alpha_k = alpha0 * k^{-q} and eps_k = eps0 * k^{-p}."""
    cg_tol0 = delta0
    k = max(float(current_total_iter), 1.0)

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
    # --- schedule / cost-accounting options ---
    p=0.0,                       # accuracy decay exponent: eps_k = eps0 * k^{-p}
    q=0.0,                       # step size decay exponent: alpha_k = alpha0 * k^{-q}
    optimizer_type="adam",       # "adam" (IAdam) or "sgd" (ISGD)
    budget=None,                 # stop when total cost (lower + CG iters) exceeds this
    checkpoint_costs=None,       # list of costs at which to snapshot test reconstructions
    log_dir="logs",
    vis_dir=None,                # if set, save visual checkpoints here
    stall_patience=None,         # precision-floor detection for the lower-level solver
    stop_criterion="relative",   # "relative" | "grad_rms" (dtype-aware RMS criterion)
    lower_mu=1.0,                # strong convexity constant used by grad_rms
    persistent_warm_start=False, # warm-start inner solves from the previous
                                 # solution of the same sample; requires the
                                 # dataloader to yield (x, y, idx) triples
    run_tag="",
):
    """Inexact stochastic gradient descent (ISGD) / Adam (IAdam) for bilevel learning.

    Each upper-level step solves the lower-level problem to tolerance eps_k = eps0 k^{-p}
    (nmAPG), solves the Hessian system to tolerance delta_k = delta0 k^{-p} (MINRES), and takes
    a step of size alpha_k = alpha0 k^{-q} with the resulting inexact hypergradient. The
    computational cost is the total number of lower-level and MINRES iterations.
    """

    def hessian_vector_product(x, v, data_fidelity, y, regularizer, lmbd, physics):
        x = x.requires_grad_(True)
        grad = data_fidelity.grad(x, y, physics) + lmbd * regularizer.grad(x)
        dot = torch.dot(grad.view(-1), v.view(-1))
        hvp = torch.autograd.grad(dot, x, create_graph=False)[0].detach()
        return hvp.detach()

    def jac_vector_product(x, v, data_fidelity, y, regularizer, lmbd, physics):
        grad_lower_level = data_fidelity.grad(x, y, physics) + lmbd * regularizer.grad(x)
        dot = torch.dot(grad_lower_level.view(-1), v.view(-1))
        params = [prm for prm in regularizer.parameters() if prm.requires_grad]
        grads = torch.autograd.grad(dot, params, create_graph=False, allow_unused=True)
        for prm, g in zip(params, grads):
            prm.grad = -g.detach() if g is not None else None
        return regularizer

    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(regularizer.parameters(), lr=lr)
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(regularizer.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")
    psnr = PSNR()
    ssim = SSIM()

    loss_train = []
    loss_val = []
    psnr_train = []
    psnr_val = []

    alpha0 = lr
    eps0 = NAG_tol_train
    delta0 = minres_tol

    # Log schema consumed by plot.py / plot_visual.py
    logs = {
        # per-batch
        "cost_batch": [],    # cumulative cost after each batch
        "lower_batch": [],   # lower-level iterations in this batch
        "cg_batch": [],      # MINRES/CG iterations in this batch
        "loss_batch": [],
        # per-epoch
        "cost_epoch": [],    # cumulative cost at the end of each epoch
        "loss_epoch": [],
        "psnr_epoch": [],
        "loss_test": [],
        "psnr_test": [],
        "ssim_test": [],
        # schedules actually applied
        "eps_k": [],
        "lr_k": [],
        # run metadata (scalars)
        "step": alpha0,
        "eps": eps0,
        "p": p,
        "q": q,
        "optimizer": optimizer_type,
        "budget": budget,
        # legacy keys
        "train_loss": [],
        "psnr": [],
        "val_loss": [],
        "val_psnr": [],
    }
    vislogs = {
        "checkpoint_cost": [],
        "img_val": [],
        "psnr_val_ckp": [],
    }
    pending_checkpoints = sorted(checkpoint_costs) if checkpoint_costs else []

    best_val_psnr = -float("inf")
    best_regularizer_state = copy.deepcopy(regularizer.state_dict())
    warm_buf = {}

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
    tag = f"{run_tag}_" if run_tag else ""
    log_path = os.path.join(log_dir, f"logs_{tag}{alpha0}_{eps0}_{p}_{q}_.pt")

    total_cost = 0
    stop_training = False

    for epoch in range(epochs):
        # ---- Training ----
        regularizer.train()
        train_loss_epoch = 0
        train_psnr_epoch = 0
        batch_idx = 0
        for x in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Train", disable=not verbose):
            batch_idx += 1
            iterations = total_iter(batch_idx, epoch, len(train_dataloader))
            lr_k, tol_k, cg_tol_k, optimizer = _update_lr_tol(
                optimizer, iterations, alpha0, eps0, delta0, p=p, q=q
            )
            logs["eps_k"].append(tol_k)
            logs["lr_k"].append(optimizer.param_groups[0]["lr"])
            sample_idx = None
            if isinstance(x, (tuple, list)):
                # fixed-observation dataset: batches are (x, y) pairs, or
                # (x, y, idx) triples when persistent warm starting is used
                if len(x) == 3:
                    x, y, sample_idx = x
                else:
                    x, y = x
                x = x.to(device).to(torch.float32)
                y = y.to(device).to(torch.float32)
            else:
                x = x.to(device).to(torch.float32)
                y = physics(x)

            if persistent_warm_start and sample_idx is not None:
                # start the inner solver from the previous solution of the
                # SAME sample (theory: Remark on warm starts -- the drift
                # ||xhat(theta_{k+1}) - xhat(theta_k)|| = O(alpha_k) is small,
                # so O(1) inner iterations suffice on average)
                x_init = y.clone()
                for j, si in enumerate(sample_idx.tolist()):
                    if si in warm_buf:
                        x_init[j] = warm_buf[si]
            else:
                x_init = y

            x_recon, lower_iters = reconstruct_nmAPG2(
                y, physics, data_fidelity, regularizer, lmbd,
                NAG_step_size, NAG_max_iter, tol_k,
                verbose=verbose, x_init=x_init,
                return_iters=True, stall_patience=stall_patience,
                stop_criterion=stop_criterion, mu=lower_mu,
            )
            if persistent_warm_start and sample_idx is not None:
                for j, si in enumerate(sample_idx.tolist()):
                    warm_buf[si] = x_recon[j].detach().clone()

            optimizer.zero_grad()
            loss_fn = lambda x_in: upper_loss(x, x_in).mean()
            batch_loss = loss_fn(x_recon).item()
            train_loss_epoch += batch_loss
            train_psnr_epoch += psnr(x_recon, x).mean().item()

            x_recon = x_recon.requires_grad_(True)
            grad_loss = torch.autograd.grad(loss_fn(x_recon), x_recon, create_graph=False)[0].detach()
            hg, cg_iters = minres(
                lambda input: hessian_vector_product(x_recon.detach(), input, data_fidelity, y, regularizer, lmbd, physics),
                grad_loss, max_iter=minres_max_iter, tol=cg_tol_k,
            )

            regularizer = jac_vector_product(x_recon.detach(), hg, data_fidelity, y, regularizer, lmbd, physics)

            for param, reg in zip(optimizer.param_groups[0]["params"], regularizer.parameters()):
                if param.grad is not None:
                    param.grad = reg.grad

            optimizer.step()

            # ---- Cost accounting ----
            total_cost += int(lower_iters) + int(cg_iters)
            logs["lower_batch"].append(int(lower_iters))
            logs["cg_batch"].append(int(cg_iters))
            logs["cost_batch"].append(total_cost)
            logs["loss_batch"].append(batch_loss)

            if budget is not None and total_cost >= budget:
                stop_training = True
                if verbose:
                    print(f"Budget {budget} reached at cost {total_cost}; stopping.")
                break

        n_batches_done = max(batch_idx, 1)
        mean_train_loss = train_loss_epoch / n_batches_done
        mean_train_psnr = train_psnr_epoch / n_batches_done
        loss_train.append(mean_train_loss)
        psnr_train.append(mean_train_psnr)
        logs["cost_epoch"].append(total_cost)
        logs["loss_epoch"].append(mean_train_loss)
        logs["psnr_epoch"].append(mean_train_psnr)
        logs["train_loss"].append(mean_train_loss)
        logs["psnr"].append(mean_train_psnr)
        print(f"[Epoch {epoch+1}] Cost: {total_cost}, Train Loss: {mean_train_loss:.2E}, PSNR: {mean_train_psnr:.2f}")

        # ---- Validation / test ----
        regularizer.eval()
        with torch.no_grad():
            val_loss_epoch = 0
            val_psnr_epoch = 0
            val_ssim_epoch = 0
            first_val_recon = None
            first_val_psnr = None
            for j, x_val in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Val", disable=not verbose)):
                if isinstance(x_val, (tuple, list)):
                    x_val, y_val = x_val
                    x_val = x_val.to(device).to(torch.float32)
                    y_val = y_val.to(device).to(torch.float32)
                else:
                    x_val = x_val.to(device).to(torch.float32)
                    y_val = physics(x_val)

                x_recon_val = reconstruct_nmAPG2(
                    y_val, physics, data_fidelity, regularizer, lmbd,
                    NAG_step_size, NAG_max_iter, NAG_tol_val,
                    verbose=verbose, x_init=y_val,
                    stall_patience=stall_patience,
                    stop_criterion=stop_criterion, mu=lower_mu,
                )

                val_loss_epoch += upper_loss(x_val, x_recon_val).mean().item()
                batch_val_psnr = psnr(x_recon_val, x_val).mean().item()
                val_psnr_epoch += batch_val_psnr
                val_ssim_epoch += ssim(x_recon_val, x_val).mean().item()
                if j == 0:
                    first_val_recon = x_recon_val.detach().cpu()
                    first_val_psnr = batch_val_psnr

            mean_val_loss = val_loss_epoch / len(val_dataloader)
            mean_val_psnr = val_psnr_epoch / len(val_dataloader)
            mean_val_ssim = val_ssim_epoch / len(val_dataloader)
            loss_val.append(mean_val_loss)
            psnr_val.append(mean_val_psnr)
            logs["loss_test"].append(mean_val_loss)
            logs["psnr_test"].append(mean_val_psnr)
            logs["ssim_test"].append(mean_val_ssim)
            logs["val_loss"].append(mean_val_loss)
            logs["val_psnr"].append(mean_val_psnr)
            print(f"[Epoch {epoch+1}] Val Loss: {mean_val_loss:.2E}, PSNR: {mean_val_psnr:.2f}, SSIM: {mean_val_ssim:.4f}")

            # ---- Visual checkpoints at prescribed costs ----
            while pending_checkpoints and total_cost >= pending_checkpoints[0]:
                ckp = pending_checkpoints.pop(0)
                vislogs["checkpoint_cost"].append(ckp)
                vislogs["img_val"].append(first_val_recon)
                vislogs["psnr_val_ckp"].append(first_val_psnr)
                if vis_dir is not None:
                    torch.save(vislogs, os.path.join(vis_dir, f"vis_logs_{tag}{alpha0}_{eps0}_{p}_{q}_.pt"))

            # ---- Save logs and best regularizer based on validation PSNR ----
            torch.save(logs, log_path)

            if mean_val_psnr > best_val_psnr:
                best_val_psnr = mean_val_psnr
                best_regularizer_state = copy.deepcopy(regularizer.state_dict())
                torch.save(regularizer.state_dict(), os.path.join("weights", f"ISGD_{tag}{alpha0}_{eps0}_{p}_{q}.pt"))

        if stop_training:
            break

    # Load best regularizer
    regularizer.load_state_dict(best_regularizer_state)

    return regularizer, loss_train, loss_val, psnr_train, psnr_val
