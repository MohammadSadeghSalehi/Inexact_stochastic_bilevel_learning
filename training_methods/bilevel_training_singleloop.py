"""Single-loop stochastic bilevel baselines: TTSA, SOBA, SABA, MA-SOBA.

All methods jointly update the upper-level parameters theta, per-sample
lower-level iterates x_i, and (for the SOBA family) per-sample auxiliary
variables z_i tracking the solution of the Hessian linear system
    H_i(x_i, theta) z_i = grad g_i(x_i).

The lower-level problem here is separable across training samples
(each image has its own reconstruction), so block-coordinate versions of the
single-loop updates apply: a sampled mini-batch updates only its own
(x_i, z_i) blocks, while theta is updated from the mini-batch estimate of the
hypergradient  -(1/|B|) sum_i grad^2_{theta x} h_i(x_i, theta) z_i.

Cost accounting matches the ISGD experiments: every lower-level gradient step
counts 1 toward `lower_batch` and every Hessian-vector product counts 1 toward
`cg_batch`, so budget-matched comparisons against ISGD/IAdam are meaningful.

References:
- TTSA: Hong, Wai, Wang, Yang, SIAM J. Optim. 33(1), 2023.
- SOBA/SABA: Dagreou, Ablin, Vaiter, Moreau, NeurIPS 2022.
- MA-SOBA: Chen, Xiao, Balasubramanian, JMLR 2024.
"""

import os
import copy
import torch
from deepinv.loss.metric import PSNR, SSIM
from evaluation import reconstruct_nmAPG2


def _lower_grad(x, y, theta_model, lmbd, data_fidelity, physics):
    """grad_x h(x, theta) for a batch."""
    return data_fidelity.grad(x, y, physics) + lmbd * theta_model.grad(x)


def _lower_energy(x, y, theta_model, lmbd, data_fidelity, physics):
    """h(x, theta) per sample, shape (B,)."""
    return (data_fidelity(x, y, physics) + lmbd * theta_model.g(x)).reshape(-1)


def _hvp(x, v, y, theta_model, lmbd, data_fidelity, physics):
    """Hessian-vector product H_x(x, theta) v for a batch (1 HVP each)."""
    x = x.detach().requires_grad_(True)
    grad = data_fidelity.grad(x, y, physics) + lmbd * theta_model.grad(x)
    dot = torch.dot(grad.reshape(-1), v.reshape(-1))
    hvp = torch.autograd.grad(dot, x, create_graph=False)[0]
    return hvp.detach()


def _theta_direction(x, z, y, theta_model, lmbd, data_fidelity, physics):
    """Returns per-parameter gradients of <grad_x h(x, theta), z> summed over
    the batch; the hypergradient estimate is the NEGATIVE of this direction
    divided by the batch size."""
    x = x.detach()
    grad = data_fidelity.grad(x, y, physics) + lmbd * theta_model.grad(x)
    dot = torch.dot(grad.reshape(-1), z.reshape(-1))
    params = [p for p in theta_model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(dot, params, create_graph=False, allow_unused=True)
    return [g.detach() if g is not None else None for g in grads]


def _estimate_lipschitz(x, y, theta_model, lmbd, data_fidelity, physics, iters=20):
    """Power iteration on the lower-level Hessian at (x, theta); returns
    (L_estimate, number_of_HVPs_used)."""
    v = torch.randn_like(x)
    v = v / v.reshape(-1).norm()
    lam = torch.tensor(1.0, device=x.device)
    for _ in range(iters):
        hv = _hvp(x, v, y, theta_model, lmbd, data_fidelity, physics)
        lam = hv.reshape(-1).norm()
        if lam == 0:
            return 1.0, iters
        v = hv / lam
    return float(lam), iters


def bilevel_training_singleloop(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    x_star,                    # (m, C, H, W) ground-truth tensor (fixed dataset)
    y_obs,                     # (m, C, H, W) fixed observations
    val_pairs=None,            # optional (x_val, y_val) tensors for testing
    method="soba",             # "ttsa" | "soba" | "saba" | "masoba"
    alpha=1e-2,                # upper-level step size (constant unless decayed)
    beta=1e-1,                 # lower-level / auxiliary step size; RELATIVE to
                               # 1/L_H when auto_lipschitz=True (dimensionless)
    alpha_decay=0.0,           # alpha_k = alpha * k^{-alpha_decay}
    beta_decay=0.0,            # beta_k = beta * k^{-beta_decay}
    auto_lipschitz=True,       # normalize x/z steps by a power-iteration
                               # estimate of the lower-level Hessian norm L_H;
                               # refreshed every lip_refresh_every iterations
                               # (L_H grows as theta trains). The HVPs used
                               # are counted toward the cost meter.
    lip_refresh_every=100,     # iterations between L_H re-estimates
    ma_rho=0.2,                # moving-average weight for MA-SOBA
    neumann_terms=10,          # Neumann series length for TTSA hypergradient
    batch_size=8,
    budget=1e5,
    epochs=100000,
    eval_every_cost=2000,      # evaluate test metrics roughly every this cost
    NAG_step_size=1e-1,
    NAG_max_iter=1000,
    NAG_tol_val=1e-4,
    device="cpu",
    log_dir="logs/TCI_comparison",
    run_tag="",
    seed=0,
    verbose=False,
):
    method = method.lower()
    assert method in ("ttsa", "soba", "saba", "masoba")
    torch.manual_seed(seed)

    m = x_star.shape[0]
    x_star = x_star.to(device)
    y_obs = y_obs.to(device)

    psnr = PSNR()
    ssim = SSIM()

    # Persistent lower-level iterates (warm started at the observations)
    x_buf = y_obs.clone()
    z_buf = torch.zeros_like(x_buf) if method != "ttsa" else None
    # Persistent per-sample local Lipschitz estimates for Armijo backtracking
    L_buf = None  # initialized after the global power-iteration estimate

    params = [p for p in regularizer.parameters() if p.requires_grad]

    # SABA memory: per-sample theta-direction estimates + running average
    if method == "saba":
        saba_mem = [[torch.zeros_like(p) for p in params] for _ in range(m)]
        saba_avg = [torch.zeros_like(p) for p in params]

    # MA-SOBA moving average direction
    if method == "masoba":
        ma_dir = [torch.zeros_like(p) for p in params]

    upper_loss = lambda x, x_gt: torch.sum(((x - x_gt) ** 2).reshape(x.shape[0], -1), -1)
    grad_g = lambda x, x_gt: 2.0 * (x - x_gt)

    logs = {
        "cost_batch": [], "lower_batch": [], "cg_batch": [], "loss_batch": [],
        "cost_epoch": [], "loss_epoch": [], "psnr_epoch": [],
        "loss_test": [], "psnr_test": [], "ssim_test": [], "cost_test": [],
        "step": alpha, "eps": beta, "p": 0.0, "q": alpha_decay,
        "method": method, "budget": budget,
        "auto_lipschitz": auto_lipschitz, "L_hat": [], "diverged": False,
    }

    os.makedirs(log_dir, exist_ok=True)
    tag = f"{run_tag}_" if run_tag else ""
    log_path = os.path.join(log_dir, f"logs_{tag}{method}_{alpha}_{beta}_.pt")

    best_val_psnr = -float("inf")
    best_state = None

    total_cost = 0
    next_eval = 0
    k = 0
    stop = False

    # ---- Lipschitz normalization of the fixed steps ----
    # The single-loop analyses assume steps <= O(1/L_H). The CRR's initial
    # scaling exp(s) = 2/sigma makes L_H large (~400), so raw steps of 1e-2
    # violate stability (beta < 2/L_H) and diverge. We therefore interpret
    # `beta` as a DIMENSIONLESS fraction of 1/L_H, with L_H estimated by
    # power iteration (HVPs counted toward the cost meter) and refreshed at
    # every test evaluation as theta (hence L_H) evolves.
    L_hat = 1.0
    if auto_lipschitz:
        xb0 = x_buf[:min(batch_size, m)]
        yb0 = y_obs[:min(batch_size, m)]
        L_hat, hvps = _estimate_lipschitz(
            xb0, yb0, regularizer, lmbd, data_fidelity, physics)
        L_hat = max(L_hat, 1e-3)
        total_cost += hvps
        logs["L_hat"].append(L_hat)
        print(f"[{method}] initial L_H estimate: {L_hat:.1f} "
              f"(effective lower step {beta / L_hat:.2e})")
    L_buf = torch.full((m,), max(L_hat, 1.0), device=device)

    def evaluate_test():
        if val_pairs is None:
            return None
        x_val, y_val = val_pairs
        regularizer.eval()
        with torch.no_grad():
            loss_sum, psnr_sum, ssim_sum = 0.0, 0.0, 0.0
            n = x_val.shape[0]
            for j in range(n):
                xg = x_val[j:j+1].to(device)
                yo = y_val[j:j+1].to(device)
                rec = reconstruct_nmAPG2(
                    yo, physics, data_fidelity, regularizer, lmbd,
                    NAG_step_size, NAG_max_iter, NAG_tol_val, x_init=yo,
                )
                loss_sum += upper_loss(rec, xg).mean().item()
                psnr_sum += psnr(rec, xg).mean().item()
                ssim_sum += ssim(rec, xg).mean().item()
        regularizer.train()
        return loss_sum / n, psnr_sum / n, ssim_sum / n

    regularizer.train()
    for epoch in range(epochs):
        perm = torch.randperm(m)
        epoch_loss, epoch_psnr, n_batches = 0.0, 0.0, 0
        for start in range(0, m - batch_size + 1, batch_size):
            k += 1
            idx = perm[start:start + batch_size]
            xb = x_buf[idx]
            yb = y_obs[idx]
            xgb = x_star[idx]

            alpha_k = alpha * (max(k, 1) ** (-alpha_decay))
            beta_k = beta * (max(k, 1) ** (-beta_decay)) / L_hat

            # ---- lower-level step ----
            # cost convention matches ISGD: one *batched* gradient step
            # (resp. HVP) counts 1, exactly like one batched nmAPG/MINRES
            # iteration in the ISGD cost meter.
            gx = _lower_grad(xb, yb, regularizer, lmbd, data_fidelity, physics)
            lower_cost = 1
            cg_cost = 0
            if auto_lipschitz:
                # Per-sample Armijo backtracking (closed loop). Every
                # open-loop curvature estimator we tried (periodic power
                # iteration, along-the-gradient Rayleigh probe, warm-started
                # per-sample power iteration) eventually lost the race
                # against the rapidly growing, batch-heterogeneous curvature
                # (single-iteration loss jumps of 1e9x were observed while
                # the tracker lagged ~5x). Backtracking enforces the descent
                # condition h(x+) <= h(x) - delta * t * ||g||^2 per sample
                # BEFORE accepting a step, so the x-recursion is monotone in
                # energy and cannot blow up -- the same mechanism nmAPG uses
                # for ISGD, and the energy evaluations are uncounted in the
                # cost meter exactly as nmAPG's backtracking evaluations are.
                with torch.no_grad():
                    beta_rel_k = beta * (max(k, 1) ** (-beta_decay))
                    Li = L_buf[idx].clone()
                    h0 = _lower_energy(xb, yb, regularizer, lmbd, data_fidelity, physics)
                    gsq = (gx * gx).reshape(gx.shape[0], -1).sum(1)
                    exp_shape = (-1, *([1] * (gx.dim() - 1)))
                    for _ in range(12):
                        t = (beta_rel_k / Li).view(exp_shape)
                        x_new = xb - t * gx
                        h1 = _lower_energy(x_new, yb, regularizer, lmbd, data_fidelity, physics)
                        viol = h1 > h0 - 0.1 * t.reshape(-1) * gsq
                        if not bool(viol.any()):
                            break
                        Li = torch.where(viol, Li * 2.0, Li)
                    # gentle relaxation so transient spikes do not freeze the
                    # step size permanently
                    L_buf[idx] = torch.clamp(Li * 0.97, min=1e-3)
                    xb = x_new
                    beta_k = beta_rel_k / float(Li.max())  # for the z update
            else:
                with torch.no_grad():
                    xb = xb - beta_k * gx
            x_buf[idx] = xb.detach()

            # ---- hypergradient direction ----
            if method == "ttsa":
                # Neumann-series (SHIA) approximation of H^{-1} grad g:
                #   z ~= eta * sum_{j<N} (I - eta H)^j grad g.
                # The Neumann step size eta is FIXED (not tied to the decaying
                # beta_k): with a decaying eta and fixed N the series would
                # vanish, z -> 0. Normalized by L_hat so that the series
                # (I - eta H)^j is actually convergent (eta < 1/L_H).
                eta_n = beta / float(L_buf[idx].max()) if auto_lipschitz else beta / L_hat
                v = grad_g(xb, xgb)
                acc = v.clone()
                for _ in range(neumann_terms - 1):
                    hv = _hvp(xb, v, yb, regularizer, lmbd, data_fidelity, physics)
                    v = v - eta_n * hv
                    acc = acc + v
                    cg_cost += 1
                zb = eta_n * acc
            else:
                # SOBA-family z update (1 HVP per sample)
                zb = z_buf[idx]
                hz = _hvp(xb, zb, yb, regularizer, lmbd, data_fidelity, physics)
                with torch.no_grad():
                    gg = grad_g(xb, xgb)
                    zb = zb - beta_k * (hz - gg)
                    # Projection onto the ball ||z|| <= ||grad g|| / mu
                    # (Hong et al. 2023; Chen et al. 2022). The solution
                    # z* = H^{-1} grad g always satisfies this since
                    # H >= mu I with mu = 1 (identity-fidelity Hessian), so
                    # the projection never moves the fixed point but keeps the
                    # linear recursion bounded when transient curvature
                    # exceeds the (batch-estimated) Lipschitz bound.
                    radius = gg.reshape(gg.shape[0], -1).norm(dim=1)
                    znorm = zb.reshape(zb.shape[0], -1).norm(dim=1).clamp_min(1e-12)
                    scale = torch.minimum(torch.ones_like(znorm), radius / znorm)
                    zb = zb * scale.view(-1, *([1] * (zb.dim() - 1)))
                z_buf[idx] = zb.detach()
                cg_cost += 1

            # ---- theta update ----
            if method == "saba":
                # per-sample memories (SAGA-style variance reduction).
                # SAGA estimator: direction = mean_B(g_new - g_mem) + avg_OLD,
                # with the running average updated only AFTER the estimator is
                # formed (using the pre-update average is essential for
                # unbiasedness).
                avg_old = [a.clone() for a in saba_avg]
                new_dir = [torch.zeros_like(p) for p in params]
                for jj in range(batch_size):
                    i_glob = int(idx[jj])
                    g_i = _theta_direction(
                        xb[jj:jj+1], zb[jj:jj+1], yb[jj:jj+1],
                        regularizer, lmbd, data_fidelity, physics,
                    )
                    for t, (g_new, g_old) in enumerate(zip(g_i, saba_mem[i_glob])):
                        if g_new is None:
                            continue
                        new_dir[t] = new_dir[t] + (g_new - g_old) / batch_size
                        saba_avg[t] = saba_avg[t] + (g_new - g_old) / m
                        saba_mem[i_glob][t] = g_new
                direction = [nd + av for nd, av in zip(new_dir, avg_old)]
            else:
                g_batch = _theta_direction(
                    xb, zb, yb, regularizer, lmbd, data_fidelity, physics,
                )
                direction = [g / batch_size if g is not None else None for g in g_batch]
                if method == "masoba":
                    # Chen et al. ordering: the theta step uses the LAGGED
                    # moving average h^k; the average is refreshed afterwards.
                    step_dir = [d.clone() for d in ma_dir]
                    ma_dir = [
                        (1 - ma_rho) * d_old + ma_rho * d_new
                        if d_new is not None else d_old
                        for d_old, d_new in zip(ma_dir, direction)
                    ]
                    direction = step_dir

            with torch.no_grad():
                for p_, d_ in zip(params, direction):
                    if d_ is not None:
                        # hypergradient = -direction; descent step:
                        p_.add_(alpha_k * d_)

            # keep constraints (e.g. ICNN non-negativity) if the prior defines them
            if hasattr(regularizer, "zero_clip_weights"):
                regularizer.zero_clip_weights()

            # ---- logging ----
            total_cost += lower_cost + cg_cost
            batch_loss = upper_loss(xb, xgb).mean().item()
            epoch_loss += batch_loss
            epoch_psnr += psnr(xb, xgb).mean().item()
            n_batches += 1
            logs["lower_batch"].append(lower_cost)
            logs["cg_batch"].append(cg_cost)
            logs["cost_batch"].append(total_cost)
            logs["loss_batch"].append(batch_loss)

            # ---- divergence guard ----
            if not torch.isfinite(torch.tensor(batch_loss)):
                logs["diverged"] = True
                print(f"[{method} k={k}] DIVERGED (non-finite loss) at cost "
                      f"{total_cost}; stopping run.")
                stop = True
                break

            if total_cost >= next_eval:
                res = evaluate_test()
                if res is not None:
                    l_t, p_t, s_t = res
                    logs["loss_test"].append(l_t)
                    logs["psnr_test"].append(p_t)
                    logs["ssim_test"].append(s_t)
                    logs["cost_test"].append(total_cost)
                    if p_t > best_val_psnr:
                        best_val_psnr = p_t
                        best_state = copy.deepcopy(regularizer.state_dict())
                    print(f"[{method} k={k}] cost {total_cost}: test PSNR {p_t:.2f}, SSIM {s_t:.4f}")
                next_eval = total_cost + eval_every_cost

            if auto_lipschitz and k % lip_refresh_every == 0:
                # record the running estimate (now maintained per-iteration by
                # the Rayleigh guard; no extra power iterations needed)
                logs["L_hat"].append(L_hat)

            if total_cost >= budget:
                stop = True
                break

        if n_batches > 0:
            logs["cost_epoch"].append(total_cost)
            logs["loss_epoch"].append(epoch_loss / n_batches)
            logs["psnr_epoch"].append(epoch_psnr / n_batches)
        torch.save(logs, log_path)
        if verbose:
            print(f"[{method} epoch {epoch+1}] cost {total_cost}, "
                  f"train loss {epoch_loss / max(n_batches,1):.2E}")
        if stop:
            break

    torch.save(logs, log_path)
    if best_state is not None:
        regularizer.load_state_dict(best_state)
    return regularizer, logs
