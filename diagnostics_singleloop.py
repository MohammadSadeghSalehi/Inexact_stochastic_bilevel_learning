"""Correctness diagnostics for the single-loop bilevel implementations.

Test A (real problem): at a fixed theta, iterate the single-loop (x, z)
updates to convergence and compare the resulting theta-direction against the
reference hypergradient computed with high-accuracy solves (nmAPG grad_rms +
tight MINRES). Reports cosine similarity and relative magnitude.
  - SOBA/SABA/MA-SOBA share the same z fixed point Hz = grad g, so one test
    covers the family; TTSA is tested with its truncated Neumann z.

Test B (synthetic, closed form): scalar-parameter quadratic bilevel
  h_i(x, w) = 1/2 ||x - y_i||^2 + w/2 ||x||^2  =>  xhat_i = y_i / (1 + w)
  f(w) = sum_i || xhat_i - x*_i ||^2  with analytic df/dw.
Checks the sign and magnitude of the theta-direction and that the full
single-loop training decreases f and approaches the analytic minimizer.

Test C (mechanism): rerun a short TTSA training and track
  || x_buf - xhat(theta) || (tracking error), true bilevel loss at xhat,
  apparent loss at x_buf, and L_hat -- to explain the climb-then-decay.
"""

import math
import torch
import torch.nn as nn

from deepinv.physics import Denoising, GaussianNoise
from deepinv.optim import L2

from priors import wcrr
from evaluation import reconstruct_nmAPG2, minres
from training_methods.bilevel_training_singleloop import (
    _lower_grad, _hvp, _theta_direction, _estimate_lipschitz,
)
from run_comparison import build_fixed_pairs

DEV = "cpu"
torch.manual_seed(23)


def flat(gs):
    return torch.cat([g.reshape(-1) for g in gs if g is not None])


def reference_direction(xs, ys, reg, lmbd, df, physics):
    """High-accuracy hypergradient direction (mean over samples):
    d_ref = mean_i grad_theta <grad_x h(xhat_i, theta), z_i>, H z_i = grad g_i."""
    m = xs.shape[0]
    dirs = None
    for i in range(m):
        y = ys[i:i+1]
        xg = xs[i:i+1]
        xhat = reconstruct_nmAPG2(
            y, physics, df, reg, lmbd, 1e-1, 5000, 1e-7,
            x_init=y, stop_criterion="grad_rms", mu=2.0, stall_patience=50,
        )
        gg = 2.0 * (xhat - xg)
        z, _ = minres(
            lambda v: _hvp(xhat, v, y, reg, lmbd, df, physics),
            gg, max_iter=5000, tol=1e-10,
        )
        d = _theta_direction(xhat, z, y, reg, lmbd, df, physics)
        dirs = d if dirs is None else [a + b for a, b in zip(dirs, d)]
    return [d / m for d in dirs]


def singleloop_direction(xs, ys, reg, lmbd, df, physics, mode, iters=3000, beta_rel=0.5, neumann=10):
    """Freeze theta; iterate the production (x, z) updates to convergence,
    then form the theta-direction exactly as bilevel_training_singleloop does."""
    m = xs.shape[0]
    x_buf = ys.clone()
    z_buf = torch.zeros_like(x_buf)
    L_hat, _ = _estimate_lipschitz(x_buf, ys, reg, lmbd, df, physics)
    beta = beta_rel / max(L_hat, 1e-3)
    for _ in range(iters):
        gx = _lower_grad(x_buf, ys, reg, lmbd, df, physics)
        x_buf = x_buf - beta * gx
        if mode == "soba":
            hz = _hvp(x_buf, z_buf, ys, reg, lmbd, df, physics)
            z_buf = z_buf - beta * (hz - 2.0 * (x_buf - xs))
    if mode == "ttsa":
        v = 2.0 * (x_buf - xs)
        acc = v.clone()
        for _ in range(neumann - 1):
            hv = _hvp(x_buf, v, ys, reg, lmbd, df, physics)
            v = v - beta * hv
            acc = acc + v
        z_buf = beta * acc
    d = _theta_direction(x_buf, z_buf, ys, reg, lmbd, df, physics)
    return [g / m if g is not None else None for g in d]


def test_A():
    print("=" * 70)
    print("TEST A: single-loop direction vs high-accuracy hypergradient")
    print("=" * 70)
    xs, ys, _, _ = build_fixed_pairs(True, 8, 2, 64, 96, 25 / 255, 23, DEV)
    physics = Denoising(noise_model=GaussianNoise(sigma=25 / 255))
    df = L2(sigma=1.0)
    reg = wcrr.WCRR(sigma=25/255, weak_convexity=0.0, nb_channels=[3, 4, 8, 64], device=DEV).to(DEV)
    lmbd = 1.0

    d_ref = flat(reference_direction(xs, ys, reg, lmbd, df, physics))
    for mode, extra in [("soba", {}), ("ttsa", {"neumann": 10}), ("ttsa", {"neumann": 100})]:
        d_sl = flat(singleloop_direction(xs, ys, reg, lmbd, df, physics, mode, **extra))
        cos = torch.dot(d_ref, d_sl) / (d_ref.norm() * d_sl.norm())
        ratio = (d_sl.norm() / d_ref.norm()).item()
        label = mode + (f"(N={extra.get('neumann')})" if mode == "ttsa" else "(z fixed point)")
        print(f"  {label:22s} cosine={cos.item():+.4f}  |d_sl|/|d_ref|={ratio:.3f}")
    print("  (expected: cosine ~ +1; TTSA N=10 ratio < 1 = truncation bias)")


class ScalarPrior(nn.Module):
    """R_w(x) = softplus-free scalar quadratic w/2 ||x||^2 with w > 0."""
    def __init__(self, w0=2.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(w0)))
    def g(self, x):
        return 0.5 * self.w * (x ** 2).reshape(x.shape[0], -1).sum(-1)
    def grad(self, x):
        return self.w * x


def test_B():
    print("=" * 70)
    print("TEST B: synthetic quadratic bilevel with closed-form hypergradient")
    print("=" * 70)
    m, n = 4, 50
    torch.manual_seed(0)
    x_star = torch.randn(m, 1, 1, n)
    y = 2.0 * x_star + 0.1 * torch.randn(m, 1, 1, n)   # optimum w* near 1
    physics = Denoising(noise_model=GaussianNoise(sigma=0.0))
    df = L2(sigma=1.0)
    reg = ScalarPrior(w0=2.0)
    lmbd = 1.0

    # analytic: xhat_i = y_i/(1+w);  f = sum_i ||xhat_i - x*_i||^2 / m
    def f_true(w):
        return (((y / (1 + w)) - x_star) ** 2).reshape(m, -1).sum(-1).mean()
    def df_dw(w):
        xh = y / (1 + w)
        return (2 * (xh - x_star) * (-y / (1 + w) ** 2)).reshape(m, -1).sum(-1).mean()

    w0 = 2.0
    # single-loop direction at converged x, z (theta frozen)
    x_buf = y.clone()
    z_buf = torch.zeros_like(x_buf)
    beta = 0.3 / (1 + w0)
    for _ in range(2000):
        gx = df.grad(x_buf, y, physics) + lmbd * reg.grad(x_buf)
        x_buf = x_buf - beta * gx
        hz = _hvp(x_buf, z_buf, y, reg, lmbd, df, physics)
        z_buf = z_buf - beta * (hz - 2.0 * (x_buf - x_star))
    d = _theta_direction(x_buf, z_buf, y, reg, lmbd, df, physics)
    d_code = (d[0] / m).item()
    print(f"  analytic  -df/dw(w0) = {-df_dw(w0).item():+.6f}")
    print(f"  code   direction     = {d_code:+.6f}   (theta step is +alpha*direction)")
    print(f"  match: {'YES' if abs(d_code - (-df_dw(w0).item())) < 5e-3 * max(1, abs(df_dw(w0))) else 'NO'}")

    # full training loop convergence to analytic minimizer
    from training_methods import bilevel_training_singleloop
    reg2 = ScalarPrior(w0=2.0)
    _, logs = bilevel_training_singleloop(
        reg2, physics, df, lmbd, x_star=x_star, y_obs=y, val_pairs=None,
        method="soba", alpha=5e-4, beta=0.3, batch_size=2, budget=4000,
        eval_every_cost=10**9, device=DEV, log_dir="logs/TCI_comparison/diag",
        run_tag="synth", seed=0, auto_lipschitz=True,
    )
    ws = torch.linspace(0.1, 3.0, 400)
    w_star = ws[torch.argmin(torch.stack([f_true(w) for w in ws]))].item()
    print(f"  trained w = {reg2.w.item():.4f}   analytic w* = {w_star:.4f}   f(w)={f_true(reg2.w.detach()).item():.4f} f(w*)={f_true(torch.tensor(w_star)).item():.4f}")


def test_C():
    print("=" * 70)
    print("TEST C: TTSA climb-then-decay mechanism (tracking-error diagnosis)")
    print("=" * 70)
    xs, ys, x_t, y_t = build_fixed_pairs(True, 16, 2, 64, 96, 25 / 255, 23, DEV)
    physics = Denoising(noise_model=GaussianNoise(sigma=25 / 255))
    df = L2(sigma=1.0)
    from deepinv.loss.metric import PSNR
    psnr = PSNR()
    lmbd = 1.0

    for label, alpha, a_dec, b_dec in [
        ("constant alpha (production)", 1e-3, 0.0, 0.0),
        ("TTSA-theory decay a~k^-3/5, b~k^-2/5", 1e-2, 0.6, 0.4),
    ]:
        torch.manual_seed(23)
        reg = wcrr.WCRR(sigma=25/255, weak_convexity=0.0, nb_channels=[3, 4, 8, 64], device=DEV).to(DEV)
        x_buf = ys.clone()
        L_hat, _ = _estimate_lipschitz(x_buf, ys, reg, lmbd, df, physics)
        params = [p for p in reg.parameters() if p.requires_grad]
        print(f"  --- {label} ---")
        m = xs.shape[0]
        k = 0
        for it in range(1, 1201):
            k += 1
            idx = torch.randperm(m)[:8]
            beta_k = 0.5 * (k ** (-b_dec)) / L_hat
            alpha_k = alpha * (k ** (-a_dec))
            xb = x_buf[idx]
            gx = _lower_grad(xb, ys[idx], reg, lmbd, df, physics)
            xb = xb - beta_k * gx
            x_buf[idx] = xb.detach()
            eta = 0.5 / L_hat
            v = 2.0 * (xb - xs[idx]); acc = v.clone()
            for _ in range(9):
                hv = _hvp(xb, v, ys[idx], reg, lmbd, df, physics)
                v = v - eta * hv
                acc = acc + v
            zb = eta * acc
            d = _theta_direction(xb, zb, ys[idx], reg, lmbd, df, physics)
            with torch.no_grad():
                for p_, d_ in zip(params, d):
                    if d_ is not None:
                        p_.add_(alpha_k * d_ / 8)
            if it % 100 == 0:
                L_hat = max(_estimate_lipschitz(x_buf[idx], ys[idx], reg, lmbd, df, physics, iters=10)[0], 1e-3)
            if it % 300 == 0:
                with torch.no_grad():
                    pass
                # true lower-level solutions on a probe batch
                probe = slice(0, 4)
                xhat = reconstruct_nmAPG2(ys[probe], physics, df, reg, lmbd, 1e-1, 3000, 1e-6,
                                          x_init=x_buf[probe], stop_criterion="grad_rms", mu=2.0,
                                          stall_patience=50)
                track = (x_buf[probe] - xhat).reshape(4, -1).norm(dim=1).mean().item()
                loss_buf = ((x_buf[probe] - xs[probe]) ** 2).reshape(4, -1).sum(-1).mean().item()
                loss_true = ((xhat - xs[probe]) ** 2).reshape(4, -1).sum(-1).mean().item()
                rec_t = reconstruct_nmAPG2(y_t[0:1], physics, df, reg, lmbd, 1e-1, 2000, 1e-4, x_init=y_t[0:1])
                p_t = psnr(rec_t, x_t[0:1]).mean().item()
                print(f"    it={it:5d} L={L_hat:7.1f} track_err={track:8.3f} "
                      f"loss(x_buf)={loss_buf:8.2f} loss(xhat)={loss_true:8.2f} test_psnr={p_t:.2f}")


if __name__ == "__main__":
    test_A()
    test_B()
    test_C()
