"""Budget-matched comparison: scheduled ISGD/IAdam vs single-loop bilevel methods.

All methods see the SAME fixed dataset (deterministic crops, fixed noise
realizations) so the bilevel problem is identical across methods, which is a
prerequisite for single-loop methods that maintain persistent (x, z) iterates.
The cost meter is shared: one batched lower-level gradient step / solver
iteration = 1, one batched Hessian-vector product / MINRES iteration = 1.

Examples:
    python run_comparison.py --method soba --alpha 1e-2 --beta 1e-1
    python run_comparison.py --method isgd --alpha0 1e-2 --eps0 1e-2 --p 0.5
"""

import argparse
import json
import torch
from torchvision.transforms import CenterCrop, Compose

from deepinv.physics import Denoising, GaussianNoise
from deepinv.optim import L2

from dataset import get_dataset
from priors import wcrr, ICNNPrior
from training_methods import bilevel_training_ISGD, bilevel_training_singleloop


def build_fixed_pairs(color, train_size, test_size, crop, test_crop, sigma, seed, device):
    """Deterministic crops + one fixed noise realization per image."""
    key = "BSDS500_color" if color else "BSDS500_gray"
    train_dataset = get_dataset(key, test=False, transform=CenterCrop(crop))
    test_dataset = get_dataset(key, test=True, transform=CenterCrop(test_crop))

    g = torch.Generator().manual_seed(seed)
    xs = torch.stack([train_dataset[i] for i in range(min(train_size, len(train_dataset)))])
    ys = xs + sigma * torch.randn(xs.shape, generator=g)
    x_test = torch.stack([test_dataset[i] for i in range(min(test_size, len(test_dataset)))])
    y_test = x_test + sigma * torch.randn(x_test.shape, generator=g)
    return xs.to(device), ys.to(device), x_test.to(device), y_test.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True,
                        choices=["isgd", "iadam", "ttsa", "soba", "saba", "masoba"])
    parser.add_argument("--arch", choices=["crr", "icnn"], default="crr")
    # single-loop hyperparameters
    parser.add_argument("--alpha", type=float, default=1e-2, help="upper-level step size")
    parser.add_argument("--beta", type=float, default=1e-1, help="lower/auxiliary step size")
    parser.add_argument("--alpha-decay", type=float, default=0.0)
    parser.add_argument("--beta-decay", type=float, default=0.0)
    parser.add_argument("--ma-rho", type=float, default=0.2)
    parser.add_argument("--neumann-terms", type=int, default=10)
    parser.add_argument("--no-auto-lipschitz", action="store_true",
                        help="use raw beta instead of beta/L_hat (legacy, unstable)")
    parser.add_argument("--warm-start", action="store_true",
                        help="ISGD/IAdam: persistent per-sample warm starts for the inner solver")
    # ISGD hyperparameters
    parser.add_argument("--stop-criterion", choices=["relative", "grad_rms"], default="relative")
    parser.add_argument("--alpha0", type=float, default=1e-2)
    parser.add_argument("--eps0", type=float, default=1e-2)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--q", type=float, default=0.0)
    # shared
    parser.add_argument("--budget", type=float, default=1e5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=16)
    parser.add_argument("--crop", type=int, default=64)
    parser.add_argument("--test-crop", type=int, default=192)
    parser.add_argument("--gray", action="store_true")
    parser.add_argument("--sigma", type=float, default=25 / 255)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-dir", default="logs/TCI_comparison")
    parser.add_argument("--tag", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    if args.smoke:
        args.train_size = 16
        args.test_size = 2
        args.budget = min(args.budget, 500)
        args.test_crop = 96

    torch.manual_seed(args.seed)

    xs, ys, x_test, y_test = build_fixed_pairs(
        color=not args.gray, train_size=args.train_size, test_size=args.test_size,
        crop=args.crop, test_crop=args.test_crop, sigma=args.sigma,
        seed=args.seed, device=args.device,
    )

    channels = 1 if args.gray else 3
    if args.arch == "crr":
        regularizer = wcrr.WCRR(
            sigma=args.sigma, weak_convexity=0.0,
            nb_channels=[channels, 4, 8, 64], device=args.device,
        ).to(args.device)
    else:
        regularizer = ICNNPrior(in_channels=channels, channels=32, device=args.device).to(args.device)

    physics = Denoising(noise_model=GaussianNoise(sigma=args.sigma))
    data_fidelity = L2(sigma=1.0)
    lmbd = 1.0

    print(json.dumps(vars(args), indent=2, default=str))
    tag = args.tag or f"{args.arch}"

    if args.method in ("isgd", "iadam"):
        if args.warm_start:
            train_pairs = torch.utils.data.TensorDataset(
                xs.cpu(), ys.cpu(), torch.arange(xs.shape[0]))
        else:
            train_pairs = torch.utils.data.TensorDataset(xs.cpu(), ys.cpu())
        test_pairs = torch.utils.data.TensorDataset(x_test.cpu(), y_test.cpu())
        train_dataloader = torch.utils.data.DataLoader(
            train_pairs, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(
            test_pairs, batch_size=1, shuffle=False)
        bilevel_training_ISGD(
            regularizer, physics, data_fidelity, lmbd,
            train_dataloader, test_dataloader,
            epochs=100000,
            NAG_step_size=1e-1, NAG_max_iter=1000,
            NAG_tol_train=args.eps0, NAG_tol_val=1e-4,
            minres_max_iter=1000, minres_tol=args.eps0,
            lr=args.alpha0, device=args.device,
            p=args.p, q=args.q,
            optimizer_type="sgd" if args.method == "isgd" else "adam",
            budget=args.budget, log_dir=args.log_dir,
            persistent_warm_start=args.warm_start,
            stop_criterion=args.stop_criterion,
            run_tag=f"{tag}_{args.method}{'_ws' if args.warm_start else ''}",
        )
    else:
        bilevel_training_singleloop(
            regularizer, physics, data_fidelity, lmbd,
            x_star=xs, y_obs=ys, val_pairs=(x_test, y_test),
            method=args.method,
            alpha=args.alpha, beta=args.beta,
            alpha_decay=args.alpha_decay, beta_decay=args.beta_decay,
            ma_rho=args.ma_rho, neumann_terms=args.neumann_terms,
            auto_lipschitz=not args.no_auto_lipschitz,
            batch_size=args.batch_size, budget=args.budget,
            eval_every_cost=100 if args.smoke else 2000,
            NAG_tol_val=1e-4, device=args.device,
            log_dir=args.log_dir, run_tag=tag, seed=args.seed,
        )


if __name__ == "__main__":
    main()
