"""Unified driver for the ISGD/IAdam schedule experiments (TCI revision).

Reproduces the paper's denoising setup: BSDS500 crops, sigma = 25/255,
CRR or ICNN regularizer, polynomial accuracy schedule eps_k = eps0 * k^{-p}
and step size schedule alpha_k = alpha0 * k^{-q}, with total computational
cost measured as lower-level iterations + Hessian-system (MINRES) iterations.

Example:
    python run_isgd.py --arch crr --alpha0 1e-2 --eps0 1e-2 --p 0.5 --q 0.0 \
        --optimizer sgd --budget 1e5
"""

import argparse
import json
import torch
from torchvision.transforms import (
    RandomCrop,
    CenterCrop,
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)

from deepinv.physics import Denoising, GaussianNoise
from deepinv.optim import L2

from dataset import get_dataset
from priors import wcrr, ICNNPrior
from training_methods import bilevel_training_ISGD


def build_datasets(color, train_size, test_size, crop, test_crop):
    key = "BSDS500_color" if color else "BSDS500_gray"
    train_transform = Compose(
        [
            RandomCrop(crop),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ]
    )
    train_dataset = get_dataset(key, test=False, transform=train_transform)
    test_dataset = get_dataset(key, test=True, transform=CenterCrop(test_crop))
    train_set = torch.utils.data.Subset(train_dataset, range(min(train_size, len(train_dataset))))
    test_set = torch.utils.data.Subset(test_dataset, range(min(test_size, len(test_dataset))))
    return train_set, test_set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["crr", "icnn"], default="crr")
    parser.add_argument("--alpha0", type=float, default=1e-2, help="initial upper-level step size")
    parser.add_argument("--eps0", type=float, default=1e-2, help="initial lower-level accuracy")
    parser.add_argument("--p", type=float, default=0.0, help="accuracy decay exponent")
    parser.add_argument("--q", type=float, default=0.0, help="step size decay exponent")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--budget", type=float, default=1e5, help="total cost budget (lower + CG iters)")
    parser.add_argument("--epochs", type=int, default=10000, help="upper bound; budget is the real stop")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=16)
    parser.add_argument("--crop", type=int, default=64)
    parser.add_argument("--test-crop", type=int, default=192)
    parser.add_argument("--gray", action="store_true", help="grayscale instead of color")
    parser.add_argument("--sigma", type=float, default=25 / 255)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-dir", default="logs/TCI")
    parser.add_argument("--vis-dir", default=None)
    parser.add_argument("--stall-patience", type=int, default=None,
                        help="precision-floor detection for the lower-level solver (e.g. 25)")
    parser.add_argument("--stop-criterion", choices=["relative", "grad_rms"], default="relative",
                        help="grad_rms = dtype-aware per-element RMS criterion (MAID/deepinv PR #1318)")
    parser.add_argument("--lower-mu", type=float, default=2.0,
                        help="strong convexity constant of the lower-level objective (denoising: 2)")
    parser.add_argument("--checkpoint-costs", type=float, nargs="*", default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny setup for a quick end-to-end validation run")
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

    train_set, test_set = build_datasets(
        color=not args.gray,
        train_size=args.train_size,
        test_size=args.test_size,
        crop=args.crop,
        test_crop=args.test_crop,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, drop_last=False, num_workers=0
    )

    channels = 1 if args.gray else 3
    if args.arch == "crr":
        regularizer = wcrr.WCRR(
            sigma=args.sigma,
            weak_convexity=0.0,
            nb_channels=[channels, 4, 8, 64],
            device=args.device,
        ).to(args.device)
    else:
        regularizer = ICNNPrior(in_channels=channels, channels=32, device=args.device).to(args.device)

    physics = Denoising(noise_model=GaussianNoise(sigma=args.sigma))
    data_fidelity = L2(sigma=1.0)
    lmbd = 1.0

    print(json.dumps(vars(args), indent=2, default=str))

    bilevel_training_ISGD(
        regularizer,
        physics,
        data_fidelity,
        lmbd,
        train_dataloader,
        test_dataloader,
        epochs=args.epochs,
        NAG_step_size=1e-1,
        NAG_max_iter=1000,
        NAG_tol_train=args.eps0,
        NAG_tol_val=1e-4,
        minres_max_iter=1000,
        minres_tol=args.eps0,
        lr=args.alpha0,
        device=args.device,
        verbose=False,
        p=args.p,
        q=args.q,
        optimizer_type=args.optimizer,
        budget=args.budget,
        checkpoint_costs=args.checkpoint_costs,
        log_dir=args.log_dir,
        vis_dir=args.vis_dir,
        stall_patience=args.stall_patience,
        stop_criterion=args.stop_criterion,
        lower_mu=args.lower_mu,
        run_tag=args.tag or args.arch + "_" + args.optimizer,
    )


if __name__ == "__main__":
    main()
