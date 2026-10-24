"""Analyze and plot the budget-matched comparison runs (TCI revision).

Usage:
    python plot_comparison.py rank logs/TCI_comparison/tune
        -> prints a ranking of tuning runs per method (final loss / best PSNR)
    python plot_comparison.py plot logs/TCI_comparison/medium figures
        -> writes cost-vs-loss and cost-vs-PSNR/SSIM figures + a summary table (LaTeX rows)
"""

import os
import re
import sys
import glob
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# Match the typography of the other figures in the paper (large fonts, since
# each panel is reproduced at roughly half a single column).
plt.rcParams.update({
    "font.size": 19,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 14,
    "lines.linewidth": 2.4,
    "figure.autolayout": True,
})

XLABEL = "Total lower-level computations"

METHOD_STYLE = {
    "isgd":   dict(color="tab:blue",   ls="-",  label="ISGD"),
    "isgd_q": dict(color="tab:cyan",   ls="--", label="ISGD ($q{=}0.5$)"),
    "iadam":  dict(color="tab:purple", ls="-",  label="IAdam"),
    "ttsa":   dict(color="tab:orange", ls="-",  label="TTSA"),
    "soba":   dict(color="tab:green",  ls="-",  label="SOBA"),
    "saba":   dict(color="tab:olive",  ls="--", label="SABA"),
    "masoba": dict(color="tab:red",    ls="-",  label="MA-SOBA"),
}


def load_logs(log_dir):
    runs = []
    for f in sorted(glob.glob(os.path.join(log_dir, "logs_*.pt"))):
        try:
            d = torch.load(f, map_location="cpu", weights_only=False, pickle_module=pickle)
        except Exception as e:
            print(f"skip {f}: {e}")
            continue
        name = os.path.basename(f)
        method = d.get("method")
        if method is None:
            # ISGD/IAdam logs: infer from filename/tag
            method = "iadam" if "iadam" in name else "isgd"
            if method == "isgd" and float(d.get("q", 0)) > 0:
                method = "isgd_q"
        runs.append(dict(file=name, method=method, logs=d))
    return runs


def moving_average(x, w=25):
    if len(x) < w:
        return np.asarray(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def rank(log_dir):
    runs = load_logs(log_dir)
    rows = []
    for r in runs:
        d = r["logs"]
        loss = [l for l in d.get("loss_batch", []) if np.isfinite(l)]
        final_loss = np.mean(loss[-10:]) if loss else float("inf")
        psnrs = [p for p in d.get("psnr_test", []) if np.isfinite(p)]
        best_psnr = max(psnrs) if psnrs else float("-inf")
        rows.append((r["method"], r["file"], final_loss, best_psnr))
    rows.sort(key=lambda t: (t[0], t[2]))
    cur = None
    for m, f, fl, bp in rows:
        if m != cur:
            print(f"\n== {m} ==")
            cur = m
        print(f"  final_loss={fl:9.3f}  best_test_psnr={bp:6.2f}  {f}")


def plot_comparison(log_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    runs = load_logs(log_dir)

    # pick the best run per method by best test PSNR
    best = {}
    for r in runs:
        psnrs = [p for p in r["logs"].get("psnr_test", []) if np.isfinite(p)]
        score = max(psnrs) if psnrs else float("-inf")
        if r["method"] not in best or score > best[r["method"]][0]:
            best[r["method"]] = (score, r)

    # --- cost vs training loss ---
    plt.figure(figsize=(7.4, 5.6))
    for m, (_, r) in sorted(best.items()):
        d = r["logs"]
        cost = np.asarray(d["cost_batch"], dtype=float)
        loss = np.asarray([l if np.isfinite(l) else np.nan for l in d["loss_batch"]])
        w = max(5, len(loss) // 100)
        ma = moving_average(loss, w)
        st = METHOD_STYLE.get(m, dict(label=m))
        plt.plot(cost[len(cost) - len(ma):], ma, **st)
    plt.xscale("log")
    plt.yscale("log")
    # Round 3 (co-author comment): crop to the informative range, less white space.
    plt.xlim(1e2, 2.1e4)
    plt.ylim(12, 190)  # headroom for the legend (upper right)
    plt.yticks([15, 20, 30, 50, 100], ["15", "20", "30", "50", "100"])
    plt.gca().yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.xlabel(XLABEL)
    plt.ylabel("Running average training loss")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, columnspacing=0.9, handlelength=1.6, framealpha=0.9, loc="upper right")
    plt.savefig(os.path.join(out_dir, "comparison_cost_loss.png"), dpi=300)
    plt.close()

    # --- cost vs test PSNR ---
    plt.figure(figsize=(7.4, 5.6))
    ymin = np.inf
    for m, (_, r) in sorted(best.items()):
        d = r["logs"]
        if "cost_test" in d and len(d.get("cost_test", [])) == len(d.get("psnr_test", [])):
            cost = np.asarray(d["cost_test"], dtype=float)
        else:
            cost = np.asarray(d.get("cost_epoch", []), dtype=float)
        psnr = np.asarray(d.get("psnr_test", []), dtype=float)
        n = min(len(cost), len(psnr))
        if n == 0:
            continue
        st = METHOD_STYLE.get(m, dict(label=m))
        plt.plot(cost[:n], psnr[:n], marker="o", ms=4, markevery=max(1, n // 12), **st)
        ymin = min(ymin, np.nanmax(psnr[:n]) - 9.0)
    plt.xscale("log")
    plt.xlabel(XLABEL)
    plt.ylabel("Average test PSNR")
    # Round 3: crop to the informative range (all runs start below 22 dB at cost < 10^2).
    plt.xlim(1e2, 2.1e4)
    plt.ylim(21.5, 29.8)
    plt.grid(True, alpha=0.3)
    # Round 3: legend shown once, in the loss panel (same colours/styles).
    plt.savefig(os.path.join(out_dir, "comparison_cost_psnr.png"), dpi=300)
    plt.close()

    # --- LaTeX summary rows ---
    print("% method & schedule/steps & best PSNR & SSIM at best")
    for m, (_, r) in sorted(best.items()):
        d = r["logs"]
        psnrs = np.asarray(d.get("psnr_test", []), dtype=float)
        ssims = np.asarray(d.get("ssim_test", []), dtype=float)
        if len(psnrs) == 0:
            continue
        i = int(np.nanargmax(psnrs))
        s = ssims[i] if i < len(ssims) else float("nan")
        label = METHOD_STYLE.get(m, dict(label=m))["label"]
        print(f"{label} & {r['file']} & {psnrs[i]:.2f} & {s:.4f} \\\\")
    print(f"figures written to {out_dir}")


def plot_tightening(log_dir, out_dir):
    """Two panels supporting Cor. (fixed horizon) and Remark (warm starts):
    (a) inner iterations per upper-level step, warm vs cold start;
    (b) test PSNR vs cost, horizon-tuned constant step vs anytime decay."""
    os.makedirs(out_dir, exist_ok=True)

    def get(pattern):
        hits = glob.glob(os.path.join(log_dir, pattern))
        if not hits:
            return None
        return torch.load(hits[0], map_location="cpu", weights_only=False, pickle_module=pickle)

    # --- (a) warm vs cold inner iterations ---
    plt.figure(figsize=(7.4, 5.6))
    for pat, lab, st in [
        ("logs_t1_ws_*.pt",   "Warm start", dict(color="tab:blue", ls="-")),
        ("logs_t1_cold_*.pt", "Cold start", dict(color="tab:red",  ls="--")),
    ]:
        d = get(pat)
        if d is None:
            continue
        li = np.asarray(d["lower_batch"], dtype=float)
        w = max(5, len(li) // 40)
        ma = moving_average(li, w)
        plt.plot(np.arange(len(ma)) + w, ma, label=lab, **st)
    plt.xlabel("Upper-level iteration $k$")
    plt.ylabel("Lower-level iterations per step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "tightening_warmstart.png"), dpi=300)
    plt.close()

    # --- (b) constant vs decaying step size ---
    plt.figure(figsize=(7.4, 5.6))
    series = [
        ("logs_t1const_1e-3_*.pt", r"const $\alpha=10^{-3}$", dict(color="tab:blue",   ls="-")),
        ("logs_t1const_3e-3_*.pt", r"const $\alpha=3\cdot10^{-3}$", dict(color="tab:green", ls="-.")),
        ("logs_t1const_1e-2_*.pt", r"const $\alpha=10^{-2}$", dict(color="tab:orange", ls=":")),
        ("logs_t1decay_*.pt",      r"decay $\alpha_k\propto k^{-0.55}$", dict(color="tab:red", ls="--")),
    ]
    for pat, lab, st in series:
        d = get(pat)
        if d is None:
            continue
        cost = np.asarray(d.get("cost_epoch", []), dtype=float)
        psnr = np.asarray(d.get("psnr_test", []), dtype=float)
        n = min(len(cost), len(psnr))
        if n == 0:
            continue
        plt.plot(cost[:n], psnr[:n], label=lab, **st)
    plt.xscale("log")
    plt.xlabel(XLABEL)
    plt.ylabel("Average test PSNR")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_dir, "tightening_stepsize.png"), dpi=300)
    plt.close()
    print(f"tightening figures written to {out_dir}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "rank"
    log_dir = sys.argv[2] if len(sys.argv) > 2 else "logs/TCI_comparison/tune"
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "figures"
    if mode == "rank":
        rank(log_dir)
    elif mode == "tightening":
        plot_tightening(log_dir, out_dir)
    else:
        plot_comparison(log_dir, out_dir)
