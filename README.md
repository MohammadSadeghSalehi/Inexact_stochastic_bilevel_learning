# Bilevel Learning via Strategically Inexact Stochastic Gradient Descent

Code for the paper

> M. S. Salehi, S. Mukherjee, L. Roberts, and M. J. Ehrhardt,
> *Bilevel Learning via Strategically Inexact Stochastic Gradient Descent*, preprint,
> [arXiv:2511.06774](https://arxiv.org/abs/2511.06774).

Inexact stochastic gradient descent (ISGD) and its Adam variant (IAdam) learn the parameters of a
variational regularizer by solving each lower-level problem to a prescribed accuracy
$\epsilon_k = \epsilon_0 k^{-p}$ and taking upper-level steps $\alpha_k = \alpha_0 k^{-q}$. The repository
reproduces the paper's experiments: accuracy and step size schedules for a convex ridge
regularizer (CRR) and an input-convex neural network (ICNN), and a budget-matched comparison with
the single-loop stochastic bilevel methods TTSA, SOBA, SABA and MA-SOBA.

The code of the earlier conference paper, *Bilevel Learning with Inexact Stochastic Gradients*
(SSVM 2025), is at [github.com/MohammadSadeghSalehi/ISGD](https://github.com/MohammadSadeghSalehi/ISGD).

## Installation

```bash
conda env create -f environment.yaml
conda activate isgd-tci
```

The BSDS500 images are downloaded automatically on first use. Every script picks the device
automatically (CUDA, Apple MPS, or CPU); pass `--device cpu` to force the CPU.

## Repository layout

| Path | Contents |
|---|---|
| `run_isgd.py` | ISGD / IAdam with polynomial accuracy and step size schedules (Sections IV-A and IV-C) |
| `run_comparison.py` | Budget-matched comparison of ISGD / IAdam with TTSA, SOBA, SABA, MA-SOBA (Section IV-D) |
| `training_methods/bilevel_training_ISGD.py` | ISGD / IAdam training loop: nmAPG lower-level solves, MINRES Hessian system, warm starts, cost meter |
| `training_methods/bilevel_training_singleloop.py` | The single-loop baselines, in their published form |
| `evaluation/nmAPG2.py`, `evaluation/minres.py` | Lower-level solver and Hessian-system solver |
| `priors/` | CRR (`wcrr.py`) and ICNN regularizers |
| `diagnostics_singleloop.py` | Correctness checks of the baselines (supplementary Appendix F) |
| `plot_comparison.py` | Figures and table of the comparison |
| `scripts/` | The run scripts behind the paper's comparison and figures |
| `logs/TCI_final/` | Logs of the final comparison and of the validation runs, to regenerate the figures |

## Reproducing the results

**Accuracy and step size schedules (Sections IV-A, IV-C).** One run per configuration, for example
the CRR with $\alpha_0 = 10^{-2}$, $\epsilon_0 = 10^{-2}$, $p = 0.5$ and a fixed step size, or the ICNN with IAdam:

```bash
python run_isgd.py --arch crr --alpha0 1e-2 --eps0 1e-2 --p 0.5 --q 0.0 --optimizer sgd --budget 1e5
python run_isgd.py --arch icnn --alpha0 1e-2 --eps0 1e-2 --p 0.25 --q 0.0 --optimizer adam --budget 1e5
```

**Comparison with single-loop methods (Section IV-D, Table II, Fig. 4).**

```bash
bash scripts/tune_vanilla.sh          # step size tuning of the baselines (long horizon)
bash scripts/run_medium_vanilla.sh    # baselines at the tuned steps, budget 2e4
bash scripts/run_eps0_sweep.sh        # ISGD / IAdam initial-accuracy sweep with warm starts
bash scripts/run_q_matched.sh         # matched fixed vs decaying step size
python plot_comparison.py plot logs/TCI_final/comparison figures   # Fig. 4 and Table II from the logs
```

**Validation of Corollary 3 and Remark 3 (supplementary Fig. 10).**

```bash
bash scripts/run_tightening.sh
python plot_comparison.py tightening logs/TCI_final/tightening figures
```

**Correctness checks of the baselines (supplementary Appendix F).**

```bash
python diagnostics_singleloop.py
```

Scripts use `python` from the active environment; set `PYTHON=/path/to/python` to override.
Results on MPS can differ from CUDA and CPU in the last digits.

## Citation

```bibtex
@misc{salehi2025bilevel,
  title  = {Bilevel Learning via Strategically Inexact Stochastic Gradient Descent},
  author = {Salehi, Mohammad Sadegh and Mukherjee, Subhadip and Roberts, Lindon and Ehrhardt, Matthias J.},
  year   = {2025},
  eprint = {2511.06774},
  archivePrefix = {arXiv},
}
```

Parts of the code, including the general structure for computing hypergradients, are adapted from
[LearnedRegularizers](https://github.com/johertrich/LearnedRegularizers) (Hertrich et al., 2025).
