#!/bin/bash
# Supporting runs for the two theory tightenings (Remark on metric conventions,
# Corollary cor:fixed_horizon, Remark rem:warmstart in the paper).
# Takes several hours; run in the background:
#     nohup bash scripts/run_tightening.sh > logs/TCI_comparison/tightening.log 2>&1 &
#
# Point 1 (fixed-horizon constant step, Cor. cor:fixed_horizon):
#   ISGD with eps_k = k^{-1/4} (p=0.25) and CONSTANT alpha (q=0) playing the
#   role of alpha = c/sqrt(K); the alpha grid spans the constant c. Compared
#   against the anytime decaying schedule q=0.55 with the same p.
#   Expectation: the best constant-alpha run matches/beats q=0.55 at the fixed
#   budget, empirically illustrating that the delta-gap is an artifact of the
#   anytime analysis.
#
# Point 2 (warm-start amortization, Remark rem:warmstart):
#   Identical ISGD configs with and without persistent per-sample warm starts.
#   The log key `lower_batch` records inner iterations per outer step; the
#   remark predicts it settles to O(1) with warm starts while growing like
#   log(k) without. Analyze with:
#     python - << 'EOF'
#     import torch
#     for f in ["logs_t1_ws_isgd_ws_0.01_0.01_0.25_0.0_.pt", "logs_t1_cold_isgd_0.01_0.01_0.25_0.0_.pt"]:
#         d = torch.load(f"logs/TCI_comparison/tightening/{f}", map_location="cpu", weights_only=False)
#         lb = d["lower_batch"]; n = len(lb)
#         print(f, "mean inner iters: first-10%", sum(lb[:n//10])/(n//10),
#               "last-10%", sum(lb[-n//10:])/(n//10))
#     EOF

set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
LOGDIR=logs/TCI_comparison/tightening
BUDGET=20000

# --- Point 1: fixed-horizon constant step vs anytime decay ---
for A in 1e-2 3e-3 1e-3; do
  "$PY" run_comparison.py --method isgd --alpha0 "$A" --eps0 1e-2 --p 0.25 --q 0 \
    --budget $BUDGET --log-dir "$LOGDIR" --tag "t1const_${A}"
done
"$PY" run_comparison.py --method isgd --alpha0 1e-2 --eps0 1e-2 --p 0.25 --q 0.55 \
  --budget $BUDGET --log-dir "$LOGDIR" --tag t1decay

# --- Point 2: warm-start amortization (same config, warm vs cold) ---
"$PY" run_comparison.py --method isgd --alpha0 1e-2 --eps0 1e-2 --p 0.25 --q 0 \
  --budget $BUDGET --log-dir "$LOGDIR" --tag t1_ws --warm-start
"$PY" run_comparison.py --method isgd --alpha0 1e-2 --eps0 1e-2 --p 0.25 --q 0 \
  --budget $BUDGET --log-dir "$LOGDIR" --tag t1_cold

echo "tightening runs done"
