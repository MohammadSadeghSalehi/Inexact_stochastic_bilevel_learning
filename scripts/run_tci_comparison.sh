#!/bin/bash
# Budget-matched comparison pipeline for the TCI revision.
#
# Usage:
#   bash scripts/run_tci_comparison.sh tune     # short step-size tuning sweeps (budget 2e3)
#   bash scripts/run_tci_comparison.sh medium   # medium-budget comparison (2e4) with tuned steps
#   bash scripts/run_tci_comparison.sh full     # full-budget (1e5) runs -- intended for a GPU machine
#
# All runs share the same fixed dataset (deterministic crops + fixed noise, seed 23).

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
MODE="${1:-tune}"

case "$MODE" in
  tune)
    BUDGET=2000
    LOGDIR=logs/TCI_comparison/tune
    for A in 1e-1 1e-2 1e-3; do
      for B in 5e-1 1e-1 1e-2; do
        for M in ttsa soba saba masoba; do
          echo "=== tune $M alpha=$A beta=$B ==="
          "$PY" run_comparison.py --method "$M" --alpha "$A" --beta "$B" \
            --budget "$BUDGET" --train-size 32 --test-size 4 --test-crop 96 \
            --log-dir "$LOGDIR" --tag "tune_${A}_${B}" || echo "FAILED: $M $A $B"
        done
      done
    done
    ;;
  medium|full)
    if [ "$MODE" = medium ]; then BUDGET=20000; else BUDGET=100000; fi
    LOGDIR=logs/TCI_comparison/$MODE
    # Step sizes = winners of the tune sweep (2026-08-11, fixed-code re-tune for saba/masoba):
    #   ttsa  (1e-2, 1e-2): loss 43,  PSNR 23.4      soba (1e-1, 1e-2): loss 230, PSNR 23.5
    #   saba  (1e-3, 1e-1): loss 113, PSNR 22.2      masoba (1e-2, 1e-2): most stable non-tiny alpha
    "$PY" run_comparison.py --method ttsa   --alpha 1e-2 --beta 1e-2 --budget "$BUDGET" --log-dir "$LOGDIR" --tag m
    "$PY" run_comparison.py --method soba   --alpha 1e-1 --beta 1e-2 --budget "$BUDGET" --log-dir "$LOGDIR" --tag m
    "$PY" run_comparison.py --method soba   --alpha 1e-3 --beta 1e-1 --budget "$BUDGET" --log-dir "$LOGDIR" --tag m2
    "$PY" run_comparison.py --method saba   --alpha 1e-3 --beta 1e-1 --budget "$BUDGET" --log-dir "$LOGDIR" --tag m
    "$PY" run_comparison.py --method masoba --alpha 1e-2 --beta 1e-2 --budget "$BUDGET" --log-dir "$LOGDIR" --tag m
    # scheduled references on the same fixed dataset
    "$PY" run_comparison.py --method isgd  --alpha0 1e-2 --eps0 1e-2 --p 0.5 --q 0   --budget "$BUDGET" --log-dir "$LOGDIR" --tag m
    "$PY" run_comparison.py --method isgd  --alpha0 1e-2 --eps0 1e-2 --p 0.5 --q 0.5 --budget "$BUDGET" --log-dir "$LOGDIR" --tag mq
    "$PY" run_comparison.py --method iadam --alpha0 1e-2 --eps0 1e-2 --p 0.25 --q 0  --budget "$BUDGET" --log-dir "$LOGDIR" --tag m
    ;;
  *)
    echo "unknown mode: $MODE" >&2; exit 1;;
esac
echo "done ($MODE)"
