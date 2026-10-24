#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
LOGDIR=logs/TCI_comparison/tune_vanilla
COMMON="--no-auto-lipschitz --budget 6000 --train-size 32 --test-size 4 --test-crop 96 --log-dir $LOGDIR"
# TTSA: published two-timescale decay a_k ~ k^-3/5, b_k ~ k^-2/5
for A in 1e-2 3e-3; do for B in 1e-1 3e-2 1e-2; do
  "$PY" run_comparison.py --method ttsa --alpha "$A" --beta "$B" --alpha-decay 0.6 --beta-decay 0.4 $COMMON --tag "v_${A}_${B}_d64" || true
done; done
# SOBA Thm 1: fixed coupled steps (horizon scaling absorbed into tuning)
for B in 1e-2 3e-3 1e-3; do for XI in 1 0.1; do
  A=$("$PY" -c "print($XI*$B)")
  "$PY" run_comparison.py --method soba --alpha "$A" --beta "$B" $COMMON --tag "v_${A}_${B}_fix" || true
done; done
# SOBA Thm 2: decreasing steps ~ t^-1/2
for B in 1e-1 3e-2 1e-2; do
  A=$("$PY" -c "print(0.1*$B)")
  "$PY" run_comparison.py --method soba --alpha "$A" --beta "$B" --alpha-decay 0.5 --beta-decay 0.5 $COMMON --tag "v_${A}_${B}_dec" || true
done
# SABA Thm 3: constant steps
for B in 3e-3 1e-3 3e-4; do for XI in 1 0.1; do
  A=$("$PY" -c "print($XI*$B)")
  "$PY" run_comparison.py --method saba --alpha "$A" --beta "$B" $COMMON --tag "v_${A}_${B}_fix" || true
done; done
# MA-SOBA: constant steps
for B in 1e-2 3e-3 1e-3; do for XI in 1 0.1; do
  A=$("$PY" -c "print($XI*$B)")
  "$PY" run_comparison.py --method masoba --alpha "$A" --beta "$B" $COMMON --tag "v_${A}_${B}_fix" || true
done; done
echo "vanilla tune done"
