#!/bin/bash
# Fig. 4 improvement: larger initial tolerance eps0 => cheaper early iterations
# => ISGD/IAdam curves start much further left on the cost axis.
# Also fixes a fairness issue: the single-loop baselines keep persistent (warm)
# lower-level iterates, so ISGD/IAdam must use --warm-start to match.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
LOGDIR=logs/TCI_comparison/eps0
B=20000
# large tolerance first (most informative for the early-cost region)
"$PY" run_comparison.py --method isgd  --alpha0 1e-3 --eps0 1    --p 2    --q 0 --warm-start --budget $B --log-dir $LOGDIR --tag e1
"$PY" run_comparison.py --method iadam --alpha0 1e-2 --eps0 1    --p 2    --q 0 --warm-start --budget $B --log-dir $LOGDIR --tag e1
"$PY" run_comparison.py --method isgd  --alpha0 1e-3 --eps0 1e-1 --p 1    --q 0 --warm-start --budget $B --log-dir $LOGDIR --tag e1m
"$PY" run_comparison.py --method iadam --alpha0 1e-2 --eps0 1e-1 --p 1    --q 0 --warm-start --budget $B --log-dir $LOGDIR --tag e1m
"$PY" run_comparison.py --method isgd  --alpha0 1e-3 --eps0 1e-2 --p 0.25 --q 0 --warm-start --budget $B --log-dir $LOGDIR --tag e2
"$PY" run_comparison.py --method iadam --alpha0 1e-2 --eps0 1e-2 --p 0.25 --q 0 --warm-start --budget $B --log-dir $LOGDIR --tag e2
echo "eps0 sweep done"
