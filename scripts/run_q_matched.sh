#!/bin/bash
# Match the decaying-step ISGD variant to the winning fixed-step setting
# (eps0 = 1e-1, p = 1, warm starts) so Fig. 4 compares like with like.
# Two alpha0 values: the fixed-step winner (1e-3) and a larger one (1e-2),
# since a decaying schedule can afford a larger initial step.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
LOGDIR=logs/TCI_comparison/eps0
B=20000
"$PY" run_comparison.py --method isgd --alpha0 1e-2 --eps0 1e-1 --p 1 --q 0.5 --warm-start --budget $B --log-dir $LOGDIR --tag e1mq
"$PY" run_comparison.py --method isgd --alpha0 1e-3 --eps0 1e-1 --p 1 --q 0.5 --warm-start --budget $B --log-dir $LOGDIR --tag e1mq
echo "q-matched runs done"
