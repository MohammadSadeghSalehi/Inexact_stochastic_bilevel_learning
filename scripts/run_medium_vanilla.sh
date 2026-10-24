#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
LOGDIR=logs/TCI_comparison/medium_vanilla
COMMON="--no-auto-lipschitz --budget 20000 --log-dir $LOGDIR"
# Winners of the faithful (published-schedule) tuning sweep, 2026-08-12
"$PY" run_comparison.py --method soba   --alpha 1e-3 --beta 1e-2 $COMMON --tag mv
"$PY" run_comparison.py --method soba   --alpha 1e-2 --beta 1e-1 --alpha-decay 0.5 --beta-decay 0.5 $COMMON --tag mvdec
"$PY" run_comparison.py --method saba   --alpha 3e-4 --beta 3e-3 $COMMON --tag mv
"$PY" run_comparison.py --method masoba --alpha 1e-3 --beta 1e-2 $COMMON --tag mv
"$PY" run_comparison.py --method ttsa   --alpha 3e-3 --beta 1e-1 --alpha-decay 0.6 --beta-decay 0.4 $COMMON --tag mv
cp logs/TCI_comparison/medium/logs_m_isgd_*.pt logs/TCI_comparison/medium/logs_mq_isgd_*.pt logs/TCI_comparison/medium/logs_m_iadam_*.pt "$LOGDIR"/ 2>/dev/null
echo "vanilla medium done"
