#!/usr/bin/env bash
# Bonus experiment: freeze the baseline, ask whether z_PH linearly predicts the
# part of its residual an equivariance-preserving correction may touch.
#
# Full cycle, including the GPU-side export, so the run does not depend on a
# pre-populated (git-excluded) probe_cache/.  Stage 1 needs the cond=none
# checkpoints and is the only part that benefits from a GPU; stages 2-4 are
# CPU-only and take a few minutes.
#
#   PY=/path/to/python bash run_residual_probe.sh          # full run
#   SKIP_EXPORT=1 bash run_residual_probe.sh               # reuse probe_cache/
set -euo pipefail
PY=${PY:-python3}
SEEDS=${SEEDS:-"0 1 2 3 4"}
PROPS=${PROPS:-"dipole polar"}
CACHE=${CACHE:-probe_cache}

echo "interpreter: $($PY -c 'import sys; print(sys.executable)')"
$PY -c 'import numpy, scipy, schnetpack, torch' \
  || { echo 'missing deps: need numpy, scipy, schnetpack, torch'; exit 1; }

# 0. freeze the baselines and export their predictions (skippable if cached)
if [ "${SKIP_EXPORT:-0}" != "1" ]; then
  for prop in $PROPS; do
    for s in $SEEDS; do
      $PY export_baseline_predictions.py --property "$prop" --seed "$s" --out "$CACHE"
    done
  done
fi

# 0b. refuse to continue on an incomplete input set
missing=0
for prop in $PROPS; do
  for s in $SEEDS; do
    f="$CACHE/${prop}_topology_ood_s${s}.npz"
    [ -s "$f" ] || { echo "MISSING $f"; missing=1; }
  done
done
[ "$missing" -eq 0 ] || { echo 'incomplete probe_cache: rerun without SKIP_EXPORT=1'; exit 1; }
echo "inputs complete: $(ls "$CACHE"/*.npz | wc -l) export files"

# 1. equivariance of every correction basis, BEFORE any fitting
$PY test_probe_equivariance.py

# 2. fit the probes + score corrected predictions (per seed, both bases, all controls)
$PY residual_probe.py

# 3. paired n=5 statistics with a Holm correction over the declared family
$PY analyze_residual_probe.py
