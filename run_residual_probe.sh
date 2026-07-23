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

# 0b. refuse to continue on an incomplete or inconsistent input set: every export
#     needs BOTH its .npz and its .json sidecar, and the checkpoint each sidecar
#     names must still hash to the value recorded when it was exported.
CACHE="$CACHE" SEEDS="$SEEDS" PROPS="$PROPS" $PY - <<'PYCHECK'
import hashlib, json, os, sys
cache = os.environ['CACHE']
seeds = os.environ['SEEDS'].split()
props = os.environ['PROPS'].split()


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


bad = []
for prop in props:
    for s in seeds:
        stem = '%s/%s_topology_ood_s%s' % (cache, prop, s)
        npz, js = stem + '.npz', stem + '.json'
        for p in (npz, js):
            if not (os.path.exists(p) and os.path.getsize(p) > 0):
                bad.append('missing or empty: ' + p)
        if not os.path.exists(js):
            continue
        m = json.load(open(js))
        ck = m.get('checkpoint')
        if not ck or not os.path.exists(ck):
            bad.append('checkpoint not found for %s: %s' % (js, ck))
        elif sha256(ck) != m.get('checkpoint_sha256'):
            bad.append('checkpoint hash changed since export: ' + ck)
if bad:
    print('INPUT CHECK FAILED')
    for b in bad:
        print('  ' + b)
    sys.exit(1)
print('inputs complete and consistent: %d exports, checkpoint hashes match'
      % (len(props) * len(seeds)))
PYCHECK

# 1. equivariance of every correction basis, BEFORE any fitting
$PY test_probe_equivariance.py

# 2. fit the probes + score corrected predictions (per seed, both bases, all controls)
$PY residual_probe.py --cache "$CACHE" --props $PROPS --seeds $SEEDS

# 3. paired n=5 statistics with a Holm correction over the declared family
$PY analyze_residual_probe.py --label 'Ridge probe'

# 4. optional nonlinear probe: same question, MLP instead of Ridge, primary basis
#    only.  Its six tests are a SEPARATE pre-declared family, not an extension of
#    the Ridge family, so it is analyzed and reported separately.
if [ "${RUN_MLP:-0}" = "1" ]; then
  $PY residual_probe_mlp.py --cache "$CACHE" --props $PROPS --seeds $SEEDS
  $PY analyze_residual_probe.py \
    --csv results/residual_probe_mlp_per_seed.csv \
    --out results/residual_probe_mlp_summary.json \
    --rows-per-seed tda=3 random=15 shuffled=15 \
    --label 'MLP probe'
fi
