#!/usr/bin/env bash
# Bonus experiment: freeze the baseline, ask whether z_PH linearly predicts the
# part of its residual an equivariance-preserving correction may touch.
#
# Stage 1 (GPU, minutes) exported baseline predictions; stages 2-4 are CPU-only.
# Prerequisite: probe_cache/<prop>_topology_ood_s<seed>.{npz,json} for seeds 0-4,
# produced by export_baseline_predictions.py against the cond=none checkpoints.
set -euo pipefail
PY=${PY:-python3}

# 1. equivariance of every correction basis, BEFORE any fitting
$PY test_probe_equivariance.py

# 2. fit the probes + score corrected predictions (per seed, both bases, all controls)
$PY residual_probe.py

# 3. paired n=5 statistics with a Holm correction over the declared family
$PY analyze_residual_probe.py
