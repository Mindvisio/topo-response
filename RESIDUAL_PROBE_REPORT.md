# Residual probe (bonus experiment)

A follow-up stress test of the main result, kept separate from it. The main study
found that a persistent-homology descriptor `z_PH` injected through a FiLM path did
not improve an E(3)-equivariant PaiNN on dipole or polarizability under topology
shift. That leaves one loophole: perhaps the signal is present in `z_PH` and the
FiLM path simply failed to use it. This experiment closes that loophole with a
strictly easier, more direct question.

## Question

Freeze each baseline (`cond=none`) checkpoint. On every molecule it makes an error
`r = y_true - y_base`. Fit a linear map (Ridge) from the invariant descriptor
`z_PH` to the coefficients of an **equivariance-preserving correction**, and ask
whether that correction reduces the held-out error — and whether it does so any
better than the same map fed a *random* descriptor of identical shape.

If `z_PH` carried usable residual structure, an oracle linear probe on 60k training
molecules should expose it, even where a jointly-trained FiLM gate did not. If even
this probe cannot beat a matched random control, the residual signal is absent, not
merely unused.

## Design

The correction never regresses Cartesian components from the invariant descriptor —
that would break equivariance. The probe predicts **invariant coefficients**; the
correction is assembled from **equivariant tensors** built from the geometry, so it
transforms correctly by construction (verified numerically to float64 round-off in
`test_probe_equivariance.py`, over 200 random rotations and reflections, before any
fitting).

Two bases per property, both declared before touching the test set:

- **primary** — the minimal form.
  Dipole: `mu_corr = mu_base + a*unit(mu_base)`.
  Polar: `A_corr = A_base + a*I + b*Q`, with `Q` the deviatoric part of `A_base`.
- **secondary** — augmented with the gyration tensor `S` of the centred coordinates.
  Dipole: `mu_corr = (1+a)*mu_base + b*(S mu_base) + c*(S^2 mu_base)`.
  Polar: adds `c*S + d*(S Q + Q S)/2`.

The primary dipole correction is collinear with `mu_base`, so it can only rescale
the vector, never rotate it; the secondary basis exists precisely so the angular
error has a way to improve if `z_PH` knows anything about direction.

Controls, all with identical preprocessing and dimensionality: `null` (no
correction), `random` (Gaussian matched to the train mean/std of `z_PH`, 5
realizations), `shuffled` (`z_PH` permuted across molecules, 5 realizations).
Standardisation is fit on train only; the Ridge `alpha` is chosen on validation
from a `1e-6 .. 1e6` grid and refit on train+val; the test set is scored once.
Molecules with `|mu_base| < 0.1 D` (dipole) or a near-isotropic `A_base` (polar)
have the correction defined to zero and are reported as a fraction, not silently
mixed in.

## Decision rule (fixed in advance)

Two independent readouts, not to be blurred:

1. **Probe R^2** on the 66,485-molecule test set — decisive at a single seed. Does
   `z_PH` predict the correction coefficients better than the random control?
2. **Physical-metric delta** (compMAE for dipole, Frobenius for polar), paired
   across the n=5 baseline seeds — the same n as the main study, and it inherits the
   same width. Does the correction actually lower the error, and by more than the
   random control?

The descriptor is judged informative only if the probe R^2 clears the random
control *and* the physical delta favours TDA over random with a Holm-corrected
p < 0.05 over the six primary-basis `tda`-vs-control tests. A non-significant
difference is reported as "no effect detected", never as proven equivalence.

## Results

Probe R^2 on held-out coefficients (mean over seeds):

| basis | property | tda | random | shuffled |
| --- | --- | --- | --- | --- |
| primary | dipole | -0.017 | -0.015 | -0.016 |
| primary | polar | -0.147 | -0.145 | -0.146 |
| secondary | dipole | -0.000 | -0.000 | -0.000 |
| secondary | polar | -0.000 | -0.000 | -0.000 |

Every R^2 is at or below zero — the probe predicts the coefficients no better than
their mean — and TDA never separates from the matched random control. In the
secondary basis the chosen `alpha` drives the correction to zero (nothing to fit),
so R^2 collapses to 0.

Physical-metric paired differences (n=5, topology-OOD; positive = correction made
the error worse):

| basis | property | delta_tda (mean +/- sd) | tda vs random | tda vs shuffled |
| --- | --- | --- | --- | --- |
| primary | dipole | +0.0002 +/- 0.0002 | +0.0000 p=0.85 | -0.0000 p=0.93 |
| primary | polar | +0.031 +/- 0.032 | +0.004 p=0.80 | +0.003 p=0.84 |
| secondary | dipole | +0.0015 +/- 0.0030 | +0.0005 p=0.75 | -0.0008 p=0.63 |
| secondary | polar | +0.0095 +/- 0.0157 | +0.0006 p=0.90 | -0.0038 p=0.41 |

After the Holm correction over the six primary-basis tests (threshold 0.0083 for
the smallest), no `tda`-vs-control test is significant — every adjusted p is 1.00.
The only moderately small raw p-values are `tda`-vs-`null` (dipole 0.069, polar
0.093), and their sign is **positive**: the correction slightly *worsens* the
baseline rather than helping it. A sensitivity refit that uses only validation
residuals (no in-sample train residuals) gives the same sign, so the direction is
not an artefact of optimistic training residuals. Fraction of molecules with the
correction defined to zero: 0.0017 (dipole), 0.043 (polar).

## Conclusion

An oracle linear probe from `z_PH` into the coefficients of an
equivariance-preserving correction does not reduce the frozen baseline's held-out
error, and does not predict the correction coefficients any better than a random
descriptor of the same shape — for either property, in either basis, on the
topology-OOD split. This strengthens the main negative: the earlier result showed
that *one* FiLM conditioning path did not benefit from `z_PH`; the probe shows that
the residual the baseline leaves is not linearly recoverable from `z_PH` at all,
beyond what a matched random feature achieves.

The usual limits apply and are not superseded by this experiment. "No effect
detected" is not proven equivalence: the coefficient targets are the linear
projection of the residual onto a fixed low-dimensional tensor basis, so a
non-linear probe, a richer equivariant basis, a different descriptor, or a
different dataset or split could still behave differently. The finding is a
qualitative negative about this descriptor and these corrections, consistent with
and reinforcing the main result.

## Reproduce

```
bash run_residual_probe.sh
```

Stage 1 (`export_baseline_predictions.py`, GPU, minutes) writes the frozen baseline
predictions to `probe_cache/`; stages 2–4 are CPU-only. `test_probe_equivariance.py`
must pass before `residual_probe.py` runs. Per-seed numbers land in
`results/residual_probe_per_seed.csv`, the aggregate in
`results/residual_probe_summary.json`.
