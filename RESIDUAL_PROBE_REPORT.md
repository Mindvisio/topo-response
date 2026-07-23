# Residual probe (bonus experiment)

A follow-up stress test of the main result, kept separate from it. The main study
found that a persistent-homology descriptor `z_PH` injected through a FiLM path did
not improve an E(3)-equivariant PaiNN on dipole or polarizability under topology
shift. That leaves an obvious question: is the signal present in `z_PH` but unused
by that particular conditioning path? This experiment probes that question
directly, without claiming to settle it in general.

## Question

Freeze each baseline (`cond=none`) checkpoint. On every molecule it makes an error
`r = y_true - y_base`. Fit a linear map (Ridge) from the invariant descriptor
`z_PH` to the coefficients of an **equivariance-preserving correction**, and ask
whether that correction reduces the held-out error — and whether it does so any
better than the same map fed a *random* descriptor of identical shape.

If `z_PH` carried linearly usable residual structure, an oracle probe fitted on
60k training molecules should expose it, even where a jointly-trained FiLM gate did
not. A null result here does not prove the descriptor is uninformative; it narrows
where any remaining signal could hide.

## Design

The correction never regresses Cartesian components from the invariant descriptor —
that would break equivariance. The probe predicts **invariant coefficients**; the
correction is assembled from **equivariant tensors** built from the geometry, so it
transforms correctly by construction (verified numerically to float64 round-off in
`test_probe_equivariance.py`, over 200 random rotations and reflections, before any
fitting).

Two bases per property, both declared before touching the test set:

- **primary** — the minimal form, and the basis the conclusions rest on.
  Dipole: `mu_corr = mu_base + a*unit(mu_base)`.
  Polar: `A_corr = A_base + a*I + b*Q`, with `Q` the deviatoric part of `A_base`.
- **secondary (exploratory)** — augmented with the gyration tensor `S` of the
  centred coordinates. Dipole: `mu_corr = (1+a)*mu_base + b*(S mu_base) +
  c*(S^2 mu_base)`. Polar: adds `c*S + d*(S Q + Q S)/2`.

The primary dipole correction is collinear with `mu_base`, so it can only rescale
the vector, never rotate it; the secondary basis exists so the angular error has a
way to improve if `z_PH` knows anything about direction. The secondary bases are
reported as **exploratory only**, because their per-molecule Gram matrices are
badly conditioned: median `cond(G)` is 3.5e4 for the dipole basis with 20% of
molecules above 1e6, and 5.5e4 for the polar basis with 8% above 1e6. The primary
bases are well conditioned (1 and 94 respectively).

Controls, all with identical preprocessing and dimensionality: `null` (no
correction), `random` (Gaussian matched to the train mean/std of `z_PH`, 5
realizations), `shuffled` (`z_PH` permuted across molecules, 5 realizations).
Descriptors and coefficient targets are standardised on train only; the Ridge
`alpha` is chosen on validation from a `1e-6 .. 1e6` grid and refit on train+val;
each fitted model is evaluated once on test, and test results are not used for
fitting, alpha selection or model selection. The intercept is left **unpenalised**, so a large
`alpha` reduces the probe to an intercept-only fit rather than shrinking the fitted
mean toward zero. Molecules with `|mu_base| < 0.1 D` (dipole,
0.70% of test) or a near-isotropic `A_base` (polar, 4.3%) have the correction
defined to zero and are reported rather than silently mixed in.

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
| primary | dipole | -0.0173 | -0.0169 | -0.0168 |
| primary | polar | -0.1500 | -0.1489 | -0.1487 |
| secondary (exploratory) | dipole | -0.0001 | -0.0000 | -0.0000 |
| secondary (exploratory) | polar | -0.0002 | -0.0002 | -0.0002 |

R^2 here is measured against a **test-mean reference** (`ss_tot` uses the mean of
the test coefficients themselves), so `R^2 <= 0` states that the probe has no
positive out-of-sample explanatory power relative to that reference — a stronger
and more precise statement than any claim about a training mean. Every value is at
or below zero, and TDA never separates from the matched random control. In the
secondary bases the selected `alpha` drives the fitted slope to zero, so R^2
collapses to the intercept-only value.

Physical-metric paired differences (n=5, topology-OOD; positive = correction made
the error worse):

| basis | property | delta_tda (mean +/- sd) | tda vs random | tda vs shuffled |
| --- | --- | --- | --- | --- |
| primary | dipole | +0.0002 +/- 0.0002 | -0.0000 p=0.49 | -0.0000 p=0.51 |
| primary | polar | +0.033 +/- 0.033 | +0.003 p=0.84 | +0.003 p=0.83 |
| secondary | dipole | +0.0015 +/- 0.0028 | +0.0004 p=0.75 | -0.0006 p=0.64 |
| secondary | polar | +0.037 +/- 0.040 | -0.0005 p=0.89 | -0.0018 p=0.59 |

Holm-adjusted p-values over the declared family of six primary-basis tests
(threshold 0.0083 for the smallest raw p):

| test | raw p | Holm-adjusted p |
| --- | --- | --- |
| dipole tda vs null | 0.078 | 0.465 |
| dipole tda vs random | 0.490 | 1.000 |
| dipole tda vs shuffled | 0.511 | 1.000 |
| polar tda vs null | 0.094 | 0.471 |
| polar tda vs random | 0.836 | 1.000 |
| polar tda vs shuffled | 0.829 | 1.000 |

No test is significant. The two smallest raw p-values are `tda`-vs-`null`, and
their sign is **positive**: the fitted correction slightly *worsens* the baseline
rather than helping it. A sensitivity refit that uses only validation residuals (no
in-sample train residuals, same estimator) gives the same sign and a comparable size
-- +0.000141 D for the dipole and +0.0281 a.u. for the polarizability against the
main-path +0.000169 and +0.0327 -- so the direction is not an artefact of optimistic
training residuals.

## Nonlinear probe (MLP)

A linear probe finding nothing leaves one reading open: that the relation exists
but is nonlinear and Ridge could not see it. To close that reading as far as a
small network can, the same experiment was repeated with an MLP
(130 -> 64 -> 32 -> k, ReLU) in place of Ridge, on the well-conditioned **primary
basis only**. Everything else is identical -- train-only standardisation of
descriptors and coefficient targets, the same architecture, optimiser, schedule
and early-stopping rule for every arm. Each fitted model is evaluated once on
test, and test results are used for neither training, early stopping nor model
selection. Each of the
5 baseline seeds gets 3 weight initialisations, and the random and shuffled
controls additionally get 5 descriptor realizations; initialisations and
realizations are averaged **within** a baseline seed, so the unit of replication
is still the seed (n=5). Early stopping uses the validation split only.

This is a **pre-specified follow-up sensitivity analysis**, analysed as its own
family of six tests rather than as an extension of the Ridge family above, so it
does not retroactively change that family's multiplicity or revise its conclusions.
Its specification was fixed before this probe was run, but after the Ridge results
on the same test set had been seen — so it is a follow-up analysis, not an
independent confirmatory experiment.

Probe R^2 (test-mean reference):

| property | tda (mean) | tda (median) | random | shuffled |
| --- | --- | --- | --- | --- |
| dipole | -0.091 | -0.040 | -0.017 | -0.016 |
| polar | -0.141 | -0.153 | -0.148 | -0.148 |

Medians are over the five seed-level means, matching the unit of replication (the
fit-level median over all 15 individual dipole fits is -0.042, a diagnostic rather
than a summary). The dipole `tda` mean is pulled down by one seed at R^2 = -0.27,
whose worst single fit reaches -0.68, so the median is the more representative
summary.

Physical-metric paired differences (n=5; positive = correction made it worse):

| property | delta_tda (mean +/- sd) | tda vs random | tda vs shuffled |
| --- | --- | --- | --- |
| dipole | +0.0016 +/- 0.0016 | +0.0014 p=0.118 | +0.0014 p=0.113 |
| polar | +0.0283 +/- 0.0188 | -0.0012 p=0.951 | -0.0003 p=0.987 |

Holm over this family: the smallest adjusted p is 0.168 (polar `tda` vs `null`,
and its sign is positive, i.e. the correction hurts); nothing is significant.

All mean R^2 values are non-positive, so the nonlinear probe likewise has no
positive out-of-sample explanatory power. Individual values are not uniformly
ordered — for the polarizability `tda` sits numerically above its controls
(-0.141 against -0.148), for the dipole below them — but no beneficial
physical-metric advantage over the controls was detected for either property. One observation is worth recording without over-reading
it: on the real descriptor the networks run markedly longer before early stopping
-- median 50 epochs for the dipole against 16-19 on random and shuffled -- yet
generalise no better, and for the dipole slightly worse. That is a difference in
optimisation and early-stopping dynamics; it is not by itself evidence that the
network found transferable structure, and no such claim is made here.

## Conclusion

Neither a linear nor a small nonlinear equivariant probe detected additional
predictive signal from `z_PH` beyond matched random and shuffled controls: neither
reduces the frozen baseline's held-out error, and neither predicts the correction
coefficients better than a random descriptor of the same shape — for either
property, on the topology-OOD split. This is consistent with the
main 5-seed negative and narrows the space in which a usable residual signal could
still be hiding.

What this does **not** show. "No effect detected" is not proven equivalence. The
coefficient targets are the projection of the residual onto a fixed
low-dimensional tensor basis, so a richer or better-conditioned equivariant basis, a
larger or differently regularised network, a different descriptor, or a different
dataset or split could still behave differently; two probes agreeing is evidence, not
proof. The secondary bases are exploratory and ill-conditioned, so
they support no independent claim. The result is a qualitative negative about this
descriptor and these corrections.

## Reproduce

```
PY=/path/to/python RUN_MLP=1 bash run_residual_probe.sh
```

The harness runs the whole cycle: it exports the frozen baseline predictions for
both properties and all five seeds (the only GPU-assisted stage, and the only one
that needs the checkpoints), refuses to continue unless all ten export files are
present, runs the equivariance gate, fits the probes, and computes the statistics.
`RUN_MLP=1` adds the nonlinear probe (`residual_probe_mlp.py`, about 20 minutes on
CPU) with its separate analysis; drop it to run the Ridge probe alone. Set
`SKIP_EXPORT=1` to reuse an existing `probe_cache/`. A transcript of the most
recent equivariance gate is kept at `results/e3_gate.log`. Per-seed numbers land in
`results/residual_probe_per_seed.csv` (including the per-basis conditioning
diagnostics), the aggregate in `results/residual_probe_summary.json`.
