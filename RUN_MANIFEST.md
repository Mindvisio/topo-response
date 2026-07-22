# TopoResponse -- Run Manifest (course task 1)

Question: does a persistent-homology (TDA) descriptor injected via FiLM conditioning improve an E(3)-equivariant PaiNN at predicting molecular dipole vector and polarizability tensor?

## Environment
See requirements.txt (torch 2.4.1+cu121, schnetpack 2.1.1, e3nn 0.6.0, giotto-tda 0.6.0, rdkit 2026.3.4, pytorch-lightning 2.6.5). Single V100 32GB.

## Data (SQuIRL / QM9, 133883 molecules)
- Dipole stored in ATOMIC UNITS (h5 metadata mislabels Debye); converted a.u.->Debye x2.541746 at eval.
- SchNetPack ASE db + cached neighbor lists (cutoff 5.0 A).
- Splits: topology-OOD (train <=1 ring 60659 / test >=2 rings 66485); group-random by canonical SMILES (107110/13387/13386, 0 cross-partition graph duplicates).
- z_PH: geometric Vietoris-Rips H0/H1 on a FIXED [0,1] filtration grid -> Betti 64x2 + persistence entropy = 130-dim. md5 8f264761bb3fdfc4d498ddbb08b8a874
- zph_elem4d: element-augmented 4D persistence (atomic number as the fourth Vietoris-Rips coordinate). md5 3ba65d84ce59056aafc5cbe0093e8564
- split_topology_ood.npz md5 6db0af72bb7e7b52927650ebabb45b0c ; split_grouprandom.npz md5 df7ebbd523fd9aacd2dc37ccbfd2d361

## Audit fixes (all verified)
1. Betti on fixed [0,1] grid (was per-molecule -> incomparable bins).
2. SubtractCenterOfGeometry in all train/eval -> polarizability head translation-invariant; full E(3) of BOTH heads verified (rot/refl/trans ~1e-6).
3. Identity-init FiLM (zero-init last layer -> TDA==baseline at init) + train-only standardization + clip[-10,10].
4. Group split (0 leakage); shuffled control permuted WITHIN split; corrected metrics (true vector MAE; tensor Frobenius/anisotropy/eigenvalue/principal-axis; a.u.->Debye; dipole angle only for |mu|>0.1 D).

## Runs
- Matrix (seed 0, 40 epochs): 5 conditions x 2 properties x 2 splits = 20 configs.
- Stage 4b/4c (seeds 1-4, 40 epochs): baseline/tda/random x 2 properties x topology-OOD = 24 configs. -> 5 seeds (0-4) for the core comparison.

## Result (5-seed, topology-OOD, 95% t-CI, 4 df; paired t-tests; raw in results_5seed.csv, recompute with compute_ci.py)
Paired differences:
- dipole tda-baseline +0.0013 [-0.0036,+0.0063] p=0.50 n.s.; tda-random +0.0016 [-0.0056,+0.0088] p=0.57 n.s.; random-baseline -0.0003 [-0.0055,+0.0049] p=0.88 n.s.
- polar tda-baseline +0.2256 [+0.069,+0.382] p=0.016; tda-random -0.056 [-0.343,+0.231] p=0.62 n.s.; random-baseline +0.2816 [-0.107,+0.670] p=0.114 n.s.

Statistical caveats (IMPORTANT):
- A non-significant tda-vs-random difference is NOT equivalence. The CIs are wide (dipole admits up to ~6% TDA benefit; polar up to ~16% of baseline). No equivalence/TOST margin was pre-specified or met, so "TDA carries no useful signal" / "equivalent to noise" is NOT established.
- random is NOT significantly worse than baseline for polar (p=0.114), so a "conditioning-path hurts" claim is unsupported. Only TDA is (nominally) worse than baseline.
- The polar tda-baseline effect (p=0.016) is only NOMINAL. No test family was pre-specified. The table above reports SIX paired comparisons (three per property), so Holm requires p < 0.05/6 = 0.0083 for the smallest p-value, and the effect does NOT survive. Restricting the family post hoc to the two primary tda-baseline tests would raise the threshold to 0.025 and flip that conclusion - which is precisely why the family is fixed here at all six reported tests rather than chosen after seeing the p-values. An exact n=5 sign-test for 5 positive diffs gives p=0.0625.

Verdict (careful, qualitative): On 5 seeds, NO advantage of geometric z_PH + FiLM over baseline PaiNN or the matched-capacity random control was DETECTED on topology-OOD. For dipole the effect is statistically indeterminate around zero. For polarizability TDA nominally worsens vs baseline, but neither an advantage nor an equivalence of TDA vs random is established. The tested hypothesis was NOT SUPPORTED in the studied configuration. This is a qualitative negative; it does NOT establish that PH is equivalent to noise, that PH carries no information, that the conditioning path provably hurts, that PaiNN is generally sufficient, or that the hypothesis is strictly rejected; it does not generalize beyond this descriptor / conditioning / dataset / OOD split.

## Equivariance check (e3_test.py)

TDACondition is identity-initialised, so a freshly built conditioned model emits exactly zero scale/shift/gate and an equivariance test on it never touches the conditioning path. The check therefore also runs on the trained checkpoints, and reports how strong the learned modulation is before testing. Checkpoints are loaded with strict=True, so a partial load fails instead of quietly scoring a half-initialised model.

| case | conditioning RMS scale/shift/gate | rotation | reflection | translation |
| --- | --- | --- | --- | --- |
| dipole, fresh identity init | 0.0000 / 0.0000 / 0.0000 (inert) | 1.2e-06 | 9.8e-07 | 1.1e-06 |
| dipole, trained baseline | n/a | 6.3e-07 | 7.7e-07 | 9.8e-07 |
| dipole, trained TDA | 0.2106 / 0.0892 / 0.1882 (active) | 9.7e-07 | 7.9e-07 | 1.3e-06 |
| polar, fresh identity init | 0.0000 / 0.0000 / 0.0000 (inert) | 2.6e-07 | 2.4e-07 | 2.4e-07 |
| polar, trained baseline | n/a | 1.6e-07 | 1.8e-07 | 2.2e-07 |
| polar, trained TDA | 0.2191 / 0.1345 / 0.2622 (active) | 1.3e-07 | 1.5e-07 | 1.3e-07 |

Relative errors are at float32 round-off, so both heads remain E(3)-equivariant under rotation, reflection and translation with the learned conditioning switched on -- not merely at initialisation. Batch: 24 molecules from the topology-OOD split, geometries centred before each forward pass.

## Checkpoints
44 configs x (best,last) on sci-node /root/topores/ckpt_*/ (zip-verified), naming ckpt[_polar]_<split>_<cond>_s<seed>. Git-excluded; re-derivable from this manifest + pinned env + seeds.
