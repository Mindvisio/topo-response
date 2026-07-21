#!/bin/bash
cd ~/topores
SEED=0; EP=40
echo "MATRIX START $(date)" > ~/mx_progress.log
for split in topology_ood grouprandom; do
  for cond in none tda shuffled random elem4d; do
    tag="${split}_${cond}_s${SEED}"
    echo "[$(date +%H:%M)] dipole $tag" >> ~/mx_progress.log
    ~/env/bin/python train_p3.py --split $split --cond $cond --seed $SEED --epochs $EP > ~/mx_dip_${tag}.log 2>&1
    CK=$(ls ckpt_${tag}/best-*.ckpt 2>/dev/null | head -1)
    ~/env/bin/python eval_ood.py --ckpt "$CK" --cond $cond --split $split >> ~/mx_dip_${tag}.log 2>&1
    ptag="polar_${split}_${cond}_s${SEED}"
    echo "[$(date +%H:%M)] polar $ptag" >> ~/mx_progress.log
    ~/env/bin/python train_p3_polar.py --split $split --cond $cond --seed $SEED --epochs $EP > ~/mx_${ptag}.log 2>&1
    CKP=$(ls ckpt_${ptag}/best-*.ckpt 2>/dev/null | head -1)
    ~/env/bin/python eval_ood_polar.py --ckpt "$CKP" --cond $cond --split $split >> ~/mx_${ptag}.log 2>&1
  done
done
echo "MATRIX DONE $(date)" >> ~/mx_progress.log
