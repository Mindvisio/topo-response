#!/bin/bash
cd ~/topores
EP=40
echo "STAGE4B START $(date)" > ~/s4b_progress.log
for seed in 1 2; do
  for cond in none tda random; do
    tag="topology_ood_${cond}_s${seed}"
    echo "[$(date +%H:%M)] dipole $tag" >> ~/s4b_progress.log
    ~/env/bin/python train_p3.py --split topology_ood --cond $cond --seed $seed --epochs $EP > ~/s4b_dip_${tag}.log 2>&1
    CK=$(ls ckpt_${tag}/best-*.ckpt 2>/dev/null | head -1)
    ~/env/bin/python eval_ood.py --ckpt "$CK" --cond $cond --split topology_ood >> ~/s4b_dip_${tag}.log 2>&1
    ptag="polar_topology_ood_${cond}_s${seed}"
    echo "[$(date +%H:%M)] polar $ptag" >> ~/s4b_progress.log
    ~/env/bin/python train_p3_polar.py --split topology_ood --cond $cond --seed $seed --epochs $EP > ~/s4b_${ptag}.log 2>&1
    CKP=$(ls ckpt_${ptag}/best-*.ckpt 2>/dev/null | head -1)
    ~/env/bin/python eval_ood_polar.py --ckpt "$CKP" --cond $cond --split topology_ood >> ~/s4b_${ptag}.log 2>&1
  done
done
echo "STAGE4B DONE $(date)" >> ~/s4b_progress.log
