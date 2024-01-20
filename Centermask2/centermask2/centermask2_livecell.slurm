#!/bin/bash 
#SBATCH -J centermask2_livecell_train
#SBATCH -A sada-cnmi
#SBATCH -p tier3
#SBATCH --time=72:0:0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=200G 
#SBATCH --gres=gpu:a100:4

cd centermask2
spack load cuda@11.0.2%gcc@9.3.0/lrd2rcw
nvidia-smi
python train_net.py --config-file centermask2/benchmark_config/livecell_train_all.yaml --num-gpus 4