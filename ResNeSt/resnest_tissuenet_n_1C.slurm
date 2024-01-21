#!/bin/bash 
#SBATCH -J resnest_tissuenet_n_1C
#SBATCH -p scavenger-gpu
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=RTXA5000:4
#SBATCH --mem=200G
#SBATCH --exclusive
#SBATCH --time=120:0:0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

cd ResNeSt
nvidia-smi
python tools/train_net.py --config-file benchmark_config/tissuenet_nuclear.yaml --num-gpus 4