#!/bin/bash 
#SBATCH -J cellpose_wholecell_train
#SBATCH -A sada-cnmi
#SBATCH -p tier3
#SBATCH --time=72:0:0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=200G 
#SBATCH --gres=gpu:a100:1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

cd cellpose
nvidia-smi
python -m cellpose --train --use_gpu --dir your wholecell training data path --test_dir your wholecell validation data path --pretrained_model None --all_channels --diameter 23. --save_png --img_filter _img --mask_filter _masks --verbose --save_each