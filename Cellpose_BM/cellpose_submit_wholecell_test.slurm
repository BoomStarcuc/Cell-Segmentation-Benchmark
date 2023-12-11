#!/bin/bash 
#SBATCH -J cellpose_w_oc_test
#SBATCH -A sada-cnmi
#SBATCH -p tier3
#SBATCH --time=72:0:0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=200G 
#SBATCH --gres=gpu:a100:1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

nvidia-smi
python -m cellpose --use_gpu --dir the path of your wholecell test data  --diameter 23. --save_png --img_filter _img --mask_filter _masks --pretrained_model the path of your wholecell pretrained model --verbose --all_channels --savedir the path of your save dir --excelname the name of your saved excel file --dataname TissueNet --nchan_test 1