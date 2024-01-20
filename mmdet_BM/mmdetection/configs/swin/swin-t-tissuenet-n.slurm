#!/bin/bash
#SBATCH -J swin-t-tissuenet-n-1C
#SBATCH -A sada-cnmi
#SBATCH -p tier3
#SBATCH --time=72:0:0
#SBATCH --error=%x_%j.err
#SBATCH --output=%x_%j.out
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:4

spack load cuda@11.0.2%gcc@9.3.0/lrd2rcw
cd mmdetection
nvidia-smi
sh mmdetection/tools/dist_train.sh mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-50e_coco_tissuenet_n.py 4 --work-dir /shared/rc/spl/mmdet_output/All_to_all/nuclear_oc/swin-t-p4-w7_fpn_ms-crop-50e_coco_tissuenet_n
