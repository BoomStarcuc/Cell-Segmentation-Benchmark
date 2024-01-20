#!/bin/bash
#SBATCH -J mask-rcnn-tissuenet-n-2C
#SBATCH -A sada-cnmi
#SBATCH -p tier3
#SBATCH --time=120:0:0
#SBATCH --error=%x_%j.err
#SBATCH --output=%x_%j.out
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:4

spack load cuda@11.0.2%gcc@9.3.0/lrd2rcw
cd mmdetection
nvidia-smi
sh mmdetection/tools/dist_train.sh mmdetection/configs/strong_baselines2C/mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco_tissuenet_n.py 4 --work-dir /shared/rc/spl/mmdet_output/All_to_all/nuclear/MaskRCNN
