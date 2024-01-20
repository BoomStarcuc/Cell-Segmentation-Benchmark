#!/bin/bash

# List of tissue names
tissues=("Colon" "lymph_node_metastasis" "Spleen" "Pancreas" "Epidermis" "Breast" "Lymph_Node" "Tonsil" "Lung" "Esophagus")
CONFIG_DIR='mmdetection/configs/swin2C'
# Loop over each tissue name
for tissue in ${tissues[@]}
do
    # Copy A.py to a new file, replacing 'Epidermis' with the tissue name
    cp $CONFIG_DIR/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_tissuenet_n_one_for_all.py $CONFIG_DIR/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_tissuenet_n_one_for_all_${tissue}.py 
    sed -i "s/Breast/${tissue}/g" $CONFIG_DIR/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_tissuenet_n_one_for_all_${tissue}.py 
done

dir=$CONFIG_DIR/tissuenet_swin-s_n_one_for_all_slurm
out_dir='/shared/rc/spl/mmdet_output/One_to_all/nuclear/swin-s'
if [[ ! -e $dir ]]; then
    mkdir $dir
elif [[ ! -d $dir ]]; then
    echo "$dir already exists but is not a directory" 1>&2
fi
for tissue in ${tissues[@]}
do
    echo "CONFIG_DIR: $CONFIG_DIR"
    echo "OUTPUT_DIR: '$out_dir'/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_tissuenet_n_one_for_all_${tissue}"
    echo '#!/bin/bash' > ${dir}/swin-s-tissuenet-n-${tissue}.slurm

    echo '#SBATCH -J swin-s-tissuenet-n-on-'${tissue} >> ${dir}/swin-s-tissuenet-n-${tissue}.slurm
    echo '#SBATCH -A sada-cnmi' >> ${dir}/swin-s-tissuenet-n-${tissue}.slurm
    echo '#SBATCH -p tier3' >> ${dir}/swin-s-tissuenet-n-${tissue}.slurm
    echo '#SBATCH --time=72:0:0' >> ${dir}/swin-s-tissuenet-n-${tissue}.slurm
    echo '#SBATCH --output=%x_%j.out' >> ${dir}/swin-s-tissuenet-n-${tissue}.slurm
    echo '#SBATCH --error=%x_%j.err' >> ${dir}/swin-s-tissuenet-n-${tissue}.slurm
    echo '#SBATCH --mem=200G' >> ${dir}/swin-s-tissuenet-n-${tissue}.slurm
    echo '#SBATCH --gres=gpu:a100:4' >> ${dir}/swin-s-tissuenet-n-${tissue}.slurm
    echo 'spack load cuda@11.0.2%gcc@9.3.0/lrd2rcw'  >> ${dir}/swin-s-tissuenet-n-${tissue}.slurm
    echo 'cd mmdetection'  >> ${dir}/swin-s-tissuenet-n-${tissue}.slurm
    echo 'nvidia-smi'  >> ${dir}/swin-s-tissuenet-n-${tissue}.slurm
    echo 'sh mmdetection/tools/dist_train.sh '$CONFIG_DIR'/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_tissuenet_n_one_for_all_'${tissue}'.py 4 --work-dir '$out_dir'/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_tissuenet_n_one_for_all_'${tissue} >> ${dir}/swin-s-tissuenet-n-${tissue}.slurm
    sbatch ${dir}/swin-s-tissuenet-n-${tissue}.slurm
    
done





# 服务器地址/shared/rc/spl