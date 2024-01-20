#!/bin/bash

# List of tissue names
tissues=("Colon" "lymph_node_metastasis" "Spleen" "Pancreas" "Epidermis" "Breast" "Lymph_Node" "Tonsil" "Lung" "Esophagus")
CONFIG_DIR='mmdetection/configs/swin2C'
# Loop over each tissue name
for tissue in ${tissues[@]}
do
    # Copy A.py to a new file, replacing 'Epidermis' with the tissue name
    cp $CONFIG_DIR/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_tissuenet_n.py $CONFIG_DIR/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_tissuenet_n_train_all_test_on_${tissue}.py 
    sed -i "s/_all/_${tissue}/g" $CONFIG_DIR/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_tissuenet_n_train_all_test_on_${tissue}.py 
done

dir=$CONFIG_DIR/tissuenet_swin-s_n_all_to_all_test_slurm
out_dir='/shared/rc/spl/mmdet_output/All_to_all/nuclear/Swin-S'
if [[ ! -e $dir ]]; then
    mkdir $dir
elif [[ ! -d $dir ]]; then
    echo "$dir already exists but is not a directory" 1>&2
fi
for tissue in ${tissues[@]}
do
    echo "CONFIG_DIR: $CONFIG_DIR"
    echo "OUTPUT_DIR: '$out_dir/predictions/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_tissuenet_n_train_all_test_on_${tissue}"
    echo '#!/bin/bash' > ${dir}/swin-s-tissuenet-n-train-all-test-on-${tissue}_2C.slurm

    echo '#SBATCH -J swin-s-tissuenet-n-train-all-test-on-'${tissue} >> ${dir}/swin-s-tissuenet-n-train-all-test-on-${tissue}_2C.slurm
    echo '#SBATCH -A sada-cnmi' >> ${dir}/swin-s-tissuenet-n-train-all-test-on-${tissue}_2C.slurm
    echo '#SBATCH -p tier3' >> ${dir}/swin-s-tissuenet-n-train-all-test-on-${tissue}_2C.slurm
    echo '#SBATCH --time=24:0:0' >> ${dir}/swin-s-tissuenet-n-train-all-test-on-${tissue}_2C.slurm
    echo '#SBATCH --output=%x_%j.out' >> ${dir}/swin-s-tissuenet-n-train-all-test-on-${tissue}_2C.slurm
    echo '#SBATCH --error=%x_%j.err' >> ${dir}/swin-s-tissuenet-n-train-all-test-on-${tissue}_2C.slurm
    echo '#SBATCH --mem=200G' >> ${dir}/swin-s-tissuenet-n-train-all-test-on-${tissue}_2C.slurm
    echo '#SBATCH --gres=gpu:a100:2' >> ${dir}/swin-s-tissuenet-n-train-all-test-on-${tissue}_2C.slurm
    echo 'spack load cuda@11.0.2%gcc@9.3.0/lrd2rcw'  >> ${dir}/swin-s-tissuenet-n-train-all-test-on-${tissue}_2C.slurm
    echo 'cd mmdetection'  >> ${dir}/swin-s-tissuenet-n-train-all-test-on-${tissue}_2C.slurm
    echo 'nvidia-smi'  >> ${dir}/swin-s-tissuenet-n-train-all-test-on-${tissue}_2C.slurm
    echo 'bash mmdetection/tools/dist_test.sh '$CONFIG_DIR'/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_tissuenet_n_train_all_test_on_'${tissue}'.py '$out_dir'/latest.pth 2 --out '$out_dir'/predictions/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_tissuenet_n_train_all_test_on_'${tissue}'/results.pkl --work-dir '$out_dir'/predictions/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_tissuenet_n_train_all_test_on_'${tissue}' --eval segm' >> ${dir}/swin-s-tissuenet-n-train-all-test-on-${tissue}_2C.slurm
    
    sbatch ${dir}/swin-s-tissuenet-n-train-all-test-on-${tissue}_2C.slurm   
done