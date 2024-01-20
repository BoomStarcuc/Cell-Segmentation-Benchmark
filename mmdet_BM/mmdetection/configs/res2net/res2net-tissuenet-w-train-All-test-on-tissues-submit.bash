#!/bin/bash

# List of tissue names
tissues=("Colon" "lymph_node_metastasis" "Spleen" "Pancreas" "Epidermis" "Breast" "Lymph_Node" "Tonsil" "Lung" "Esophagus")
CONFIG_DIR='mmdetection/configs/res2net'
# Loop over each tissue name
for tissue in ${tissues[@]}
do
    # Copy A.py to a new file, replacing 'Epidermis' with the tissue name
    cp $CONFIG_DIR/cascade_mask_rcnn_r2_101_fpn_20e_coco_tissuenet_w.py $CONFIG_DIR/cascade_mask_rcnn_r2_101_fpn_20e_coco_tissuenet_w_train_all_test_on_${tissue}.py 
    sed -i "s/_all/_${tissue}/g" $CONFIG_DIR/cascade_mask_rcnn_r2_101_fpn_20e_coco_tissuenet_w_train_all_test_on_${tissue}.py 
done

dir=$CONFIG_DIR/res2net_tissuenet_w_all_to_all_test_slurm
out_dir='/shared/rc/spl/mmdet_output/All_to_all/wholecell_oc/cascade_mask_rcnn_r2_101_fpn_20e_coco_tissuenet_w'
if [[ ! -e $dir ]]; then
    mkdir $dir
elif [[ ! -d $dir ]]; then
    echo "$dir already exists but is not a directory" 1>&2
fi
for tissue in ${tissues[@]}
do
    echo "CONFIG_DIR: $CONFIG_DIR"
    echo "OUTPUT_DIR: '$out_dir/predictions/cascade_mask_rcnn_r2_101_fpn_20e_coco_tissuenet_w_train_all_test_on_${tissue}"
    # echo ${i##*/}
    echo '#!/bin/bash' > ${dir}/res2net-tissuenet-w-train-all-test-on-${tissue}.slurm

    echo '#SBATCH -J res2net-tissuenet-w-train-all-test-on-'${tissue} >> ${dir}/res2net-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo '#SBATCH -A sada-cnmi' >> ${dir}/res2net-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo '#SBATCH -p tier3' >> ${dir}/res2net-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --time=24:0:0' >> ${dir}/res2net-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --output=%x_%j.out' >> ${dir}/res2net-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --error=%x_%j.err' >> ${dir}/res2net-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --mem=200G' >> ${dir}/res2net-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --gres=gpu:a100:1' >> ${dir}/res2net-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo 'spack load cuda@11.0.2%gcc@9.3.0/lrd2rcw'  >> ${dir}/res2net-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo 'cd mmdetection'  >> ${dir}/res2net-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo 'nvidia-smi'  >> ${dir}/res2net-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo 'python mmdetection/tools/test.py '$CONFIG_DIR'/cascade_mask_rcnn_r2_101_fpn_20e_coco_tissuenet_w_train_all_test_on_'${tissue}'.py '$out_dir'/latest.pth --out '$out_dir'/predictions/cascade_mask_rcnn_r2_101_fpn_20e_coco_tissuenet_w_train_all_test_on_'${tissue}'/results.pkl --work-dir '$out_dir'/predictions/cascade_mask_rcnn_r2_101_fpn_20e_coco_tissuenet_w_train_all_test_on_'${tissue}' --eval segm' >> ${dir}/res2net-tissuenet-w-train-all-test-on-${tissue}.slurm
    
    sbatch ${dir}/res2net-tissuenet-w-train-all-test-on-${tissue}.slurm
done