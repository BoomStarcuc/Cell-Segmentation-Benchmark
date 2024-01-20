#!/bin/bash

# List of tissue names
tissues=("Colon" "lymph_node_metastasis" "Spleen" "Pancreas" "Epidermis" "Breast" "Lymph_Node" "Tonsil" "Lung" "Esophagus")
CONFIG_DIR='mmdetection/configs/seesaw_loss2C'
# Loop over each tissue name
for tissue in ${tissues[@]}
do
    # Copy A.py to a new file, replacing 'Epidermis' with the tissue name
    cp $CONFIG_DIR/cascade_mask_rcnn_r101_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1_tissuenet_n.py $CONFIG_DIR/cascade_mask_rcnn_r101_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1_tissuenet_n_train_all_test_on_${tissue}.py 
    sed -i "s/_all/_${tissue}/g" $CONFIG_DIR/cascade_mask_rcnn_r101_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1_tissuenet_n_train_all_test_on_${tissue}.py
done

dir=$CONFIG_DIR/seesaw_tissuenet_n_all_to_all_test_2C_slurm
out_dir='/shared/rc/spl/mmdet_output/All_to_all/nuclear/Cascade_Mask_RCNN_seesaw'
if [[ ! -e $dir ]]; then
    mkdir $dir
elif [[ ! -d $dir ]]; then
    echo "$dir already exists but is not a directory" 1>&2
fi
for tissue in ${tissues[@]}
do
    echo "CONFIG_DIR: $CONFIG_DIR"
    echo "OUTPUT_DIR: '$out_dir/predictions/seesaw_loss_normed_mask_mstrain_2x_lvis_v1_tissuenet_n_train_all_test_on_${tissue}"
    # echo ${i##*/}
    echo '#!/bin/bash' > ${dir}/seesaw-tissuenet-n-train-all-test-on-${tissue}.slurm

    echo '#SBATCH -J seesaw-tissuenet-n-train-all-test-on-'${tissue} >> ${dir}/seesaw-tissuenet-n-train-all-test-on-${tissue}.slurm
    echo '#SBATCH -A sada-cnmi' >> ${dir}/seesaw-tissuenet-n-train-all-test-on-${tissue}.slurm
    echo '#SBATCH -p tier3' >> ${dir}/seesaw-tissuenet-n-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --time=24:0:0' >> ${dir}/seesaw-tissuenet-n-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --output=%x_%j.out' >> ${dir}/seesaw-tissuenet-n-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --error=%x_%j.err' >> ${dir}/seesaw-tissuenet-n-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --mem=200G' >> ${dir}/seesaw-tissuenet-n-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --gres=gpu:a100:1' >> ${dir}/seesaw-tissuenet-n-train-all-test-on-${tissue}.slurm
    echo 'spack load cuda@11.0.2%gcc@9.3.0/lrd2rcw'  >> ${dir}/seesaw-tissuenet-n-train-all-test-on-${tissue}.slurm
    echo 'cd mmdetection'  >> ${dir}/seesaw-tissuenet-n-train-all-test-on-${tissue}.slurm
    echo 'nvidia-smi'  >> ${dir}/seesaw-tissuenet-n-train-all-test-on-${tissue}.slurm
    echo 'python mmdetection/tools/test.py '$CONFIG_DIR'/cascade_mask_rcnn_r101_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1_tissuenet_n_train_all_test_on_'${tissue}'.py '$out_dir'/latest.pth --out '$out_dir'/predictions/seesaw_loss_normed_mask_mstrain_2x_lvis_v1_tissuenet_n_train_all_test_on_'${tissue}'/results.pkl --work-dir '$out_dir'/predictions/seesaw_loss_normed_mask_mstrain_2x_lvis_v1_tissuenet_n_train_all_test_on_'${tissue}' --eval segm' >> ${dir}/seesaw-tissuenet-n-train-all-test-on-${tissue}.slurm
    
    sbatch ${dir}/seesaw-tissuenet-n-train-all-test-on-${tissue}.slurm
done