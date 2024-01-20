#!/bin/bash

# List of tissue names
tissues=("Colon" "lymph_node_metastasis" "Spleen" "Pancreas" "Epidermis" "Breast" "Lymph_Node" "Tonsil" "Lung" "Esophagus")
CONFIG_DIR='mmdetection/configs/strong_baselines'
# Loop over each tissue name
for tissue in ${tissues[@]}
do
    # Copy A.py to a new file, replacing 'Epidermis' with the tissue name
    cp $CONFIG_DIR/mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco_tissuenet_w.py $CONFIG_DIR/mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco_tissuenet_w_train_all_test_on_${tissue}.py 
    sed -i "s/_all/_${tissue}/g" $CONFIG_DIR/mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco_tissuenet_w_train_all_test_on_${tissue}.py 
done

dir=$CONFIG_DIR/MRF2conv_tissuenet_w_all_to_all_test_slurm
out_dir='/shared/rc/spl/mmdet_output/All_to_all/wholecell_oc/mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco_tissuenet_w'
if [[ ! -e $dir ]]; then
    mkdir $dir
elif [[ ! -d $dir ]]; then
    echo "$dir already exists but is not a directory" 1>&2
fi
for tissue in ${tissues[@]}
do
    echo "CONFIG_DIR: $CONFIG_DIR"
    echo "OUTPUT_DIR: '$out_dir/predictions/mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco_tissuenet_w_train_all_test_on_${tissue}"
    # echo ${i##*/}
    echo '#!/bin/bash' > ${dir}/MRF2conv-tissuenet-w-train-all-test-on-${tissue}.slurm

    echo '#SBATCH -J MRF2conv-tissuenet-w-train-all-test-on-'${tissue} >> ${dir}/MRF2conv-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo '#SBATCH -A sada-cnmi' >> ${dir}/MRF2conv-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo '#SBATCH -p tier3' >> ${dir}/MRF2conv-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --time=24:0:0' >> ${dir}/MRF2conv-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --output=%x_%j.out' >> ${dir}/MRF2conv-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --error=%x_%j.err' >> ${dir}/MRF2conv-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --mem=200G' >> ${dir}/MRF2conv-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --gres=gpu:a100:2' >> ${dir}/MRF2conv-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo 'spack load cuda@11.0.2%gcc@9.3.0/lrd2rcw'  >> ${dir}/MRF2conv-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo 'cd mmdetection'  >> ${dir}/MRF2conv-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo 'nvidia-smi'  >> ${dir}/MRF2conv-tissuenet-w-train-all-test-on-${tissue}.slurm
    echo 'bash mmdetection/tools/dist_test.sh '$CONFIG_DIR'/mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco_tissuenet_w_train_all_test_on_'${tissue}'.py '$out_dir'/latest.pth 2 --out '$out_dir'/predictions/mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco_tissuenet_w_train_all_test_on_'${tissue}'/results.pkl --work-dir '$out_dir'/predictions/mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco_tissuenet_w_train_all_test_on_'${tissue}' --eval segm' >> ${dir}/MRF2conv-tissuenet-w-train-all-test-on-${tissue}.slurm
    # echo 'python mmdetection/tools/test.py '$CONFIG_DIR'/mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco_tissuenet_w_train_all_test_on_'${tissue}'.py '$out_dir'/latest.pth --out '$out_dir'/predictions/mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco_tissuenet_w_train_all_test_on_'${tissue}'/results.pkl --work-dir '$out_dir'/predictions/mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco_tissuenet_w_train_all_test_on_'${tissue}' --eval segm' >> ${dir}/MRF2conv-tissuenet-w-train-all-test-on-${tissue}.slurm
    
    sbatch ${dir}/MRF2conv-tissuenet-w-train-all-test-on-${tissue}.slurm
done