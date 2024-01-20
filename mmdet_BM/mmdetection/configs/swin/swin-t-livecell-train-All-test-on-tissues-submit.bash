#!/bin/bash

# List of tissue names
tissues=('A172' 'BT474' 'BV2' 'Huh7' 'MCF7' 'SHSY5Y' 'SkBr3' 'SKOV3')
CONFIG_DIR='mmdetection/configs/swin'
# Loop over each tissue name
for tissue in ${tissues[@]}
do
    # Copy A.py to a new file, replacing 'Epidermis' with the tissue name
    cp $CONFIG_DIR/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-50e_coco_livecell.py $CONFIG_DIR/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-50e_coco_livecell_train_all_test_on_${tissue}.py 
    sed -i "s/_all/_${tissue}/g" $CONFIG_DIR/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-50e_coco_livecell_train_all_test_on_${tissue}.py
done

dir=$CONFIG_DIR/livecell_swin-t_all_to_all_test_slurm
out_dir='/shared/rc/spl/mmdet_output/All_to_all/livecell/swin-t-p4-w7_fpn_ms-crop-50e_coco_livecell'
if [[ ! -e $dir ]]; then
    mkdir $dir
elif [[ ! -d $dir ]]; then
    echo "$dir already exists but is not a directory" 1>&2
fi
for tissue in ${tissues[@]}
do
    echo "CONFIG_DIR: $CONFIG_DIR"
    echo "OUTPUT_DIR: '$out_dir/predictions/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-50e_coco_livecell_train_all_test_on_${tissue}"
    # echo ${i##*/}
    echo '#!/bin/bash' > ${dir}/swin-t-livecell-train-all-test-on-${tissue}.slurm

    echo '#SBATCH -J swin-t-livecell-train-all-test-on-'${tissue} >> ${dir}/swin-t-livecell-train-all-test-on-${tissue}.slurm
    echo '#SBATCH -A sada-cnmi' >> ${dir}/swin-t-livecell-train-all-test-on-${tissue}.slurm
    echo '#SBATCH -p tier3' >> ${dir}/swin-t-livecell-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --time=24:0:0' >> ${dir}/swin-t-livecell-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --output=%x_%j.out' >> ${dir}/swin-t-livecell-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --error=%x_%j.err' >> ${dir}/swin-t-livecell-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --mem=200G' >> ${dir}/swin-t-livecell-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --gres=gpu:a100:1' >> ${dir}/swin-t-livecell-train-all-test-on-${tissue}.slurm
    echo 'spack load cuda@11.0.2%gcc@9.3.0/lrd2rcw'  >> ${dir}/swin-t-livecell-train-all-test-on-${tissue}.slurm
    echo 'cd mmdetection'  >> ${dir}/swin-t-livecell-train-all-test-on-${tissue}.slurm
    echo 'nvidia-smi'  >> ${dir}/swin-t-livecell-train-all-test-on-${tissue}.slurm
    echo 'python mmdetection/tools/test.py '$CONFIG_DIR'/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-50e_coco_livecell_train_all_test_on_'${tissue}'.py '$out_dir'/latest.pth --out '$out_dir'/predictions/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-50e_coco_livecell_train_all_test_on_'${tissue}'/results.pkl --work-dir '$out_dir'/predictions/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-50e_coco_livecell_train_all_test_on_'${tissue}' --eval segm' >> ${dir}/swin-t-livecell-train-all-test-on-${tissue}.slurm
    
    sbatch ${dir}/swin-t-livecell-train-all-test-on-${tissue}.slurm
done