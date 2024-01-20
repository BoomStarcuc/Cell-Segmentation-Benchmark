#!/bin/bash

# List of tissue names
tissues=('A172' 'BT474' 'BV2' 'Huh7' 'MCF7' 'SHSY5Y' 'SkBr3' 'SKOV3')
CONFIG_DIR='mmdetection/configs/rfnext'
# Loop over each tissue name
for tissue in ${tissues[@]}
do
    # Copy A.py to a new file, replacing 'Epidermis' with the tissue name
    cp $CONFIG_DIR/rfnext_fixed_multi_branch_cascade_mask_rcnn_r2_101_fpn_200e_coco_livecell.py $CONFIG_DIR/rfnext_fixed_multi_branch_cascade_mask_rcnn_r2_101_fpn_200e_coco_livecell_train_all_test_on_${tissue}.py 
    sed -i "s/_all/_${tissue}/g" $CONFIG_DIR/rfnext_fixed_multi_branch_cascade_mask_rcnn_r2_101_fpn_200e_coco_livecell_train_all_test_on_${tissue}.py
done

dir=$CONFIG_DIR/rfnext_livecell_all_to_all_test_slurm
out_dir='/shared/rc/spl/mmdet_output/All_to_all/livecell/rfnext_fixed_multi_branch_cascade_mask_rcnn_r2_101_fpn_200e_coco_livecell'
if [[ ! -e $dir ]]; then
    mkdir $dir
elif [[ ! -d $dir ]]; then
    echo "$dir already exists but is not a directory" 1>&2
fi
for tissue in ${tissues[@]}
do
    echo "CONFIG_DIR: $CONFIG_DIR"
    echo "OUTPUT_DIR: '$out_dir/predictions/rfnext_fixed_multi_branch_cascade_mask_rcnn_r2_101_fpn_200e_coco_livecell_train_all_test_on_${tissue}"
    # echo ${i##*/}
    echo '#!/bin/bash' > ${dir}/rfnext-livecell-train-all-test-on-${tissue}.slurm

    echo '#SBATCH -J rfnext-livecell-train-all-test-on-'${tissue} >> ${dir}/rfnext-livecell-train-all-test-on-${tissue}.slurm
    echo '#SBATCH -A sada-cnmi' >> ${dir}/rfnext-livecell-train-all-test-on-${tissue}.slurm
    echo '#SBATCH -p tier3' >> ${dir}/rfnext-livecell-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --time=24:0:0' >> ${dir}/rfnext-livecell-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --output=%x_%j.out' >> ${dir}/rfnext-livecell-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --error=%x_%j.err' >> ${dir}/rfnext-livecell-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --mem=200G' >> ${dir}/rfnext-livecell-train-all-test-on-${tissue}.slurm
    echo '#SBATCH --gres=gpu:a100:1' >> ${dir}/rfnext-livecell-train-all-test-on-${tissue}.slurm
    echo 'spack load cuda@11.0.2%gcc@9.3.0/lrd2rcw'  >> ${dir}/rfnext-livecell-train-all-test-on-${tissue}.slurm
    echo 'cd mmdetection'  >> ${dir}/rfnext-livecell-train-all-test-on-${tissue}.slurm
    echo 'nvidia-smi'  >> ${dir}/rfnext-livecell-train-all-test-on-${tissue}.slurm
    echo 'python mmdetection/tools/test.py '$CONFIG_DIR'/rfnext_fixed_multi_branch_cascade_mask_rcnn_r2_101_fpn_200e_coco_livecell_train_all_test_on_'${tissue}'.py '$out_dir'/latest.pth --out '$out_dir'/predictions/rfnext_fixed_multi_branch_cascade_mask_rcnn_r2_101_fpn_200e_coco_livecell_train_all_test_on_'${tissue}'/results.pkl --work-dir '$out_dir'/predictions/rfnext_fixed_multi_branch_cascade_mask_rcnn_r2_101_fpn_200e_coco_livecell_train_all_test_on_'${tissue}' --eval segm' >> ${dir}/rfnext-livecell-train-all-test-on-${tissue}.slurm
    
    sbatch ${dir}/rfnext-livecell-train-all-test-on-${tissue}.slurm
done