#!/bin/bash

tissues=('A172' 'BT474' 'BV2' 'Huh7' 'MCF7' 'SHSY5Y' 'SkBr3' 'SKOV3')
CONFIG_DIR='mmdetection/configs/swin'

for tissue in ${tissues[@]}
do
    # Copy A.py to a new file, replacing 'Epidermis' with the tissue name
    tissues_test=('A172' 'BT474' 'BV2' 'Huh7' 'MCF7' 'SHSY5Y' 'SkBr3' 'SKOV3')
    for tis_test in ${tissues_test[@]}
    do 
        cp $CONFIG_DIR/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_livecell_${tissue}.py $CONFIG_DIR/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_livecell_${tissue}_test_on_${tis_test}.py
        sed -i "s/_all/_${tis_test}/g" $CONFIG_DIR/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_livecell_${tissue}_test_on_${tis_test}.py 
    done
done

dir=$CONFIG_DIR/livecell_swin-s_one_for_all_test_slurm
out_dir='/shared/rc/spl/mmdet_output/One_to_all/livecell/swin-s'
if [[ ! -e $dir ]]; then
    mkdir $dir
elif [[ ! -d $dir ]]; then
    echo "$dir already exists but is not a directory" 1>&2
fi
for tissue in ${tissues[@]}
do
    for tis_test in ${tissues_test[@]}
    do 
        echo "CONFIG_DIR: $CONFIG_DIR"
        echo "OUTPUT_DIR: '$out_dir/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_livecell_${tissue}/predictions/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_livecell_${tissue}_test_on_${tis_test}"
        # echo ${i##*/}
        echo '#!/bin/bash' > ${dir}/swin-s-livecell-${tissue}-test-on-${tis_test}.slurm

        echo '#SBATCH -J swin-s-livecell-on-'${tissue}'-test-on-'${tis_test}>> ${dir}/swin-s-livecell-${tissue}-test-on-${tis_test}.slurm
        echo '#SBATCH -A sada-cnmi' >> ${dir}/swin-s-livecell-${tissue}-test-on-${tis_test}.slurm
        echo '#SBATCH -p tier3' >> ${dir}/swin-s-livecell-${tissue}-test-on-${tis_test}.slurm
        echo '#SBATCH --time=2:0:0' >> ${dir}/swin-s-livecell-${tissue}-test-on-${tis_test}.slurm
        echo '#SBATCH --output=%x_%j.out' >> ${dir}/swin-s-livecell-${tissue}-test-on-${tis_test}.slurm
        echo '#SBATCH --error=%x_%j.err' >> ${dir}/swin-s-livecell-${tissue}-test-on-${tis_test}.slurm
        echo '#SBATCH --mem=200G' >> ${dir}/swin-s-livecell-${tissue}-test-on-${tis_test}.slurm
        echo '#SBATCH --gres=gpu:a100:2' >> ${dir}/swin-s-livecell-${tissue}-test-on-${tis_test}.slurm
        echo 'spack load cuda@11.0.2%gcc@9.3.0/lrd2rcw'  >> ${dir}/swin-s-livecell-${tissue}-test-on-${tis_test}.slurm
        echo 'cd mmdetection'  >> ${dir}/swin-s-livecell-${tissue}-test-on-${tis_test}.slurm
        echo 'nvidia-smi'  >> ${dir}/swin-s-livecell-${tissue}-test-on-${tis_test}.slurm
        echo 'sh mmdetection/tools/dist_test.sh '$CONFIG_DIR'/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_livecell_'${tissue}'_test_on_'${tis_test}'.py '$out_dir'/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_livecell_'${tissue}'/latest.pth 2 --out '$out_dir'/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_livecell_'${tissue}'/predictions/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_livecell_'${tissue}'_test_on_'${tis_test}'/results.pkl --work-dir '$out_dir'/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_livecell_'${tissue}'/predictions/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_livecell_'${tissue}'_test_on_'${tis_test}' --eval segm' >> ${dir}/swin-s-livecell-${tissue}-test-on-${tis_test}.slurm
        # echo 'python mmdetection/tools/test.py '$CONFIG_DIR'/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_livecell_'${tissue}'_test_on_'${tis_test}'.py '$out_dir'/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_livecell_'${tissue}'/latest.pth --out '$out_dir'/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_livecell_'${tissue}'/predictions/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_livecell_'${tissue}'_test_on_'${tis_test}'/results.pkl --work-dir '$out_dir'/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_livecell_'${tissue}'/predictions/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-50e_coco_livecell_'${tissue}'_test_on_'${tis_test}' --eval segm' >> ${dir}/swin-s-livecell-${tissue}-test-on-${tis_test}.slurm
        
        sbatch ${dir}/swin-s-livecell-${tissue}-test-on-${tis_test}.slurm
    done
done