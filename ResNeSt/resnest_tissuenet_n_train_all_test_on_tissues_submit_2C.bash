#! \bin\bash

declare -a arr=("Colon" "lymph_node_metastasis" "Spleen" "Pancreas" "Epidermis" "Breast" "Lymph_Node" "Tonsil" "Lung" "Esophagus")
RESNEST_ROOT_DIR='ResNeSt'
CONFIG_DIR='benchmark_config'
dir=$RESNEST_ROOT_DIR/tissuenet_n_all_to_all_test_2C_slurm
out_dir='/shared/rc/spl/mmdet_output/All_to_all/nuclear/Resnest_tissuenet_n_train_2C'
if [[ ! -e $dir ]]; then
    mkdir $dir
elif [[ ! -d $dir ]]; then
    echo "$dir already exists but is not a directory" 1>&2
fi
for i in "${arr[@]}"
do
    echo "EXPERIMENT:$i"
    echo "ROOT_DIR: $RESNEST_ROOT_DIR"
    echo "CONFIG_DIR: $CONFIG_DIR"
    echo "OUTPUT_DIR: '$out_dir/predictions/resnest_tissuenet_n_train_all_test_2C_on_$i"
    echo ${i##*/}
    echo '#!/bin/bash' > ${dir}/resnest_tissuenet_n_train_All_test_2C_on_${i}.slurm

    echo '#SBATCH -J resnest_tissuenet_n_train_all_test_2C_on_'${i} >> ${dir}/resnest_tissuenet_n_train_All_test_2C_on_${i}.slurm
    echo '#SBATCH -p scavenger-gpu' >> ${dir}/resnest_tissuenet_n_train_All_test_2C_on_${i}.slurm
    echo '#SBATCH --cpus-per-task=10' >> ${dir}/resnest_tissuenet_n_train_All_test_2C_on_${i}.slurm
    echo '#SBATCH --gpus-per-node=RTXA5000:1' >> ${dir}/resnest_tissuenet_n_train_All_test_2C_on_${i}.slurm
    echo '#SBATCH --mem=100G' >> ${dir}/resnest_tissuenet_n_train_All_test_2C_on_${i}.slurm
    echo '#SBATCH --exclusive' >> ${dir}/resnest_tissuenet_n_train_All_test_2C_on_${i}.slurm
    echo '#SBATCH --time=24:0:0' >> ${dir}/resnest_tissuenet_n_train_All_test_2C_on_${i}.slurm
    echo '#SBATCH --output=%x_%j.out' >> ${dir}/resnest_tissuenet_n_train_All_test_2C_on_${i}.slurm
    echo '#SBATCH --error=%x_%j.err' >> ${dir}/resnest_tissuenet_n_train_All_test_2C_on_${i}.slurm
    echo 'cd ResNeSt'  >> ${dir}/resnest_tissuenet_n_train_All_test_2C_on_${i}.slurm
    echo 'nvidia-smi'  >> ${dir}/resnest_tissuenet_n_train_All_test_2C_on_${i}.slurm
    echo 'python '$RESNEST_ROOT_DIR'/tools/train_net.py --config-file '$CONFIG_DIR'/tissuenet_nuclear_train_all_test_2C_on_'$i'.yaml --eval-only MODEL.WEIGHTS '$out_dir'/model_0049999.pth OUTPUT_DIR '$out_dir'/predictions/resnest_tissuenet_n_train_all_test_2C_on_'$i >> ${dir}/resnest_tissuenet_n_train_All_test_2C_on_${i}.slurm
    
    sbatch ${dir}/resnest_tissuenet_n_train_All_test_2C_on_${i}.slurm
done
