# Centermask2 (Detectron2)

## Data transformation

The structure of the datasets is identical to mmdet_BM, so you can directly follow data transformation in mmdet_BM. Please generate it once for use by three methods, including ResNeSt, Centermask2, and mmdet_BM.

## Installation

1. Create conda environments, use:

  ```
  conda create -n centermask2 python=3.7
  conda activate centermask2
  ```

2. Install PyTorch, use:

  ```
  conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
  ```

3. Install the required packages, use:

  ```
  pip install cython==0.29.30
  pip install pycocotools==2.0.4
  ```
  
  Note: On my server, I installed it using ```conda install -c conda-forge pycocotools```, and conda also installed Cython along with it.

4. Install Detectron2 using a local clone, use:

  Before installing Detectron2, ensure that your machine has CUDA_HOME set up, as its absence can lead to a lack of GPU support when running programs; this can be verified by running the command ```python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'```. You might need to load CUDA 11.0.2 with GCC 9.3.0 using ```spack load cuda@11.0.2%gcc@9.3.0/lrd2rcw```, but this step depends on your server's setup and might be unnecessary if CUDA_HOME and GCC are already available.
  
  ```
  python -m pip install -e detectron2
  ```

  Note: my Detectron2 version is 0.3.
  
## Training from scratch

Before beginning the training process, please ensure to update the path in the ```register_coco_instances``` functions in the ```train_net.py``` file to your specific dataset directory. In addition, please modify the ```OUTPUT_DIR``` path in all ```benchmark_config/*.yaml``` files to your specific path.

For single-channel data training:

```
cd centermask2
 
sbatch centermask2_livecell.slurm

sbatch centermask2_tissuenet_n_1C.slurm

sbatch centermask2_tissuenet_w_1C.slurm
```

For dual-channel data training:

```
cd centermask22C
 
sbatch centermask2_tissuenet_n_2C.slurm

sbatch centermask2_tissuenet_w_2C.slurm
```

Note: All the ```#SBATCH``` configurations in the above ```.slurm``` files are based on my current server settings. You will need to modify these parameters according to the specific requirements of your server.


## Test on tissues

For single-channel data testing:

```
cd centermask2

sh centermask2_livecell_train_all_test_on_tissues_submit.bash

sh centermask2_tissuenet_n_train_all_test_on_tissues_submit.bash

sh centermask2_tissuenet_w_train_all_test_on_tissues_submit.bash
```

For dual-channel data testing:

```
cd centermask22C
 
sh centermask2_tissuenet_n_train_all_test_on_tissues_submit.bash

sh centermask2_tissuenet_w_train_all_test_on_tissues_submit.bash
```

Note: Same as training, you need to modify ```#SBATCH``` configurations based on your server.


## Citation

```
@inproceedings{lee2020centermask,
  title={CenterMask: Real-Time Anchor-Free Instance Segmentation},
  author={Lee, Youngwan and Park, Jongyoul},
  booktitle={CVPR},
  year={2020}
}
```
