# ResNeSt (Detectron2)

# Cellpose

## Data transformation

The structure of the datasets is identical to mmdet_BM, so you can directly follow data transformation in mmdet_BM. Please generate it once for use by three methods, including ResNeSt, Centermask2, and mmdet_BM.

## Installation

1. Create conda environments, use:

  ```
  conda create -n resnest python=3.7
  conda activate resnest
  ```

2. Install PyTorch, use:

  ```
  conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
```

3. Install the required packages, use:

  ```
  pip install cython==0.29.30
  pip install pycocotools==2.0.4
  ```
  
  Note: On my server, I installed it using ```conda install -c conda-forge pycocotools```, and conda also installed Cython along with it.

4. Install ResNeSt using a local clone, use:

  Before installing Detectron2, ensure that your machine has CUDA_HOME set up, as its absence can lead to a lack of GPU support when running programs; this can be   verified by running the command python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'. It's important to note that for ResNeSt, I have only successfully compiled it in a CUDA 10 environment, so you might need to load CUDA 10.2.89 with GCC 7.4.0 using spack load cuda@10.2.89%gcc@7.4.0, but this step depends on your server's setup and might be unnecessary if CUDA_HOME and GCC are already available.
  
  ```
  cd ResNeSt
  python -m pip install -e ResNeSt
  ```

## Training from scratch

Before beginning the training process, please ensure to update the path in ```register_coco_instances``` functions in ```tools/train_net.py``` file to your specific dataset directory. In addition, please modify the ```OUTPUT_DIR``` path in all ```benchmark_config/*.yaml``` files to your specific path.

See ```resnest_livecell.slurm```, ```resnest_tissuenet_n_1C.slurm```, ```resnest_tissuenet_n_2C.slurm```, ```resnest_tissuenet_w_1C.slurm``` and ```resnest_tissuenet_w_2C.slurm``` files.

Note: All the ```#SBATCH``` configurations in the above ```.slurm``` files are based on my current server settings. You will need to modify these parameters according to the specific requirements of your server.

## Test on tissues

Use:

```
sh resnest_livecell_train_all_test_on_tissues_submit.bash

sh resnest_tissuenet_n_train_all_test_on_tissues_submit.bash

sh resnest_tissuenet_n_train_all_test_on_tissues_submit_2C.bash

sh resnest_tissuenet_w_train_all_test_on_tissues_submit.bash

sh resnest_tissuenet_w_train_all_test_on_tissues_submit_2C.bash
```

Note: Same as training, you need to modify ```#SBATCH``` configurations based on your server.


## Citation

```
@article{zhang2020resnest,
title={ResNeSt: Split-Attention Networks},
author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
journal={arXiv preprint arXiv:2004.08955},
year={2020}
}
```
