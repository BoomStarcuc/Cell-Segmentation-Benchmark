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

  ```conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts```

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
  python -m pip install -e detectron2-ResNeSt
  ```

## Training from scratch

See ```cellpose_submit_livecell_train.slurm```, ```cellpose_submit_nuclear_train.slurm```, and ```cellpose_submit_wholecell_train.slurm``` files.

Note: Cellpose will automatically identify the number of channels of your input. You need to follow the code from the data transformation directory to generate the correct structure of the dataset.

## Test

See ```cellpose_submit_livecell_test.slurm```, ```cellpose_submit_nuclear_test.slurm```, and ```cellpose_submit_wholecell_test.slurm``` files.

Note: --nchan_test needs to be modified based on the number of channels of your training. --nchan_test can be set to 1 or 2.

## Citation

```
@article{zhang2020resnest,
title={ResNeSt: Split-Attention Networks},
author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
journal={arXiv preprint arXiv:2004.08955},
year={2020}
}
```
