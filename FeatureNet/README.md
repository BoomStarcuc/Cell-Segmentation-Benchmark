# FeatureNet

## Data transformation
You can directly use the provided datasets without any transformation.

## Installation

If you have already followed the installation steps for Mesmer, you can ignore the following instructions, as the FeatureNet environments are identical to those of Mesmer.

1. Create conda environments, use:
```
conda create -n deepcell python=3.8
conda activate deepcell
```
   
2. Install the required packages:

```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/ 
python3 -m pip install tensorflow
pip install deepcell
pip install pandas==1.5.1
pip install openpyxl==3.0.10
pip install imagecorruptions==1.1.2
pip install imgaug==0.4.0
```

Note: if your GPU is not recognized after installation, try to ```spack load cuda@11.0.2%gcc@9.3.0/lrd2rcw``` before starting the installation process.

## Training from scratch

For five experiment scenarios, run:

1. TissueNet. Data: dual-channel, label: nucleus
   
```
python featurenet_nuclear_2C.py
```

2. TissueNet. Data: nucleus, label: nucleus
   
```
python featurenet_nuclear_1C.py
```

3. TissueNet. Data: dual-channel, label: whole-cell
   
```
python featurenet_wholecell_2C.py
```

4. TissueNet. Data: whole-cell, label: whole-cell
   
```
python featurenet_wholecell_1C.py
```

5. LIVECell. Data: nucleus, label: nucleus

```
python featurenet_livecell.py
```

Note: ```data_dir``` needs to be modified to your corresponding dataset directory.

## Test

For five experiment scenarios, test:

1. TissueNet. Data: dual-channel, label: nucleus
   
```
python featurenet_nuclear_2C_test.py
```

2. TissueNet. Data: nucleus, label: nucleus
   
```
python featurenet_nuclear_1C_test.py
```

3. TissueNet. Data: dual-channel, label: whole-cell
   
```
python featurenet_wholecell_2C_test.py
```

4. TissueNet. Data: whole-cell, label: whole-cell
   
```
python featurenet_wholecell_1C_test.py
```

5. LIVECell. Data: nucleus, label: nucleus

```
python featurenet_livecell_test.py
```

Note: ```data_dir``` needs to be modified to your corresponding dataset directory.

## Citation

```
@article{van2016deep,
  title={Deep learning automates the quantitative analysis of individual cells in live-cell imaging experiments},
  author={Van Valen, David A and Kudo, Takamasa and Lane, Keara M and Macklin, Derek N and Quach, Nicolas T and DeFelice, Mialy M and Maayan, Inbal and Tanouchi, Yu and Ashley, Euan A and Covert, Markus W},
  journal={PLoS computational biology},
  volume={12},
  number={11},
  pages={e1005177},
  year={2016},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

