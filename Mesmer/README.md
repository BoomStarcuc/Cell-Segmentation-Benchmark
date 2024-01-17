# Mesmer

## Data transformation
You can directly use the provided datasets without any transformation.

## Installation

If you have already followed the installation steps for FeatureNet, you can ignore the following instructions, as the Mesmer environments are identical to those of FeatureNet.

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
python mesmer_nuclear_2C.py
```

2. TissueNet. Data: nucleus, label: nucleus
   
```
python mesmer_nuclear_1C.py
```

3. TissueNet. Data: dual-channel, label: whole-cell
   
```
python mesmer_wholecell_2C.py
```

4. TissueNet. Data: whole-cell, label: whole-cell
   
```
python mesmer_wholecell_1C.py
```

5. LIVECell. Data: nucleus, label: nucleus

```
python mesmer_livecell.py
```

Note: ```data_dir``` needs to be modified to your corresponding dataset directory.

## Test

For five experiment scenarios, test:

1. TissueNet. Data: dual-channel, label: nucleus
   
```
python mesmer_nuclear_2C_test.py
```

2. TissueNet. Data: nucleus, label: nucleus
   
```
python mesmer_nuclear_1C_test.py
```

3. TissueNet. Data: dual-channel, label: whole-cell
   
```
python mesmer_wholecell_2C_test.py
```

4. TissueNet. Data: whole-cell, label: whole-cell
   
```
python mesmer_wholecell_1C_test.py
```

5. LIVECell. Data: nucleus, label: nucleus

```
python mesmer_livecell_test.py
```

Note: ```data_dir``` needs to be modified to your corresponding dataset directory.

## Citation

```
@ARTICLE{Greenwald2022-uc,
  title     = "Whole-cell segmentation of tissue images with human-level
               performance using large-scale data annotation and deep learning",
  author    = "Greenwald, Noah F and Miller, Geneva and Moen, Erick and Kong,
               Alex and Kagel, Adam and Dougherty, Thomas and Fullaway,
               Christine Camacho and McIntosh, Brianna J and Leow, Ke Xuan and
               Schwartz, Morgan Sarah and Pavelchek, Cole and Cui, Sunny and
               Camplisson, Isabella and Bar-Tal, Omer and Singh, Jaiveer and
               Fong, Mara and Chaudhry, Gautam and Abraham, Zion and Moseley,
               Jackson and Warshawsky, Shiri and Soon, Erin and Greenbaum,
               Shirley and Risom, Tyler and Hollmann, Travis and Bendall, Sean
               C and Keren, Leeat and Graf, William and Angelo, Michael and Van
               Valen, David",
  abstract  = "A principal challenge in the analysis of tissue imaging data is
               cell segmentation-the task of identifying the precise boundary
               of every cell in an image. To address this problem we
               constructed TissueNet, a dataset for training segmentation
               models that contains more than 1 million manually labeled cells,
               an order of magnitude more than all previously published
               segmentation training datasets. We used TissueNet to train
               Mesmer, a deep-learning-enabled segmentation algorithm. We
               demonstrated that Mesmer is more accurate than previous methods,
               generalizes to the full diversity of tissue types and imaging
               platforms in TissueNet, and achieves human-level performance.
               Mesmer enabled the automated extraction of key cellular
               features, such as subcellular localization of protein signal,
               which was challenging with previous approaches. We then adapted
               Mesmer to harness cell lineage information in highly multiplexed
               datasets and used this enhanced version to quantify cell
               morphology changes during human gestation. All code, data and
               models are released as a community resource.",
  journal   = "Nat. Biotechnol.",
  publisher = "Springer Science and Business Media LLC",
  volume    =  40,
  number    =  4,
  pages     = "555--565",
  month     =  apr,
  year      =  2022,
  language  = "en"
}
```

