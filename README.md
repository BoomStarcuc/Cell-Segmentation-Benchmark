# A systematic evaluation of computation methods for cell segmentation

## Overview
<div align=center><img src="https://github.com/BoomStarcuc/Cell-Segmentation-Benchmark/blob/master/data/Overview.png" width="1000" height="550"/></div> 

## Datasets

The preprocessed datasets are available:

1. [TissueNet datasets](https://drive.google.com/drive/folders/1dUtqhvkF-M7nSwtxUpY0QmgHIgA4pinc?usp=sharing).
   
    Data structure:

   - **X**: Images with the shape of (N, 256, 256, 2)
      - **Channel 0**: Nuclear images
      - **Channel 1**: Whole-cell images
   - **y**: Labels with the shape of (N, 256, 256, 2)
      - **Channel 1**: Nuclear labels
      - **Channel 0**: Whole-cell labels
   - **filenames**: All filenames with the count of 14107
   - **tissue_list**: Tissue name corresponding to each image, including 10 different tissue types
      - 'Breast'
      - 'Colon'
      - 'Epidermis'
      - 'Esophagus'
      - 'Lung'
      - 'Lymph Node'
      - 'Pancreas'
      - 'Spleen'
      - 'Tonsil'
      - 'lymph node metastasis'

2. [LIVECell datasets](https://drive.google.com/drive/folders/1mJayXI2W9DLL17fsD3j2AcFySebnsoza?usp=sharing).
   
    Data structure:

   - **X**: Nuclear images with the shape of (N, 256, 256)
   - **y**: Nuclear labels with the shape of (N, 256, 256)
   - **tissue_list**: Cell type name corresponding to each image, including 8 different cell types
      - 'A172'
      - 'BT474'
      - 'BV2'
      - 'Huh7'
      - 'MCF7'
      - 'SHSY5Y'
      - 'SKOV3'
      - 'SkBr3'

## Methods

We provide the code for all 18 cell instance segmentation methods. Among these, Swin-S, Swin-T, Cascade Mask RCNN seesaw, SOLOv2, Res2Net, RF-Next, HRNet, Mask2former, Mask RCNN, and MS RCNN are integrated within the mmdetection framework. For training and testing instructions for each method, please refer to the detailed explanations provided within each method.

## Pre-trained models

We provide all pre-trained models conducted by our experiments, please find them on the provided [website](https://boomstarcuc.github.io/cellseg-benchmarking/). Therefore, you can select a model based on your own data.
