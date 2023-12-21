# Cell-Segmentation-Benchmark

## Overview
<div align=center><img src="https://github.com/BoomStarcuc/Cell-Segmentation-Benchmark/blob/master/data/Overview.png" width="1000" height="400"/></div> 

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
