# Cell-Segmentation-Benchmark

## Datasets

The preprocessed datasets are available:

1. [TissueNet datasets](https://drive.google.com/drive/folders/1dUtqhvkF-M7nSwtxUpY0QmgHIgA4pinc?usp=sharing).
    Data structure:
   ```
   - **X**: Images with the shape of (14107, 256, 256, 2)
      - **Channel 0**: Nuclear images
      - **Channel 1**: Whole-cell images
   - **y**: Labels with the shape of (14107, 256, 256, 2)
      - **Channel 1**: Nuclear labels
      - **Channel 0**: Whole-cell labels
   - **filenames**: All filenames with the count of 14107
   - **tissue_list**: All tissue names including ten different tissue types
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
   ```
   
