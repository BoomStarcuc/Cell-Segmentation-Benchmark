# Stardist

## Data transformation
You can directly use the provided datasets without any transformation.

## installation

1. Create conda environments, use:
   ```conda create -n stardist python=3.7
      conda activate stardist```
   
2. Install the required packages:

```pip install stardist```

```pip install pandas==1.5.1```

```pip install openpyxl==3.0.10```

```pip install imagecorruptions==1.1.2```

```pip install imgaug==0.4.0```


## Training from scratch

See ```cellpose_submit_livecell_train.slurm```, ```cellpose_submit_nuclear_train.slurm```, and ```cellpose_submit_wholecell_train.slurm``` files.

Note: Cellpose will automatically identify the number of channels of your input. You need to follow the code from the data transformation directory to generate the correct structure of the dataset.

## Test

See ```cellpose_submit_livecell_test.slurm```, ```cellpose_submit_nuclear_test.slurm```, and ```cellpose_submit_wholecell_test.slurm``` files.

Note: --nchan_test needs to be modified based on the number of channels of your training. --nchan_test can be set to 1 or 2.
