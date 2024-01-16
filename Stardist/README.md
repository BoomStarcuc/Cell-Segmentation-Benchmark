# Stardist

## Data transformation
You can directly use the provided datasets without any transformation.

## Installation
1. Create conda environments, use:
```
conda create -n stardist python=3.7
conda activate stardist
```
   
2. Install the required packages:

```
pip install stardist
pip install pandas==1.5.1
pip install openpyxl==3.0.10
pip install imagecorruptions==1.1.2
pip install imgaug==0.4.0
```

## Training from scratch

For five experiment scenarios, run:

1. TissueNet. Data: dual-channel, label: nucleus
   
```
python stardist_nuclear_2C.py
```

2. TissueNet. Data: nucleus, label: nucleus
   
```
python stardist_nuclear_1C.py
```

3. TissueNet. Data: dual-channel, label: whole-cell
   
```
python stardist_wholecell_2C.py
```

4. TissueNet. Data: whole-cell, label: whole-cell
   
```
python stardist_wholecell_1C.py
```

5. LIVECell. Data: nucleus, label: nucleus

```
python stardist_livecell.py
```

Note: ```data_dir``` needs to be modified to your corresponding dataset path.

## Test

See ```cellpose_submit_livecell_test.slurm```, ```cellpose_submit_nuclear_test.slurm```, and ```cellpose_submit_wholecell_test.slurm``` files.

Note: --nchan_test needs to be modified based on the number of channels of your training. --nchan_test can be set to 1 or 2.
