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

Note: ```data_dir``` needs to be modified to your corresponding dataset directory.

## Test

For five experiment scenarios, test:

1. TissueNet. Data: dual-channel, label: nucleus
   
```
python stardist_nuclear_2C_test.py
```

2. TissueNet. Data: nucleus, label: nucleus
   
```
python stardist_nuclear_1C_test.py
```

3. TissueNet. Data: dual-channel, label: whole-cell
   
```
python stardist_wholecell_2C_test.py
```

4. TissueNet. Data: whole-cell, label: whole-cell
   
```
python stardist_wholecell_1C_test.py
```

5. LIVECell. Data: nucleus, label: nucleus

```
python stardist_livecell_test.py
```

Note: ```data_dir``` and ```model_dir``` need to be modified to your corresponding dataset directory and pre-trained model file.
