# Segment Anything

## Data transformation
Since the structure of the datasets provided does not satisfy the format of the Segment Anything in test, you need to run the following code from the data transformation directory

```python transform_tissuenet.py```

```python transform_livecell.py```

Note: ```data_dir``` needs to be modified to your corresponding path.

## Installation

1. Create conda environments, use:
```
conda create -n segmentanything python=3.8
conda activate segmentanything
```
   
2. Install the required packages:

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install pandas==1.5.1
pip install openpyxl==3.0.10
pip install imagecorruptions==1.1.2
pip install imgaug==0.4.0
```

## Test

For five experiment scenarios, test:

1. TissueNet. Data: dual-channel, label: nucleus
   
```
python segmentanything_nuclear_2C_test.py
```

2. TissueNet. Data: nucleus, label: nucleus
   
```
python segmentanything_nuclear_1C_test.py
```

3. TissueNet. Data: dual-channel, label: whole-cell
   
```
python segmentanything_wholecell_2C_test.py
```

4. TissueNet. Data: whole-cell, label: whole-cell
   
```
python segmentanything_wholecell_1C_test.py
```

5. LIVECell. Data: nucleus, label: nucleus

```
python segmentanything_livecell_test.py
```

Note: ```data_dir``` and ```save_dir ``` need to be modified to correspond to your specific directories.

## Citation

```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
