# Cellpose

## Data transformation
Since the structure of the datasets provided does not satisfy the format of the training cellpose, you need to run the following code from the data transformation directory

```python transform_tissuenet.py```

```python transform_livecell.py```

Note: ```data_dir``` needs to be modified to your corresponding path.

## Local installation (< 2 minutes)

### System requirements

Linux, Windows and Mac OS are supported for running the code. At least 8GB of RAM is required to run the software. 16GB-32GB may be required for larger images and 3D volumes. The software has been heavily tested on Windows 10 and Ubuntu 18.04 and less well-tested on Mac OS. Please open an issue if you have problems with installation.

### Dependencies

Cellpose relies on the following excellent packages (which are automatically installed with conda/pip if missing):
- [pytorch](https://pytorch.org/)
- [pyqtgraph](http://pyqtgraph.org/)
- [PyQt5](http://pyqt.sourceforge.net/Docs/PyQt5/)
- [numpy](http://www.numpy.org/) (>=1.16.0)
- [numba](http://numba.pydata.org/numba-doc/latest/user/5minguide.html)
- [scipy](https://www.scipy.org/)
- [natsort](https://natsort.readthedocs.io/en/master/)
- [pandas](https://pypi.org/project/pandas/1.5.1/) (1.5.1)
- [openpyxl](https://pypi.org/project/openpyxl/3.0.10/)(3.0.10)
- [imagecorruptions](https://pypi.org/project/imagecorruptions/)(1.1.2)
- [imgaug](https://pypi.org/project/imgaug/)(0.4.0)

### Installation of GitHub version

Follow the steps from above to install the dependencies. Then download cellpose_BM in the GitHub repository folder. And then cd 'cellpose_BM' and run `pip install -e .`.

## Training from scratch

See ```cellpose_submit_livecell_train.slurm```, ```cellpose_submit_nuclear_train.slurm```, and ```cellpose_submit_wholecell_train.slurm``` files.

Note: Cellpose will automatically identify the number of channels of your input. You need to follow the code from the data transformation directory to generate the correct structure of the dataset.

## Test

See ```cellpose_submit_livecell_test.slurm```, ```cellpose_submit_nuclear_test.slurm```, and ```cellpose_submit_wholecell_test.slurm``` files.

Note: --nchan_test needs to be modified based on the number of channels of your training. --nchan_test can be set to 1 or 2.

## Citation

```
@article{stringer2021cellpose,
  title={Cellpose: a generalist algorithm for cellular segmentation},
  author={Stringer, Carsen and Wang, Tim and Michaelos, Michalis and Pachitariu, Marius},
  journal={Nature methods},
  volume={18},
  number={1},
  pages={100--106},
  year={2021},
  publisher={Nature Publishing Group US New York}
}
```
