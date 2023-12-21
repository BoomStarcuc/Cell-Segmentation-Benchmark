# Cellpose

## Data transformation
Since the structure of datasets provided do not satisfy the format of training cellpose, so you need to run the following code from data transformation directory

```python transform_tissuenet.py```

```python transform_livecell.py```

Note: ```data_dir``` need to be modified to your corresponding path.

## Local installation (< 2 minutes)

### System requirements

Linux, Windows and Mac OS are supported for running the code. At least 8GB of RAM is required to run the software. 16GB-32GB may be required for larger images and 3D volumes. The software has been heavily tested on Windows 10 and Ubuntu 18.04 and less well-tested on Mac OS. Please open an issue if you have problems with installation.

### Dependencies

cellpose relies on the following excellent packages (which are automatically installed with conda/pip if missing):
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

### Installation of github version

Follow steps from above to install the dependencies. Then download cellpose_BM in the github repository folder. And then cd 'cellpose_BM' and run `pip install -e .`.
