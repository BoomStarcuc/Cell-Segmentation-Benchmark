## Local installation (< 2 minutes)

### System requirements

Linux, Windows and Mac OS are supported for running the code. For running the graphical interface you will need a Mac OS later than Yosemite. At least 8GB of RAM is required to run the software. 16GB-32GB may be required for larger images and 3D volumes. The software has been heavily tested on Windows 10 and Ubuntu 18.04 and less well-tested on Mac OS. Please open an issue if you have problems with installation.

### Dependencies
cellpose relies on the following excellent packages (which are automatically installed with conda/pip if missing):
- [pytorch](https://pytorch.org/)
- [pyqtgraph](http://pyqtgraph.org/)
- [PyQt5](http://pyqt.sourceforge.net/Docs/PyQt5/)
- [numpy](http://www.numpy.org/) (>=1.16.0)
- [numba](http://numba.pydata.org/numba-doc/latest/user/5minguide.html)
- [scipy](https://www.scipy.org/)
- [natsort](https://natsort.readthedocs.io/en/master/)

### Instructions

If you have an older `cellpose` environment you can remove it with `conda env remove -n cellpose` before creating a new one.

If you are using a GPU, make sure its drivers and the cuda libraries are correctly installed.

1. Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python -- Choose **Python 3.8** and your operating system. Note you might need to use an anaconda prompt if you did not add anaconda to the path.
2. Open an anaconda prompt / command prompt with `conda` for **python 3** in the path
3. Create a new environment with `conda create --name cellpose python=3.8`.
4. To activate this new environment, run `conda activate cellpose`
5. To install the minimal version of cellpose, run `python -m pip install cellpose`.  
6. To install cellpose and the GUI, run `python -m pip install cellpose[gui]`. If you're on a zsh server, you may need to use ' ' around the cellpose[gui] call: `python -m pip install 'cellpose[gui]'.

To upgrade cellpose (package [here](https://pypi.org/project/cellpose/)), run the following in the environment:

~~~sh
python -m pip install cellpose --upgrade
~~~

Note you will always have to run `conda activate cellpose` before you run cellpose. If you want to run jupyter notebooks in this environment, then also `conda install jupyter` and `python -m pip install matplotlib`.

You can also try to install cellpose and the GUI dependencies from your base environment using the command

~~~~sh
python -m pip install cellpose[gui]
~~~~

If you have **issues** with installation, see the [docs](https://cellpose.readthedocs.io/en/latest/installation.html) for more details. You can also use the cellpose environment file included in the repository and create a cellpose environment with `conda env create -f environment.yml` which may solve certain dependency issues.

If these suggestions fail, open an issue.

### GPU version (CUDA) on Windows or Linux

If you plan on running many images, you may want to install a GPU version of *torch* (if it isn't already installed).

Before installing the GPU version, remove the CPU version:
~~~
pip uninstall torch
~~~

Follow the instructions [here](https://pytorch.org/get-started/locally/) to determine what version to install. The Anaconda install is strongly recommended, and then choose the CUDA version that is supported by your GPU (newer GPUs may need newer CUDA versions > 10.2). For instance this command will install the 11.3 version on Linux and Windows (note the `torchvision` and `torchaudio` commands are removed because cellpose doesn't require them):

~~~
conda install pytorch cudatoolkit=11.3 -c pytorch
~~~~

### Installation of github version

Follow steps from above to install the dependencies. Then run 
~~~
pip install git+https://www.github.com/mouseland/cellpose.git
~~~

If you want edit ability to the code, in the github repository folder, run `pip install -e .`. If you want to go back to the pip version of cellpose, then say `pip install cellpose`.
