Installation
------------------------------

For basic install instructions, look up the main github readme. 

Built-in model directory
~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the pretrained cellpose models are downloaded to ``$HOME/.cellpose/models/``.
This path on linux would look like ``/home/USERNAME/.cellpose/``, and on Windows, 
``C:/Users/USERNAME/.cellpose/models/``. These models are downloaded the first time you 
try to use them, either on the command line, in the GUI or in a notebook.

If you'd like to download the models to a different directory, 
and are using the command line or the GUI, before you run ``python -m cellpose ...``, 
you will need to always set the environment variable ``CELLPOSE_LOCAL_MODELS_PATH`` 
(thanks Chris Roat for implementing this!).

To set the environment variable in the command line/Anaconda prompt on windows run the following command modified for your path:
``set CELLPOSE_LOCAL_MODELS_PATH=C:/PATH_FOR_MODELS/``. To set the environment variable in the command line on 
linux, run ``export CELLPOSE_LOCAL_MODELS_PATH=/PATH_FOR_MODELS/``.

To set this environment variable when running cellpose in a jupyter notebook, run 
this code at the beginning of your notebook before you import cellpose:

::
   
   import os 
   os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = "/PATH_FOR_MODELS/"

M1 Mac installation
~~~~~~~~~~~~~~~~~~~~~~~

Please use the instructions provided on `image.sc <https://forum.image.sc/t/cellpose-on-macos-m1-pro-apple-silicon-arm64/68018/4>` 
by Peter Sobolewski. 

Common issues
~~~~~~~~~~~~~~~~~~~~~~~

If you are having issues with CUDA on Windows, or want to use 
Cuda Toolkit 10, please follow these `instructions <https://github.com/MouseLand/cellpose/issues/481#issuecomment-1080137885>`_:

::
   
   conda create -n cellpose pytorch=1.8.2 cudatoolkit=10.2 -c pytorch-lts
   conda activate cellpose
   pip install cellpose

If you receive the error: ``No module named PyQt5.sip``, then try
uninstalling and reinstalling pyqt5

::

   pip uninstall pyqt5 pyqt5-tools
   pip install pyqt5 pyqt5-tools pyqt5.sip

If you have errors related to OpenMP and libiomp5, then try 

::
   conda install nomkl

If you receive an error associated with **matplotlib**, try upgrading
it:

::

   pip install matplotlib --upgrade

If you receive the error: ``ImportError: _arpack DLL load failed``, then try uninstalling and reinstalling scipy
::

   pip uninstall scipy
   pip install scipy

If you are having issues with the graphical interface, make sure you have **python 3.7** and not python 3.8 installed.

If you are on Yosemite Mac OS or earlier, PyQt doesn't work and you won't be able
to use the graphical interface for cellpose. More recent versions of Mac
OS are fine. The software has been heavily tested on Windows 10 and
Ubuntu 18.04, and less well tested on Mac OS. Please post an issue if
you have installation problems.


Dependencies
~~~~~~~~~~~~~~~~~~~~~~

cellpose relies on the following excellent packages (which are
automatically installed with conda/pip if missing):

-  `pytorch`_
-  `pyqtgraph`_
-  `PyQt5`_
-  `numpy`_ (>=1.16.0)
-  `numba`_
-  `scipy`_
-  `scikit-image`_
-  `natsort`_
-  `matplotlib`_

.. _Anaconda: https://www.anaconda.com/download/
.. _environment.yml: https://github.com/MouseLand/cellpose/blob/master/environment.yml?raw=true
.. _here: https://pypi.org/project/cellpose/

.. _pytorch: https://pytorch.org/
.. _pyqtgraph: http://pyqtgraph.org/
.. _PyQt5: http://pyqt.sourceforge.net/Docs/PyQt5/
.. _numpy: http://www.numpy.org/
.. _numba: http://numba.pydata.org/numba-doc/latest/user/5minguide.html
.. _scipy: https://www.scipy.org/
.. _scikit-image: https://scikit-image.org/
.. _natsort: https://natsort.readthedocs.io/en/master/
.. _matplotlib: https://matplotlib.org/
