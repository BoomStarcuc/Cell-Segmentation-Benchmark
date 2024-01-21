# Swin-S, Swin-T, Cascade Mask RCNN seesaw, SOLOv2, Res2Net, RF-Next, HRNet, Mask2former, Mask RCNN, and MS RCNN

## Data transformation
Since the structure of the datasets provided does not satisfy the format of the training these methods, you need to run the following code from the data transformation directory

1. **Creating TissueNet single-channel data**

   - **Step 1.** Generating images and annotations for all tissues
  
     ```python transform_tissuenet_1C.py```
  
     Note: ```sample_type``` needs to be modified, including train, val, and test.

   - **Step 2.** Splitting all tissues into individual tissue in the annotation file
  
     ```python tissuenet_split_annotation.py```
  
     Note: ```sample_type```, ```channel```, and ```sample``` needs to be modified.


2. **Creating TissueNet dual-channel data**
   
   - **Step 1.** Generating images and annotations for all tissues
  
     ```python transform_tissuenet_2C.py```
  
     Note: ```sample_type``` needs to be modified, including train, val, and test.
  
   - **Step 2.** Splitting all tissues into individual tissue in the annotation file
  
     ```python tissuenet_split_annotation.py```
  
     Note: ```sample_type```, ```channel```, and ```sample``` needs to be modified.


3. **Creating livecell data**

   - **Step 1.** Generating images and annotations for all cell types
  
     ```python transform_livecell.py```
  
     Note: ```sample_type``` needs to be modified, including train, val, and test.
  
   - **Step 2.** Splitting all cell types into individual cell type in the annotation file
  
     ```python livecell_split_annotation.py```
  
     Note: ```sample_type``` needs to be modified.


## Installation

### Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch.

MMDetection works on Linux, Windows and macOS. It requires Python 3.7+, CUDA 9.2+ and PyTorch 1.5+.

```{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](#installation). Otherwise, you can follow these steps for the preparation.
```

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

### Installation

We recommend that users follow our best practices to install MMDetection. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

#### Best Practices

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmcv-full
```

**Step 1.** Install MMDetection.

Case a: If you develop and run mmdet directly, install it from source:

```shell
git clone https://github.com/BoomStarcuc/Cell-Segmentation-Benchmark.git
cd mmdet_BM/mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

#### Verify the installation

To verify whether MMDetection is installed correctly, we provide some sample codes to run an inference demo.

**Step 1.** We need to download config and checkpoint files.

```shell
mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
```

The downloading will take several seconds or more, depending on your network environment. When it is done, you will find two files `yolov3_mobilenetv2_320_300e_coco.py` and `yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth` in your current folder.

**Step 2.** Verify the inference demo.

Option (a). If you install mmdetection from source, just run the following command.

```shell
python demo/image_demo.py demo/demo.jpg yolov3_mobilenetv2_320_300e_coco.py yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth --device cpu --out-file result.jpg
```

You will see a new image `result.jpg` on your current folder, where bounding boxes are plotted on cars, benches, etc.

## Methods

Before starting the training with the methods listed below, please update the ```data_root``` path in the files ```coco_livecell_instance.py```, ```coco_tissuenet_n_instance.py```, ```coco_tissuenet_n_instance_2C.py```, ```coco_tissuenet_w_instance.py```, and ```coco_tissuenet_w_instance_2C.py``` located in the ```_base_/datasets``` directory to match your specific dataset directory. Additionally, for each method used, ensure to modify the ```data_root``` path in the configs to reflect your specific dataset path.

## Swin-S

### Train from scratch

For single-channel data training:

```
cd configs/swin
 
sbatch swin-s-livecell.slurm

sbatch swin-s-tissuenet-n.slurm

sbatch swin-s-tissuenet-w.slurm
```

For dual-channel data training:

```
cd configs/swin2C
 
sbatch swin-s-tissuenet-n.slurm

sbatch swin-s-tissuenet-w.slurm
```

Note: All the ```#SBATCH``` configurations in the above ```.slurm``` files are based on my current server settings. You will need to modify these parameters according to the specific requirements of your server.

### Test on tissues

For single-channel data testing:

```
cd configs/swin

sh swin-s-livecell-train-All-test-on-tissues-submit.bash

sh swin-s-tissuenet-n-train-All-test-on-tissues-submit.bash

sh swin-s-tissuenet-w-train-All-test-on-tissues-submit.bash
```

For dual-channel data testing:

```
cd configs/swin2C
 
sh swin-s-tissuenet-n-train-All-test-on-tissues-submit.bash

sh swin-s-tissuenet-w-train-All-test-on-tissues-submit.bash
```

Note: Same as training, you need to modify ```#SBATCH``` configurations based on your server.

******************************************************************************************

### Train on one tissue

For single-channel data training:

```
cd configs/swin
 
sh swin-s-livecell-train-one-submit.bash

sh swin-s-tissuenet-n-train-one-submit.bash

sh swin-s-tissuenet-w-train-one-submit.bash
```

For dual-channel data training:

```
cd configs/swin2C
 
sh swin-s-tissuenet-n-train-one-submit.bash

sh swin-s-tissuenet-w-train-one-submit.bash
```

### Test on tissues

For single-channel data testing:

```
cd configs/swin

sh swin-s-livecell-train-one-test-on-tissues-submit.bash

sh swin-s-tissuenet-n-train-one-test-on-tissues-submit.bash

sh swin-s-tissuenet-w-train-one-test-on-tissues-submit.bash
```

For dual-channel data testing:

```
cd configs/swin2C
 
sh swin-s-tissuenet-n-train-one-test-on-tissues-submit.bash

sh swin-s-tissuenet-w-train-one-test-on-tissues-submit.bash
```

## Swin-T

### Train from scratch

For single-channel data training:

```
cd configs/swin
 
sbatch swin-t-livecell.slurm

sbatch swin-t-tissuenet-n.slurm

sbatch swin-t-tissuenet-w.slurm
```

For dual-channel data training:

```
cd configs/swin2C
 
sbatch swin-t-tissuenet-n.slurm

sbatch swin-t-tissuenet-w.slurm
```

### Test on tissues

For single-channel data testing:

```
cd configs/swin

sh swin-t-livecell-train-All-test-on-tissues-submit.bash

sh swin-t-tissuenet-n-train-All-test-on-tissues-submit.bash

sh swin-t-tissuenet-w-train-All-test-on-tissues-submit.bash
```

For dual-channel data testing:

```
cd configs/swin2C
 
sh swin-t-tissuenet-n-train-All-test-on-tissues-submit.bash

sh swin-t-tissuenet-w-train-All-test-on-tissues-submit.bash
```

## Cascade Mask RCNN seesaw

### Train from scratch

For single-channel data training:

```
cd configs/seesaw_loss
 
sbatch seesaw-livecell.slurm

sbatch seesaw-tissuenet-n.slurm

sbatch seesaw-tissuenet-w.slurm
```

For dual-channel data training:

```
cd configs/seesaw_loss2C
 
sbatch seesaw-tissuenet-n.slurm

sbatch seesaw-tissuenet-w.slurm
```

### Test on tissues

For single-channel data testing:

```
cd configs/seesaw_loss

sh seesaw-livecell-train-All-test-on-tissues-submit.bash

sh seesaw-tissuenet-n-train-All-test-on-tissues-submit.bash

sh seesaw-tissuenet-w-train-All-test-on-tissues-submit.bash
```

For dual-channel data testing:

```
cd configs/seesaw_loss2C
 
sh seesaw-tissuenet-n-train-All-test-on-tissues-submit.bash

sh seesaw-tissuenet-w-train-All-test-on-tissues-submit.bash
```

## SOLOv2

### Train from scratch

For single-channel data training:

```
cd configs/solov2
 
sbatch solov2-livecell.slurm

sbatch solov2-tissuenet-n.slurm

sbatch solov2-tissuenet-w.slurm
```

For dual-channel data training:

```
cd configs/solov22C
 
sbatch solov2-tissuenet-n.slurm

sbatch solov2-tissuenet-w.slurm
```

### Test on tissues

For single-channel data testing:

```
cd configs/solov2

sh solov2-livecell-train-All-test-on-tissues-submit.bash

sh solov2-tissuenet-n-train-All-test-on-tissues-submit.bash

sh solov2-tissuenet-w-train-All-test-on-tissues-submit.bash
```

For dual-channel data testing:

```
cd configs/solov22C
 
sh solov2-tissuenet-n-train-All-test-on-tissues-submit.bash

sh solov2-tissuenet-n-train-All-test-on-tissues-submit.bash
```

## Res2Net

### Train from scratch

For single-channel data training:

```
cd configs/res2net
 
sbatch res2net-livecell.slurm

sbatch res2net-tissuenet-n.slurm

sbatch res2net-tissuenet-w.slurm
```

For dual-channel data training:

```
cd configs/res2net2C
 
sbatch res2net-tissuenet-n.slurm

sbatch res2net-tissuenet-w.slurm
```

### Test on tissues

For single-channel data testing:

```
cd configs/res2net

sh res2net-livecell-train-All-test-on-tissues-submit.bash

sh res2net-tissuenet-n-train-All-test-on-tissues-submit.bash

sh res2net-tissuenet-w-train-All-test-on-tissues-submit.bash
```

For dual-channel data testing:

```
cd configs/res2net2C
 
sh res2net-tissuenet-n-train-All-test-on-tissues-submit.bash

sh res2net-tissuenet-w-train-All-test-on-tissues-submit.bash
```

## RF-Next

### Train from scratch

For single-channel data training:

```
cd configs/rfnext
 
sbatch rfnext-livecell.slurm

sbatch rfnext-tissuenet-n.slurm

sbatch rfnext-tissuenet-w.slurm
```

For dual-channel data training:

```
cd configs/rfnext2C
 
sbatch rfnext-tissuenet-n.slurm

sbatch rfnext-tissuenet-w.slurm
```

### Test on tissues

For single-channel data testing:

```
cd configs/rfnext

sh rfnext-livecell-train-All-test-on-tissues-submit.bash

sh rfnext-tissuenet-n-train-All-test-on-tissues-submit.bash

sh rfnext-tissuenet-w-train-All-test-on-tissues-submit.bash
```

For dual-channel data testing:

```
cd configs/rfnext2C
 
sh rfnext-tissuenet-n-train-All-test-on-tissues-submit.bash

sh rfnext-tissuenet-w-train-All-test-on-tissues-submit.bash
```

## HRNet

### Train from scratch

For single-channel data training:

```
cd configs/hrnet
 
sbatch hrnet-livecell.slurm

sbatch hrnet-tissuenet-n.slurm

sbatch hrnet-tissuenet-w.slurm
```

For dual-channel data training:

```
cd configs/hrnet2C
 
sbatch hrnet-tissuenet-n.slurm

sbatch hrnet-tissuenet-w.slurm
```

### Test on tissues

For single-channel data testing:

```
cd configs/hrnet

sh hrnet-livecell-train-All-test-on-tissues-submit.bash

sh hrnet-tissuenet-n-train-All-test-on-tissues-submit.bash

sh hrnet-tissuenet-w-train-All-test-on-tissues-submit.bash
```

For dual-channel data testing:

```
cd configs/hrnet2C
 
sh hrnet-tissuenet-n-train-All-test-on-tissues-submit.bash

sh hrnet-tissuenet-w-train-All-test-on-tissues-submit.bash
```

## Mask2former

### Train from scratch

For single-channel data training:

```
cd configs/mask2former
 
sbatch mask2former-livecell.slurm

sbatch mask2former-tissuenet-n.slurm

sbatch mask2former-tissuenet-w.slurm
```

For dual-channel data training:

```
cd configs/mask2former2C
 
sbatch mask2former-tissuenet-n.slurm

sbatch mask2former-tissuenet-w.slurm
```

### Test on tissues

For single-channel data testing:

```
cd configs/mask2former

sh mask2former-livecell-train-All-test-on-tissues-submit.bash

sh mask2former-tissuenet-n-train-All-test-on-tissues-submit.bash

sh mask2former-tissuenet-w-train-All-test-on-tissues-submit.bash
```

For dual-channel data testing:

```
cd configs/mask2former2C
 
sh mask2former-tissuenet-n-train-All-test-on-tissues-submit.bash

sh mask2former-tissuenet-w-train-All-test-on-tissues-submit.bash
```

## Mask RCNN

### Train from scratch

For single-channel data training:

```
cd configs/strong_baselines
 
sbatch mask-rcnn-fpn-2conv-livecell.slurm

sbatch mask-rcnn-fpn-2conv-tissuenet-n.slurm

sbatch mask-rcnn-fpn-2conv-tissuenet-w.slurm
```

For dual-channel data training:

```
cd configs/strong_baselines2C
 
sbatch mask-rcnn-fpn-2conv-tissuenet-n.slurm

sbatch mask-rcnn-fpn-2conv-tissuenet-w.slurm
```

### Test on tissues

For single-channel data testing:

```
cd configs/strong_baselines

sh MRF2conv-livecell-train-All-test-on-tissues-submit.bash

sh MRF2conv-tissuenet-n-train-All-test-on-tissues-submit.bash

sh MRF2conv-tissuenet-w-train-All-test-on-tissues-submit.bash
```

For dual-channel data testing:

```
cd configs/strong_baselines2C
 
sh MRF2conv-tissuenet-n-train-All-test-on-tissues-submit.bash

sh MRF2conv-tissuenet-w-train-All-test-on-tissues-submit.bash
```

## MS RCNN

### Train from scratch

For single-channel data training:

```
cd configs/ms_rcnn
 
sbatch msrcnn-livecell.slurm

sbatch msrcnn-tissuenet-n.slurm

sbatch msrcnn-tissuenet-w.slurm
```

For dual-channel data training:

```
cd configs/ms_rcnn2C
 
sbatch msrcnn-tissuenet-n.slurm

sbatch msrcnn-tissuenet-w.slurm
```

### Test on tissues

For single-channel data testing:

```
cd configs/ms_rcnn

sh ms_rcnn-livecell-train-All-test-on-tissues-submit.bash

sh ms_rcnn-tissuenet-n-train-All-test-on-tissues-submit.bash

sh ms_rcnn-tissuenet-w-train-All-test-on-tissues-submit.bash
```

For dual-channel data testing:

```
cd configs/ms_rcnn2C
 
sh ms_rcnn-tissuenet-n-train-All-test-on-tissues-submit.bash

sh ms_rcnn-tissuenet-w-train-All-test-on-tissues-submit.bash
```
