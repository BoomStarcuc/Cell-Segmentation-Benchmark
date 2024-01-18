import argparse
import colorsys
import cv2
import fastremap
import numpy as np
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import find_objects
from scipy.optimize import linear_sum_assignment
from skimage import morphology
from skimage.measure import label
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import SGD
from tensorflow.python.data import Dataset
from tqdm import tqdm

from deepcell import image_generators, losses, model_zoo
from deepcell.image_generators import ImageFullyConvDataGenerator
from deepcell.utils import train_utils, tracking_utils
from deepcell.utils.train_utils import count_gpus, get_callbacks, rate_scheduler

def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """ average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Parameters
    ------------
    
    masks_true: list of ND-arrays (int) or ND-array (int) 
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int) 
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    ap: array [len(masks_true) x len(threshold)]
        average precision at thresholds
    tp: array [len(masks_true) x len(threshold)]
        number of true positives at thresholds
    fp: array [len(masks_true) x len(threshold)]
        number of false positives at thresholds
    fn: array [len(masks_true) x len(threshold)]
        number of false negatives at thresholds

    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]
    
    if len(masks_true) != len(masks_pred):
        raise ValueError('metrics.average_precision requires len(masks_true)==len(masks_pred)')

    ap  = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn  = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))
    
    for n in range(len(masks_true)):
        #_,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k,th in enumerate(threshold):
                tp[n,k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])  
        
    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    return ap, tp, fp, fn


def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs
    
    Parameters
    ------------
    
    masks_true: ND-array, int 
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]
    
    ------------
    How it works:
        The overlap matrix is a lookup table of the area of intersection
        between each set of labels (true and predicted). The true labels
        are taken to be along axis 0, and the predicted labels are taken 
        to be along axis 1. The sum of the overlaps along axis 0 is thus
        an array giving the total overlap of the true labels with each of
        the predicted labels, and likewise the sum over axis 1 is the
        total overlap of the predicted labels with each of the true labels.
        Because the label 0 (background) is included, this sum is guaranteed
        to reconstruct the total area of each label. Adding this row and
        column vectors gives a 2D array with the areas of every label pair
        added together. This is equivalent to the union of the label areas
        except for the duplicated overlap area, so the overlap matrix is
        subtracted to find the union matrix. 

    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y 
    
    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    
    """
    # put label arrays into standard form then flatten them 
    x = x.ravel()
    y = y.ravel()
    
    # preallocate a 'contact map' matrix
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    
    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image 
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap


def _true_positive(iou, th):
    """ true positive at threshold th
    
    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold
        
    ------------
    How it works:
        (1) Find minimum number of masks
        (2) Define cost matrix; for a given threshold, each element is negative
            the higher the IoU is (perfect IoU is 1, worst is 0). The second term
            gets more negative with higher IoU, but less negative with greater
            n_min (but that's a constant...)
        (3) Solve the linear sum assignment problem. The costs array defines the cost
            of matching a true label with a predicted label, so the problem is to 
            find the set of pairings that minimizes this cost. The scipy.optimize
            function gives the ordered lists of corresponding true and predicted labels. 
        (4) Extract the IoUs fro these parings and then threshold to get a boolean array
            whose sum is the number of true positives that is returned. 

    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
#     print("true_ind", true_ind, "pred_ind", pred_ind)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch FeatureNet Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-dir', type=str, default='path/to/your/dataset/directory', 
                        help='path to the data')
    parser.add_argument('--exp-name', type=str, default='featurenet_train_on_all_types_wholecell_1C', 
                        help='name of the experiment')
    parser.add_argument('--model-name', type=str, default='featurenet', 
                        help='name of the experiment')
    parser.add_argument('--log-dir', type=str, default='./featurenet_logs', 
                        help='where you want to put your logs')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Start training with existing weights')
    parser.add_argument('--chooseone', default='All',
                        help='choose from Tonsil Breast BV2 Epidermis SHSY5Y Esophagus \
                        A172 SkBr3 Lymph_Node Lung SKOV3 Pancreas Colon MCF7 Huh7 BT474 lymph node metastasis Spleen')
    parser.add_argument('--excelname', type=str, default='featurenet_wholecell_1C', help='save masks as tif and outlines as text file for ImageJ')           
    args = parser.parse_args()

    # create folder for this set of experiments
    experiment_folder = args.exp_name
    MODEL_DIR = os.path.join("./FeatureNet", experiment_folder)

    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print('loading test data')
    data_dir = args.data_dir
    test_dict = np.load(os.path.join(data_dir, 'tissuenet_test_split_256x256_memserpreprocess.npz'), allow_pickle=True)

    X_test = test_dict['X'][...,1].squeeze()
    y_test = test_dict['y'][...,0].squeeze()
    y_test = y_test.astype('int32').squeeze()
    tissue_list = test_dict['tissue_list']
    filename_list = test_dict['filenames']

    norm_method = 'std'  # data normalization
    receptive_field = 61  # should be adjusted for the scale of the data
    n_skips = 1  # number of skip-connections (only for FC training)
    dilation_radius = 1  # change dilation radius for edge dilation
    separate_edge_classes = True  # break edges into cell-background edge, cell-cell edge
    pixelwise_kwargs = {
        'dilation_radius': dilation_radius,
        'separate_edge_classes': separate_edge_classes,
    }

    fgbg_model = model_zoo.bn_feature_net_skip_2D(
        n_features=2,  # segmentation mask (is_cell, is_not_cell)
        receptive_field=receptive_field,
        norm_method=norm_method,
        n_skips=n_skips,
        n_conv_filters=32,
        n_dense_filters=128,
        input_shape=tuple(X_test.shape[1:]),
        last_only=False)

    fgbg_model_name = 'conv_fgbg_model'
    pixelwise_model_name = 'conv_edgeseg_model'

    fgbg_model.load_weights(os.path.join(MODEL_DIR, '{}.h5'.format(fgbg_model_name)))

    # Create the pixelwise FeatureNet Model
    pixelwise_model = model_zoo.bn_feature_net_skip_2D(
        fgbg_model=fgbg_model,
        n_features=4 if separate_edge_classes else 3,
        receptive_field=receptive_field,
        norm_method=norm_method,
        n_skips=n_skips,
        n_conv_filters=32,
        n_dense_filters=128,
        last_only=False,
        input_shape=tuple(X_test.shape[1:]))

    pixelwise_model.load_weights(os.path.join(MODEL_DIR, '{}.h5'.format(pixelwise_model_name)))

    print("pixelwise predicting")    
    test_images = pixelwise_model.predict(X_test)[-1]
    print("fgbg predicting")
    test_images_fgbg = fgbg_model.predict(X_test)[-1]

    X_test = X_test[..., np.newaxis]
    y_test = y_test[..., np.newaxis]

    print('watershed transform shape:', test_images.shape)
    print('segmentation mask shape:', test_images_fgbg.shape)

    # threshold the foreground/background
    # and remove back ground from edge transform
    threshold = 0.9

    fg_thresh = test_images_fgbg[..., 1] > threshold
    fg_thresh = np.expand_dims(fg_thresh, axis=-1)

    test_images_post_fgbg = test_images * fg_thresh

    labeled_images = []
    for i in range(test_images_post_fgbg.shape[0]):
        interior = test_images_post_fgbg[i, ..., 2] > .2
        labeled_image = label(interior)
        labeled_image = morphology.remove_small_objects(
            labeled_image, min_size=50, connectivity=1)
        labeled_images.append(labeled_image)
    labeled_images = np.array(labeled_images)
    labeled_images = np.expand_dims(labeled_images, axis=-1)
    print('labeled_images shape:', labeled_images.shape)

    unique_list = ['Breast', 'Colon', 'Epidermis', 'Esophagus', 'Lung', 'lymph node metastasis', 'Lymph Node', 'Pancreas', 'Spleen', 'Tonsil', 'All']
    output_data = np.empty([11, 5])
    for tissue_idx ,tissue in enumerate(unique_list):
        if tissue == "All":
            idx = [k for k in range(len(tissue_list))]
        else:
            idx = np.where(tissue_list==tissue)[0]
        print(tissue)
            
        for count, value in enumerate(idx):
            if count ==0:
                ap, tp, fp, fn= average_precision(y_test[value], labeled_images[value], threshold=[0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95])
            else:
                ap2, tp2, fp2, fn2= average_precision(y_test[value], labeled_images[value], threshold=[0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95])
                tp+=tp2
                fp+=fp2
                fn+=fn2
        output_data[tissue_idx] = np.array([np.average(tp/(tp+fn+fp)), (tp/(tp+fn+fp))[0], (tp/(tp+fn+fp))[5], (tp/(tp+fn))[0], (tp/(tp+fn))[5]])
    # print(output_data)
    data_df = pd.DataFrame(output_data)
    data_df.columns = ['mAP','AP50','AP75','Recall50','Recall75']
    data_df.index = unique_list
    writer = pd.ExcelWriter(args.excelname+'.xlsx')
    data_df.to_excel(writer,'page_1',float_format='%.5f')
    writer.save()

if __name__ == '__main__':
    main()