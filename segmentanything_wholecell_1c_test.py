import os
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import cv2
from skimage.measure import label
from scipy.ndimage import find_objects
import matplotlib.pyplot as plt
from natsort import natsorted
import glob
import sys
import fastremap
from tqdm import tqdm
import pandas as pd
from PIL import Image
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def rgb_to_hsv(arr):
    rgb_to_hsv_channels = np.vectorize(colorsys.rgb_to_hsv)
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv_channels(r, g, b)
    hsv = np.stack((h,s,v), axis=-1)
    return hsv

def hsv_to_rgb(arr):
    hsv_to_rgb_channels = np.vectorize(colorsys.hsv_to_rgb)
    h, s, v = np.rollaxis(arr, axis=-1)
    r, g, b = hsv_to_rgb_channels(h, s, v)
    rgb = np.stack((r,g,b), axis=-1)
    return rgb

def masks_to_outlines(masks):
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array'%masks.ndim)
    outlines = np.zeros(masks.shape, bool)
    
    if masks.ndim==3:
        for i in range(masks.shape[0]):
            outlines[i] = masks_to_outlines(masks[i])
        return outlines
    else:
        slices = find_objects(masks.astype(int))
        for i,si in enumerate(slices):
            if si is not None:
                sr,sc = si
                mask = (masks[sr, sc] == (i+1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T            
                vr, vc = pvr + sr.start, pvc + sc.start 
                outlines[vr, vc] = 1
        return outlines

def mask_overlay(img, masks, colors=None):
    """ overlay masks on image (set image to grayscale)

    Parameters
    ----------------

    img: int or float, 2D or 3D array
        img is of size [Ly x Lx (x nchan)]

    masks: int, 2D array
        masks where 0=NO masks; 1,2,...=mask labels

    colors: int, 2D array (optional, default None)
        size [nmasks x 3], each entry is a color in 0-255 range

    Returns
    ----------------

    RGB: uint8, 3D array
        array of masks overlaid on grayscale image

    """
    if colors is not None:
        if colors.max()>1:
            colors = np.float32(colors)
            colors /= 255
        colors = rgb_to_hsv(colors)
    if img.ndim>2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)
    np.random.seed(5000)
    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip((img / 255. if img.max() > 1 else img) * 1.5, 0, 1)
    hues = np.linspace(0, 1, masks.max()+1)[np.random.permutation(masks.max())]
    for n in range(int(masks.max())):
        ipix = (masks==n+1).nonzero()
        if colors is None:
            HSV[ipix[0],ipix[1],0] = hues[n]
        else:
            HSV[ipix[0],ipix[1],0] = colors[n,0]
        HSV[ipix[0],ipix[1],1] = 0.7
    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB

def normalize99(Y, lower=1,upper=99):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01)
    return X

def image_to_rgb(img0, channels=[0,0]):
    """ image is 2 x Ly x Lx or Ly x Lx x 2 - change to RGB Ly x Lx x 3 """
    img = img0.copy()
    img = img.astype(np.float32)
    if img.ndim<3:
        img = img[:,:,np.newaxis]
    if img.shape[0]<5:
        img = np.transpose(img, (1,2,0))
    if channels[0]==0:
        img = img.mean(axis=-1)[:,:,np.newaxis]
    for i in range(img.shape[-1]):
        if np.ptp(img[:,:,i])>0:
            img[:,:,i] = np.clip(normalize99(img[:,:,i]), 0, 1)
            img[:,:,i] = np.clip(img[:,:,i], 0, 1)
    img *= 255
    img = np.uint8(img)
    RGB = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    if img.shape[-1]==1:
        RGB = np.tile(img,(1,1,3))
    else:
        RGB[:,:,channels[0]-1] = img[:,:,0]
        if channels[1] > 0:
            RGB[:,:,channels[1]-1] = img[:,:,1]
    return RGB

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
    return ap, tp, fp, fn, n_pred[0]


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
#     x = (utils.format_labels(x)).ravel()
#     y = (utils.format_labels(y)).ravel()
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


def normalize(x, pmin=0.0, pmax=100.0, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x


def intersect_matrices(mat1, mat2):
    if not (mat1.shape == mat2.shape):
        return False

    mat_intersect = np.where((mat1 == mat2), mat1, 0)
    return mat_intersect




sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

tissues = ["Colon", "lymph node metastasis", "Spleen", "Pancreas", "Epidermis", "Breast", "Lymph Node", "Tonsil", "Lung", "Esophagus"]
data_dir = '/path/to/your/test/data/dir'
save_dir = '/path/to/your/save/dir'
# for tissue in tissues:
image_file_list = []
label_file_list = []
tissue_list = []
for tissue_idx ,tissue in enumerate(tissues):
    image_files = natsorted(glob.glob(os.path.join('{}/{}/{}'.format(data_dir, 'images', tissue), '*.png')))
    label_files = natsorted(glob.glob(os.path.join('{}/{}/{}'.format(data_dir, 'labels', tissue), '*.png')))
    image_file_list.extend(image_files)
    label_file_list.extend(label_files)
    tissue_list.extend(np.repeat([tissue], len(label_files)))

print("image_file_list:", len(image_file_list))
print("label_file_list:", len(label_file_list))
print("tissue_list:", len(tissue_list))

label_list = []
pred_list = []
# y_pred = np.empty([len(label_file_list), 256, 256], dtype='uint8')
for i, (image_file, label_file, tissue_name) in enumerate(zip(image_file_list, label_file_list, tissue_list)):
    print("image_file:", image_file)
    print("label_file:", label_file)
    image = cv2.imread(image_file, 0)
    label = cv2.imread(label_file, -1)
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # img = image_to_rgb(image)
    filename = os.path.basename(image_file)
    print("filename:", filename)
    print("------------------------------------")

    
    sam_masks = mask_generator.generate(img)

    # print(len(sam_masks))
    filtered_dict_list = []
    sorted_masks = sorted(sam_masks, key=(lambda x: x['area']), reverse=True)
    for d in sorted_masks:
        if d['area'] < 10000:
            filtered_dict_list.append(d)
    height, width = 256, 256
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    identity_combined_mask = np.zeros((height, width), dtype=np.uint8)

    segmentations = filtered_dict_list  # Replace this with your list of segmentations

    label_i = 1
    for seg in segmentations:
        current_mask = seg['segmentation'].astype(np.uint8)
    #     mask_without_overlap = apply_mask_without_overlap(current_mask, combined_mask, label)
        if not np.any(combined_mask):
            combined_mask += current_mask
            identity_combined_mask += current_mask
        if np.any(combined_mask):
            if not np.any(intersect_matrices(current_mask, identity_combined_mask)):
                identity_combined_mask += current_mask
                combined_mask += current_mask * int(label_i)
                label_i += 1
                continue
            if np.any(intersect_matrices(current_mask, identity_combined_mask)):
                intersec = current_mask & identity_combined_mask
                cur_area = len(np.where(current_mask == 1)[0])
                gt_area = len(np.where(identity_combined_mask == 1)[0])
                inter_area = len(np.where(intersec == 1)[0])
                cor = np.where(intersec == 1)
                if (inter_area / cur_area) >= 0.7:

                    identity_combined_mask[cor] = 0
                    combined_mask[cor] = 0
                    identity_combined_mask += current_mask
                    combined_mask += current_mask * int(label_i)
                else:
                    current_mask[cor] = 0
                    identity_combined_mask += current_mask
                    combined_mask += current_mask * int(label_i)
                
                label_i += 1

    pred_list.append(combined_mask)
    label_list.append(label)

print("pred_list:", np.array(pred_list).shape, np.array(pred_list).dtype)
print("label_list:", np.array(label_list).shape, np.array(label_list).dtype)
preds = np.array(pred_list)
labels = np.array(label_list)
tissue_list = np.array(tissue_list)

unique_list = ['Breast', 'Colon', 'Epidermis', 'Esophagus', 'Lung', 'lymph node metastasis', 'Lymph Node', 'Pancreas', 'Spleen', 'Tonsil', 'All']
output_data = np.empty([11, 5])
for tissue_idx ,tissue in enumerate(unique_list):
    if tissue == "All":
        idx = [k for k in range(len(tissue_list))]
    else:
        idx = np.where(tissue_list==tissue)[0]
    print(tissue)
    for count, value in tqdm(enumerate(idx)):
        if count ==0:
            ap, tp, fp, fn, num_pred= average_precision(labels[value], preds[value], threshold=[0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95])
        else:
            ap2, tp2, fp2, fn2, num_pred2= average_precision(labels[value], preds[value], threshold=[0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95])
            tp+=tp2
            fp+=fp2
            fn+=fn2
    output_data[tissue_idx] = np.array([np.average(tp/(tp+fn+fp)), (tp/(tp+fn+fp))[0], (tp/(tp+fn+fp))[5], (tp/(tp+fn))[0], (tp/(tp+fn))[5]])
    
print(output_data)
data_df = pd.DataFrame(output_data)
data_df.columns = ['mAP','AP50','AP75','Recall50','Recall75']
data_df.index = unique_list
writer = pd.ExcelWriter('{}/segment_wholecell_1C.xlsx'.format(save_dir))
data_df.to_excel(writer,'page_1',float_format='%.5f')
writer.save()
