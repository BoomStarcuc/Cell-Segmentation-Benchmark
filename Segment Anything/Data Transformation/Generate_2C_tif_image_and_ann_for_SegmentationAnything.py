import matplotlib.pyplot as plt
from matplotlib.path import Path
import cv2
import glob
import csv
from natsort import natsorted
import os
import sys
import h5py
import pickle
import numpy as np
import scipy.io as scio
from PIL import Image
import fastremap
import colorsys
from scipy.ndimage import find_objects
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from pycocotools import _mask as coco_mask
import zlib
import base64
import mmcv
import tifffile as tiff

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

def visualization_seg(annotation_n_file, annotation_w_file, image_dir, out_dir_images, out_dir_ann_n, out_dir_ann_w):
    print("annotation_n_file:", annotation_n_file)
    print("annotation_w_file:", annotation_w_file)

    coco_n = COCO(annotation_n_file)
    catIds_n = coco_n.getCatIds(catNms=['str']) # 获取指定类别 id
    imgIds_n = coco_n.getImgIds(catIds=catIds_n) # 获取图片id
    print("imgIds_n:", len(imgIds_n))

    coco_w = COCO(annotation_w_file)
    catIds_w = coco_w.getCatIds(catNms=['str']) # 获取指定类别 id
    imgIds_w = coco_w.getImgIds(catIds=catIds_w) # 获取图片id
    print("imgIds_w:", len(imgIds_w))
    print(imgIds_n == imgIds_w)
    assert imgIds_n == imgIds_w

    for imgId in imgIds_n:
        img_n = coco_n.loadImgs(imgId)[0]
        img_w = coco_w.loadImgs(imgId)[0]
        print(img_n['id'], img_w['id'], img_n['file_name'], img_w['file_name'])
        if img_n['id'] == img_w['id'] and img_n['file_name'] == img_w['file_name']:
            img_file = "{}/{}".format(image_dir, img_n['file_name'])

            image =  tiff.imread(img_file)
            # image = image[:,:,0] + image[:,:,1]
            # # print("image1", image.max(), image.min())
            # image = (image/image.max())*255
            # # print("image2", image.max(), image.min())
            # image = np.repeat(image[:,:, np.newaxis], 3, axis=2)
            # image = np.array(image, dtype='int32')
            # print("image3", image.shape, image.max(), image.min())

            annIds_n = coco_n.getAnnIds(imgIds=img_n['id'], catIds=catIds_n, iscrowd=None)
            anns_n = coco_n.loadAnns(annIds_n)

            annIds_w = coco_w.getAnnIds(imgIds=img_w['id'], catIds=catIds_w, iscrowd=None)
            anns_w = coco_w.loadAnns(annIds_w)
            
            label_n = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)
            label_w = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)

            for c, ann in enumerate(anns_n):
                if 'segmentation' in ann:
                    binarymask = mask_utils.decode(ann['segmentation'])
                    # print("binarymask:", binarymask.shape, binarymask.max(), binarymask.min())
                    label_n += binarymask*(c+1)
            
            for c, ann in enumerate(anns_w):
                if 'segmentation' in ann:
                    binarymask = mask_utils.decode(ann['segmentation'])
                    # print("binarymask:", binarymask.shape, binarymask.max(), binarymask.min())
                    label_w += binarymask*(c+1)

            # cv2.imwrite("{}/{}.png".format(out_dir_images, img_n['file_name'][:-4]), image)
            tiff.imsave("{}/{}.png".format(out_dir_images, img_n['file_name'][:-4]), image)
            #-------------------------------------------------------------------------------------------------------------------
            # label_n = fastremap.renumber(label_n, in_place=True)[0]
            # # print("label_n:", np.unique(label_n))
            # outlines = masks_to_outlines(label_n)
            # outy, outx = np.nonzero(outlines)
            # plt.scatter(outx, outy, marker = ',', c='r', lw=0, s=5)
            # plt.imshow(image)
            # plt.axis("off")
            # plt.savefig("{}_vis/{}_n_GT.jpg".format(out_dir_ann_n, img_n['file_name'][:-4]), bbox_inches='tight', pad_inches = 0)
            # plt.clf()

            # im_n = Image.fromarray(label_n)
            # im_path = os.path.join(out_dir_ann_n, img_n['file_name'][:-4] + '_GT.png')
            # im_n.save(im_path)
            # #----------------------------------------------------------------------------------------------------------------------
            # label_w = fastremap.renumber(label_w, in_place=True)[0]
            # # print("label_w:", np.unique(label_w))
            # outlines = masks_to_outlines(label_w)
            # outy, outx = np.nonzero(outlines)
            # plt.scatter(outx, outy, marker = ',', c='r', lw=0, s=5)
            # plt.imshow(image)
            # plt.axis("off")
            # plt.savefig("{}_vis/{}_n_GT.jpg".format(out_dir_ann_w, img_w['file_name'][:-4]), bbox_inches='tight', pad_inches = 0)
            # plt.clf()

            # im_w = Image.fromarray(label_w)
            # im_path = os.path.join(out_dir_ann_w, img_w['file_name'][:-4] + '_GT.png')
            # im_w.save(im_path)

if __name__ == '__main__':
    root_dir = "/shared/rc/spl/hx5239_homedir/hx5239/data/val_merge/COCO_TissueNet_2Channel"
    dataset_type = 'test'
    image_dir = "/shared/rc/spl/hx5239_homedir/hx5239/data/val_merge/COCO_TissueNet_2Channel/{}".format(dataset_type)
    annotation_dir = "/shared/rc/spl/hx5239_homedir/hx5239/data/val_merge/COCO_TissueNet_2Channel"

    annotation_n_file = "{}/tissuenet_nuclear_all_{}_2C.json".format(annotation_dir, dataset_type)
    annotation_w_file = "{}/tissuenet_wholecell_all_{}_2C.json".format(annotation_dir, dataset_type)
    out_dir_images = "{}/{}_SAM/images_tif".format(root_dir, dataset_type)
    out_dir_ann_n = "{}/{}_SAM/labels_n".format(root_dir, dataset_type)
    out_dir_ann_w = "{}/{}_SAM/labels_w".format(root_dir, dataset_type)

    visualization_seg(annotation_n_file, annotation_w_file, image_dir, out_dir_images, out_dir_ann_n, out_dir_ann_w)