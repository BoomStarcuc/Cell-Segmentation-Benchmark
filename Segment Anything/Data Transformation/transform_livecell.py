import os
import numpy as np
import tifffile
import random
import cv2
from PIL import Image

data_dir = 'F:/Bechmark_cellsegmentation/data' 
samples_set = "test"
test_dict = np.load(os.path.join(data_dir, 'LIVECell_{}_all.npz'.format(samples_set)), allow_pickle=True)

#nuclear
tiff_dir = '/SAM_livecell/{}'.format(samples_set)
if not os.path.isdir(tiff_dir):
    os.makedirs(tiff_dir)

#nuclear
X_test = test_dict['X']
y_test = test_dict['y']
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

tissue_type = test_dict['tissue_list']

X = X_test.squeeze()
y = y_test.astype('int32').squeeze()
print(X.shape, y.shape)
print(np.unique(tissue_type))

assert X.shape[0] == y.shape[0], 'X and y should have the same number of images.'

tissues = np.unique(tissue_type)
for tissue in tissues:
    X_tissue = X[tissue_type==tissue]
    y_tissue = y[tissue_type==tissue]

    img_dir = '{}/images/{}'.format(tiff_dir, tissue)
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    
    lbl_dir = '{}/labels/{}'.format(tiff_dir, tissue)
    if not os.path.isdir(lbl_dir):
        os.makedirs(lbl_dir)

    for i in range(len(X_tissue)):
        img_path = '{}/{}_{:06d}.png'.format(img_dir, tissue, i)
        mask_filename = '{}/{}_{:06d}.png'.format(lbl_dir, tissue, i)

        img = (X_tissue[i] - X_tissue[i].min())/(X_tissue[i].max() - X_tissue[i].min())
        img = np.asarray(img)*255
        cv2.imwrite(img_path, img.astype(np.uint8))

        mask = Image.fromarray(y_tissue[i])
        mask_path = mask_filename
        mask.save(mask_path)

print('saved %s files to %s' % (len(X), tiff_dir))