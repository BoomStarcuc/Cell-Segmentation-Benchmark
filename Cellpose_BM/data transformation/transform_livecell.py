import os
import numpy as np
import tifffile
import random
import cv2

data_dir = '/path/to/your/data/dir'
samples_set = "val" #train, val, or test
test_dict = np.load(os.path.join(data_dir, 'LIVECell_{}_all.npz'.format(samples_set)), allow_pickle=True)

#nuclear
tiff_dir = '/cellpose_livecell/{}'.format(samples_set)
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

assert X.shape[0] == y.shape[0], 'X and y should have the same number of images.'

for i in range(len(X)):
    img_filename = '{:06d}_img.tif'.format(i)
    mask_filename = '{:06d}_masks.tif'.format(i)
    
    img = (X[i] - X[i].min())/(X[i].max() - X[i].min())
    tifffile.imwrite(os.path.join(tiff_dir, img_filename), img*255)
    tifffile.imwrite(os.path.join(tiff_dir, mask_filename), y[i])

print('saved %s files to %s' % (len(X), tiff_dir))
