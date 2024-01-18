import os
import numpy as np
import tifffile
import random
from PIL import Image
import cv2

# File path to read
data_dir = '/path/to/your/data/dir'
is_single = True #single or dual channel image
is_nuclear = True #nuclear or whole-cell image
samples_set = "test"
test_dict = np.load(os.path.join(data_dir, 'tissuenet_{}_split_256x256_memserpreprocess.npz'.format(samples_set)), allow_pickle=True)


if is_single and is_nuclear:
    print("singel-channel nuclear data")
    X_test = test_dict['X'][..., 0]
    y_test = test_dict['y'][..., 1]

    #save path for singel-channel nuclear data
    tiff_dir = '/SAM_nuclear_1C/{}'.format(samples_set)
elif is_single and not is_nuclear:
    print("singel-channel wholecell data")
    X_test = test_dict['X'][..., 1]
    y_test = test_dict['y'][..., 0]

    #save path for singel-channel wholecell data
    tiff_dir = '/SAM_wholecell_1C/{}'.format(samples_set)
elif not is_single and is_nuclear:
    print("dual-channel images and nuclear annotations")
    X_test = test_dict['X']
    y_test = test_dict['y'][..., 1]

    #save path for dual-channel images and nuclear annotations 
    tiff_dir = '/SAM_nuclear_2C/{}'.format(samples_set)
elif not is_single and not is_nuclear:
    print("dual-channel images and wholecell annotations")
    X_test = test_dict['X']
    y_test = test_dict['y'][..., 0]

    # save path for dual-channel images and wholecell annotations 
    tiff_dir = '/SAM_wholecell_2C/{}'.format(samples_set)

if not os.path.isdir(tiff_dir):
    os.makedirs(tiff_dir)

tissue_type = test_dict['tissue_list']
tissue_filename = test_dict['filenames']

X = X_test.squeeze()
y = y_test.astype('int32').squeeze()
print(X.shape, y.shape)
print(np.unique(tissue_type))

assert X.shape[0] == y.shape[0], 'X and y should have the same number of images.'

tissues = np.unique(tissue_type)
for tissue in tissues:
    X_tissue = X[tissue_type==tissue]
    y_tissue = y[tissue_type==tissue]
    filename_tissue = tissue_filename[tissue_type==tissue]

    img_dir = '{}/images/{}'.format(tiff_dir, tissue)
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    
    lbl_dir = '{}/labels/{}'.format(tiff_dir, tissue)
    if not os.path.isdir(lbl_dir):
        os.makedirs(lbl_dir)

    if is_single:
        for i in range(X_tissue.shape[0]):
            tissue_file_name = filename_tissue[i]
            img_path = '{}/{}.png'.format(img_dir, tissue_file_name[:-4])
            mask_filename = '{}/{}.png'.format(lbl_dir, tissue_file_name[:-4])

            new_x = np.asarray(X_tissue[i])*255
            cv2.imwrite(img_path, new_x.astype(np.uint8))

            mask = Image.fromarray(y_tissue[i])
            mask_path = mask_filename
            mask.save(mask_path)
    else:
        for i in range(X_tissue.shape[0]):
            tissue_file_name = filename_tissue[i]
            img_path = '{}/{}.png'.format(img_dir, tissue_file_name[:-4])
            mask_filename = '{}/{}.png'.format(lbl_dir, tissue_file_name[:-4])

            height, width, c = X_tissue[i].shape
            zeros_channel = np.zeros((height, width), dtype=X_tissue[i].dtype)
            new_x = np.asarray(X_tissue[i])*255
            merged_img = np.dstack((new_x, zeros_channel))[..., [1,0,2]]
            merged_img = merged_img.astype(np.uint8)

            cv2.imwrite(img_path, merged_img)

            mask = Image.fromarray(y_tissue[i])
            mask_path = mask_filename
            mask.save(mask_path)

print('saved %s files to %s' % (len(X), tiff_dir))