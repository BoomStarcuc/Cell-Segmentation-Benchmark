import os
import numpy as np
import tifffile
import random


# File path to read
data_dir = '/path/to/your/data/dir'
is_single = False #single or dual channel image
is_nuclear = True #nuclear or whole-cell image
samples_set = "test"
test_dict = np.load(os.path.join(data_dir, 'tissuenet_{}_split_256x256_memserpreprocess.npz'.format(samples_set)), allow_pickle=True)
# test_dict = np.load(os.path.join(data_dir, 'tissuenet_{}_split_256x256_memserpreprocess.npz'.format(samples_set)), allow_pickle=True)
# test_dict = np.load(os.path.join(data_dir, 'tissuenet_{}_split_256x256_memserpreprocess.npz'.format(samples_set)), allow_pickle=True)


if is_single and is_nuclear:
    print("singel-channel nuclear data")
    X_test = test_dict['X'][..., 0]
    y_test = test_dict['y'][..., 1]

    #save path for singel-channel nuclear data
    tiff_dir = '/cellpose_nuclear_oc/{}'.format(samples_set)
elif is_single and not is_nuclear:
    print("singel-channel wholecell data")
    X_test = test_dict['X'][..., 1]
    y_test = test_dict['y'][..., 0]

    #save path for singel-channel wholecell data
    tiff_dir = '/cellpose_wholecell_oc/{}'.format(samples_set)
elif not is_single and is_nuclear:
    print("dual-channel images and nuclear annotations")
    X_test = test_dict['X']
    y_test = test_dict['y'][..., 1]

    #save path for dual-channel images and nuclear annotations 
    tiff_dir = '/cellpose_nuclear_2c/{}'.format(samples_set)
elif not is_single and not is_nuclear:
    print("dual-channel images and wholecell annotations")
    X_test = test_dict['X']
    y_test = test_dict['y'][..., 0]

    # save path for dual-channel images and wholecell annotations 
    tiff_dir = '/cellpose_wholecell_2c/{}'.format(samples_set)


if not os.path.isdir(tiff_dir):
    os.makedirs(tiff_dir)

tissue_type = test_dict['tissue_list']
tissue_filename = test_dict['filenames']

X = X_test.squeeze()
y = y_test.astype('int32').squeeze()
print(X.shape, y.shape)

assert X.shape[0] == y.shape[0], 'X and y should have the same number of images.'

if is_single:
    for i in range(X.shape[0]):
        tissue_file_name = tissue_filename[i]
        img_filename = '{}_img.tif'.format(tissue_file_name[:-4])
        mask_filename = '{}_masks.tif'.format(tissue_file_name[:-4])

        tifffile.imwrite(os.path.join(tiff_dir, img_filename), X[i])
        tifffile.imwrite(os.path.join(tiff_dir, mask_filename), y[i])
else:
    for i in range(X.shape[0]):
        tissue_file_name = tissue_filename[i]
        img_filename = '{}_img.tif'.format(tissue_file_name[:-4])
        mask_filename = '{}_masks.tif'.format(tissue_file_name[:-4])

        tifffile.imwrite(os.path.join(tiff_dir, img_filename), X[i].transpose(2, 0, 1))
        tifffile.imwrite(os.path.join(tiff_dir, mask_filename), y[i])

print('saved %s files to %s' % (len(X), tiff_dir_test))