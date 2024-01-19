import os
import numpy as np
from itertools import chain
from tifffile import imwrite
import tifffile
import json
import math
from pycocotools import mask
from skimage import measure
from skimage.io import imsave, imread
from PIL import Image, ImageOps
from matplotlib import pyplot as plt

data_dir = "path/to/your/data/dir"
sample_type = 'val'
# sample_type = 'train'
# sample_type = 'test'
dictdata = np.load(os.path.join(data_dir, 'tissuenet_{}_split_256x256_memserpreprocess.npz'.format(sample_type)), allow_pickle=True)
path_to_save_folder = "COCO_TissueNet_2Channel/{}".format(sample_type)
path_to_save_anno_folder = 'COCO_TissueNet_2Channel'

if not os.path.isdir(path_to_save_folder):
    os.makedirs(path_to_save_folder)

if not os.path.isdir(path_to_save_anno_folder):
    os.makedirs(path_to_save_anno_folder)

X = dictdata['X']
y_nuclear = dictdata['y'][...,1].squeeze()
y_wholecell = dictdata['y'][...,0].squeeze()

filenames = dictdata['filenames']

categories = [{"supercategory": "cell", "id": 1, "name": "cell"}]
images_info = []
annotations_nuclear = []
annotations_wholecell = []

suffix_zeros = math.ceil(math.log10(len(X)))

print("X", X.shape)
print("y_nuclear, y_wholecell:", y_nuclear.shape, y_wholecell.shape)
print("filenames:", filenames, len(filenames))
print("suffix_zeros:", suffix_zeros)
for index, (x, yn, yw, filename) in enumerate(zip(X, y_nuclear, y_wholecell, filenames)):
        print("index, filename:", index, filename)
        
        height, width, c = x.shape
        zeros_channel = np.zeros((height, width), dtype=x.dtype)
        new_x = np.asarray(x)*255
        merged_img = np.dstack((new_x, zeros_channel))
        merged_img = merged_img.astype(int)
        save_path = os.path.join(path_to_save_folder, filename)
        tifffile.imsave(save_path, merged_img)

        images_info.append({"file_name": filename, "id": index, "width": width, "height": height})

        label_list = np.unique(yn)[1:]
        for j in range(len(label_list)):
                ground_truth_binary_mask = (yn == label_list[j])
                ground_truth_binary_mask = ground_truth_binary_mask.astype(np.uint8)
                fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
                rle = mask.encode(fortran_ground_truth_binary_mask)
                ground_truth_area = int(mask.area(rle))
                ground_truth_bounding_box = mask.toBbox(rle)
                rle['counts'] = rle['counts'].decode('utf-8')
                
                anno = {
                "segmentation": rle,
                "area": ground_truth_area,
                "bbox": ground_truth_bounding_box.tolist(),
                "image_id": index,
                "category_id": 1,
                "id": int(f"{index:0>{suffix_zeros}}{j:0>{suffix_zeros}}"),
                "iscrowd": 0,
                }
                
                annotations_nuclear.append(anno)
        
        label_list = np.unique(yw)[1:]
        for j in range(len(label_list)):
                ground_truth_binary_mask = (yw == label_list[j])
                ground_truth_binary_mask = ground_truth_binary_mask.astype(np.uint8)
                fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
                rle = mask.encode(fortran_ground_truth_binary_mask)
                ground_truth_area = int(mask.area(rle))
                ground_truth_bounding_box = mask.toBbox(rle)
                rle['counts'] = rle['counts'].decode('utf-8')

                anno = {
                "segmentation": rle,
                "area": ground_truth_area,
                "bbox": ground_truth_bounding_box.tolist(),
                "image_id": index,
                "category_id": 1,
                "id": int(f"{index:0>{suffix_zeros}}{j:0>{suffix_zeros}}"),
                "iscrowd": 0,
                }
                
                annotations_wholecell.append(anno)

coco = {
    "images": images_info,
    "categories": categories,
    "annotations": annotations_nuclear
}
with open(os.path.join(path_to_save_anno_folder, "tissuenet_nuclear_all_{}_2C.json".format(sample_type)), "w") as f:
    json.dump(coco, f)

coco = {
    "images": images_info,
    "categories": categories,
    "annotations": annotations_wholecell
}
with open(os.path.join(path_to_save_anno_folder, "tissuenet_wholecell_all_{}_2C.json".format(sample_type)), "w") as f:
    json.dump(coco, f)