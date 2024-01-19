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
dictdata = np.load(os.path.join(data_dir, 'LIVECell_{}_all.npz'.format(sample_type)), allow_pickle=True)
path_to_save_folder = "COCO_LIVECell/{}".format(sample_type)
path_to_save_anno_folder = 'COCO_LIVECell'

if not os.path.isdir(path_to_save_folder):
    os.makedirs(path_to_save_folder)

X = dictdata['X'].squeeze()
y = dictdata['y'].squeeze()
y = y.astype('int32').squeeze()
tissue_types = dictdata['tissue_list']


categories = [{"supercategory": "cell", "id": 1, "name": "cell"}]
images_info = []
annotations = []

suffix_zeros = math.ceil(math.log10(len(X)))

print("X", X.shape)
print("y:", y.shape)
print("tissue_types:", tissue_types, len(tissue_types))
print("suffix_zeros:", suffix_zeros)

for index, (x, yn, tissue_type) in enumerate(zip(X, y, tissue_types)):
        print("index, tissue_type:", index, tissue_type)
        
        height, width = x.shape
        
        filename = '{}_{:06d}.tif'.format(tissue_type, index)
        save_path = '{}/{}'.format(path_to_save_folder, filename)
        
        img = (x - x.min())/(x.max() - x.min())
        tifffile.imwrite(save_path, img.astype(np.float32))

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
                
                annotations.append(anno)

coco = {
    "images": images_info,
    "categories": categories,
    "annotations": annotations
}
with open(os.path.join(path_to_save_anno_folder, "LIVECell_all_{}.json".format(sample_type)), "w") as f:
    json.dump(coco, f)