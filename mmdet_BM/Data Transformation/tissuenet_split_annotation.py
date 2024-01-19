import pandas as pd 
import os
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import imagecodecs
from pycocotools.coco import COCO
from PIL import Image
import os
import tqdm
import cv2
import numpy as np
import json

tissue_list = ['Colon', 'lymph node metastasis', 'Spleen', 'Pancreas', 'Epidermis', 'Breast', 'Lymph Node', 'Tonsil', 'Lung', 'Esophagus']
sample_type = 'val' #test, train, and val
channel = 'single' # singel or dual
sample = 'wholecell' # wholecell or nuclear

chan = {'single' : '1', 'dual' : '2'}
if channel == 'dual':
    JSON_LOC='COCO_TissueNet_{}Channel/tissuenet_{}_all_{}_{}C.json'.format(chan[channel], sample, sample_type, chan[channel])
else:
    JSON_LOC='COCO_TissueNet_{}Channel/tissuenet_{}_all_{}.json'.format(chan[channel], sample, sample_type)

SAVE_DIR = 'COCO_TissueNet_{}Channel'.format(chan[channel])

print("JSON_LOC:", JSON_LOC)
print("SAVE_DIR:", SAVE_DIR)

for tissue in tissue_list:
    print("tissue:", tissue)
    val_json = open(JSON_LOC, "r")
    json_object = json.load(val_json)
    val_json.close()
    keep_list = []
    for idx, instance in enumerate(json_object["images"]):
        if instance['file_name'].split("_")[0] == tissue:
            keep_list.append(idx)
    
    json_object["images"] = [i for j, i in enumerate(json_object["images"]) if j in keep_list]
    count = 0
    for i in json_object["images"]:
        count += 1
    print("count:", count)

    if channel == 'dual':
        json_file = open("{}/tissuenet_{}_{}_{}_{}C.json".format(SAVE_DIR, sample, tissue, sample_type, chan[channel]), "w")
    else:
        json_file = open("{}/tissuenet_{}_{}_{}.json".format(SAVE_DIR, sample, tissue, sample_type), "w")

    json.dump(json_object, json_file)
    json_file.close()
