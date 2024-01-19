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

tissue_list = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SkBr3', 'SHSY5Y', 'SKOV3']
sample_type = 'val' #test, train, and val


JSON_LOC='COCO_LIVECell/LIVECell_all_{}.json'.format(sample_type)
SAVE_DIR = 'COCO_LIVECell'

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

    json_file = open("{}/LIVECell_{}_{}.json".format(SAVE_DIR, tissue, sample_type), "w")
    json.dump(json_object, json_file)
    json_file.close()
