import cv2
import numpy as np

# mask = cv2.imread("F:/SAM_nuclear_2c/test/labels/Breast/Breast_20200526_COH_BC_37_exp.png", -1)
# print(mask.shape, np.unique(mask))

# mask = cv2.imread("F:/SAM_nuclear_oc/test/labels/Breast/Breast_20200526_COH_BC_36_exp.png", -1)
# print(mask.shape, np.unique(mask))

# mask = cv2.imread("F:/SAM_livecell/test/labels/A172/A172_000000.png", -1)
# print(mask.shape, np.unique(mask))

mask = cv2.imread("F:/SAM_livecell/test/labels/BT474/BT474_000000.png", -1)
print(mask.shape, np.unique(mask))

