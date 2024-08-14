import skimage.io as skio
import numpy as np
import cv2
from tifffile import imread, imwrite

import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage.measure import regionprops


raw_path = sys.argv[1]
imgfiles = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if f.endswith('.tif') or f.endswith('.tiff')]
imgfiles.sort()

image_folder_01 =raw_path
save_folder = sys.argv[2]
os.makedirs(save_folder, exist_ok=True)


for i,ids in enumerate(imgfiles):
    y = skio.imread(ids, plugin="tifffile")
    target = y
    file_name = ids.split('/')[-1].split('.')[0].split('_')[-1]
    print(file_name)
    file_name_id = int(file_name)
    image_path = os.path.join(save_folder, f'man_seg{i:03d}.tif')
    print('....................................')
    print(file_name)
    print(image_path)
    print('....................................')
    imwrite(image_path,target)