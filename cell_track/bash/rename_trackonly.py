#!/usr/bin/env python
# coding: utf-8

# In[109]:


import skimage.io as skio
import numpy as np
import cv2
from tifffile import imread, imwrite

import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage.measure import regionprops

# In[110]:
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

print('>>rename<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
root_path = os.path.join(sys.argv[2],sys.argv[3])
file_name = sys.argv[3]+r'.tif'




# y = skio.imread(root_path + r'/01/' + file_name, plugin="tifffile")
y = skio.imread(root_path +'/PRE/test.tif', plugin="tifffile")
# y = skio.imread(root_path + file_name, plugin="tifffile")


print("y:")
print(y.shape)
print(y.dtype)

# In[116]:


image_folder_01 =root_path + r'/01/'  # 图片保存的文件夹路径
for i in range(y.shape[0]):
    target = y[i]
    image_path = os.path.join(image_folder_01, f't{i:03d}.tif')
    imwrite(image_path,target)


# In[117]:


import shutil
shutil.move(root_path + '/01/' + file_name, root_path)
# shutil.rmtree( root_path + '/PRE/')

