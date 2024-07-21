#!/usr/bin/env python
# coding: utf-8

# In[76]:


import os
import sys


import numpy as np

import os



import numpy as np





import matplotlib.pyplot as plt
import os
import skimage.io as skio
import numpy as np 


import numpy as np

import skimage.io as skio
import matplotlib.pyplot as plt


# In[77]:
print('>>SAMPRE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')


# In[82]:


# root_path = os.path.join(sys.argv[2],sys.argv[3])
root_path = r'/mnt/sda/cell_data/l3/PAX6_18_29_5h_76h_LLS/PAX6_18_29_5h_76h_LLS/'
#/mnt/sda/cell_data/l3/PAX6_18_29_5h_76h_LLS/PAX6_18_29_5h_76h_LLS/01_GT/SEG/

# file_name = sys.argv[3]+r'.tif'

# img_raw = skio.imread(root_path +'/01/'+ file_name, plugin="tifffile")
# img_raw = skio.imread(root_path + file_name, plugin="tifffile")
# print(img_raw.shape)


#读取一系列图像

# root_path = r'/data/sunrui/celldata/20230824_HBEC_test_DL/10%Laser_300ms_1x1bin/'
raw_path = root_path + r'/01_GT/SEG/'#原始图像路径
imgfiles = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if f.endswith('.tif') or f.endswith('.tiff')]
imgfiles.sort()

img_raw = []
print(len(imgfiles))


x = skio.imread(imgfiles[0]).astype(np.uint16)
x = np.array(x)
print(x.shape)
# x = np.expand_dims(x,axis=2)
# x = np.expand_dims(x,axis=3)
# print('x:')
# print(x.shape)
# print(x.dtype)
# print('.............................')
from tifffile import imread, imwrite

total_frames = len(imgfiles)

batch_num = 200
num_frames = batch_num + 1

target_image = x


saveDir_spilt = root_path + r'/spilt/'
if not os.path.exists(saveDir_spilt):
    os.makedirs(saveDir_spilt)

t=1

for i in range(0,total_frames,batch_num):
    saveDir_spilt_1 = os.path.join(saveDir_spilt,f'spilt_{t:04d}')
    t+=1
    if not os.path.exists(saveDir_spilt_1):
        os.makedirs(saveDir_spilt_1)

    for j in range(num_frames):

        img_raw0 = skio.imread(imgfiles[i+j]).astype(np.uint16)
        img_raw0 = np.array(img_raw0)
        # img_raw0 = np.expand_dims(img_raw0,axis=3)

        img_pre = img_raw0
        #img_pre = background_noise(img_pre)
        # img_pre = hist_match(img_pre, target_image)  

        saveDir_spilt_01 = saveDir_spilt_1 + r'/01_GT/SEG/'
        if not os.path.exists(saveDir_spilt_01):
            os.makedirs(saveDir_spilt_01)
        image_path_01 = os.path.join(saveDir_spilt_01, f'man_seg{j:04d}.tif')
        imwrite(image_path_01, img_pre)



        pass
   







