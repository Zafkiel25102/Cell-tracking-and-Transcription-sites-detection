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

# In[111]:


#读取一系列图像
raw_path = root_path + '/01_GT/SEG_16/'#原始图像路径
imgfiles = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if f.endswith('.tif') or f.endswith('.tiff')]
imgfiles.sort()

img_raw = []
print(len(imgfiles))
for i in range(len(imgfiles)):
    img_raw.append(skio.imread(imgfiles[i]).astype(np.uint16))
print(len(img_raw))
img_raw = np.array(img_raw)
print(img_raw.shape)
print(img_raw.shape[0])
x = np.expand_dims(img_raw,axis = 3)
print(x.shape)


# In[112]:


# x = skio.imread(root_path + r'/PRE/test_seg.tif', plugin="tifffile")
# print("x:")
# print(x.shape)
# print(x.dtype)
# print(np.max(x))
# print(np.min(x))
# print(np.mean(x))
# x = x.astype(np.uint16)
# print(x.dtype)
# print(np.max(x))
# print(np.min(x))
# print(np.mean(x))


# In[113]:


all_non_zero = False  # 变量用于记录是否所有图像都不为零

while not all_non_zero:
    all_non_zero = True  # 假设所有图像都不为零，如果发现全零图像则置为 False
    
    for i in range(x.shape[0]):
        img_curr = x[i]
        
        if np.max(img_curr) == 0:
            if i > 0 and np.max(x[i-1]) > 0:
                x[i] = x[i-1].copy()
                print('copy img!!')
            elif i < x.shape[0] - 1 and np.max(x[i+1]) > 0:
                x[i] = x[i+1].copy()
                print('copy img!!')
            else:
                all_non_zero = False  # 存在全零图像，继续迭代
        
# for i in range(x.shape[0]):
#     img_curr = x[i]
    
#     # 检查当前图像是否全零
#     if np.max(img_curr) == 0:
#         all_non_zero = False
#         break  # 存在全零图像，跳出循环

# for i in range(x.shape[0]):
#     img_curr = x[i]
#     if i==0 and np.max(img_curr)==0 :
#         img_curr = x[i+1].copy()
#     elif np.max(img_curr) == 0:
#         img_curr = x[i-1].copy()
#     x[i] = img_curr

# In[114]:


image_folder =root_path + r'/01_GT/SEG/'  # 图片保存的文件夹路径
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

for i in range(x.shape[0]):
    target = x[i]
    image_path = os.path.join(image_folder, f'man_seg{i:04d}.tif')
    imwrite(image_path,target)




# In[115]:


# # y = skio.imread(root_path + r'/01/' + file_name, plugin="tifffile")
# y = skio.imread(root_path +'/PRE/test.tif', plugin="tifffile")
# # y = skio.imread(root_path + file_name, plugin="tifffile")


# print("y:")
# print(y.shape)
# print(y.dtype)

# # In[116]:


# image_folder_01 =root_path + r'/01/'  # 图片保存的文件夹路径
# for i in range(y.shape[0]):
#     target = y[i]
#     image_path = os.path.join(image_folder_01, f't{i:04d}.tif')
#     imwrite(image_path,target)


# In[117]:


import shutil
# shutil.move(root_path + '/01/' + file_name, root_path)
shutil.rmtree( root_path + '/01_GT/SEG_16/')
#shutil.rmtree( root_path + '/01_GT/SAMSEG/')
#shutil.rmtree( root_path + '/PRE/')

