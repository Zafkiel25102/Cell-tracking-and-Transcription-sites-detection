#!/usr/bin/env python
# coding: utf-8

# In[78]:


import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import cv2
import numpy as np
import skimage.io as skio
import os
from skimage.measure import regionprops

print('>>SEGpostprocess<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
root_path = os.path.join(sys.argv[2],sys.argv[3])
# root_path = r'/data/sunrui/celldata/20231018-SOX2-CELLID-A/1SOX2-CELLID-A/'
# file_name = r'10%Laser_300ms_2x2bin.tif'

# 读取一系列图像
raw_path = root_path + '/01_GT/SAMSEG/'  # 原始图像路径
imgfiles = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if f.endswith('.tif') or f.endswith('.tiff')]
imgfiles.sort()

img_raw = []

print(len(imgfiles))
for i in range(len(imgfiles)):
    img_raw.append(skio.imread(imgfiles[i]).astype(np.uint16))
print(len(img_raw))
img_raw = np.array(img_raw)
print(img_raw.shape)

# 创建与图像序列大小相同的空白图像
img_post = np.zeros(img_raw.shape, dtype=np.uint16)
print(img_post.shape)


def check_mask_intensity(image_t,mask_t,intensity_t):
    
    if intensity_t == 0:
        return False
    
    properties = regionprops(np.uint16(mask_t == 255), mask_t)[0]

    center_x = properties.centroid[1].round().astype(np.int16)
    center_y = properties.centroid[0].round().astype(np.int16)
    # 找到掩模中的非零点
    # nonzero_points = np.column_stack(np.where(mask_t > 0))

    # if len(nonzero_points) == 0:

    #     return False

    # # 计算掩模中心
    # center_x = int(np.mean(nonzero_points[:, 1]))
    # center_y = int(np.mean(nonzero_points[:, 0]))

    # 提取中心附近的像素强度值
    radius = 2  # 设置半径为10像素，你可以根据需要调整
    intensity_values = []
    for x in range(center_x - radius, center_x + radius + 1):
        for y in range(center_y - radius, center_y + radius + 1):
            # 确保不超出图像范围
            if 0 <= x < image_t.shape[1] and 0 <= y < image_t.shape[0]:
                intensity_values.append(image_t[y, x])
    
    # 判断中心附近的强度值是否与掩模中心的值相同
    center_intensity = intensity_t
    are_values_same = all(value == center_intensity for value in intensity_values)
    if not are_values_same:
        # plt.figure(figsize=(25, 25))
        # plt.subplot(1,2,1)
        # print('find a hole!')
        # print('intensity',intensity_t)
        # print(intensity_values)
        # print(center_x,center_y)
        # plt.scatter(center_x, center_y, color='red', marker='*', label='Center Point')
        
        # plt.imshow(mask_t)

        # plt.subplot(1,2,2)
        # plt.imshow(image_t)
        # plt.show()

        # x = 1

        pass
    return are_values_same


# 遍历图像序列
for i in range(img_raw.shape[0]):
    image = img_raw[i, :, :, 0]

    # 获取图像的最小灰度值和最大灰度值
    min_intensity = np.min(image)
    max_intensity = np.max(image)

    # 遍历灰度值范围
    # for intensity in range(min_intensity, max_intensity + 1):
    for id, intensity in enumerate(np.unique(image)):
        
        if intensity == 0:
            continue
        # 创建一个与原始图像大小相同的空白图像
        gray_value_image = (image == intensity).astype('uint8') * 255

        # 定义结构元素（这里使用3x3的正方形结构元素）
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # 进行腐蚀操作
        eroded_image_0 = cv2.erode(gray_value_image, structuring_element)
        eroded_image_1 = cv2.erode(eroded_image_0, structuring_element)
        # 进行膨胀操作
        dilated_image_0 = cv2.dilate(eroded_image_1, structuring_element)
        dilated_image_1 = cv2.dilate(dilated_image_0, structuring_element)

        # 计算非零像素的数量（PRE）
        non_zero_pixels_pre = np.count_nonzero(gray_value_image)
        # print('pre:')
        # print(non_zero_pixels_pre)

        # 计算非零像素的数量（POST）
        non_zero_pixels_post = np.count_nonzero(dilated_image_1)
        # print('post:')
        # print(non_zero_pixels_post)

        # 结果
        post_result = (dilated_image_1 / 255) * intensity
        post_result = post_result.astype('uint16')
        ss = np.max(post_result)
        # print('label:')
        # print(ss)
        # print('...................................................')

        # # 更新img_post
        # if non_zero_pixels_post > 850:
        #     img_post[i, :, :, 0] += post_result

        # 更新img_post
        if non_zero_pixels_post > 230:
            # print('frame_ID:',i )
            is_same = check_mask_intensity(image,gray_value_image,intensity)
            
            if is_same:
                img_post[i, :, :, 0] += post_result


# In[79]:


print(img_post.shape)
print(img_post.dtype)
img_post_16 = img_post.astype(np.uint16)

# In[80]:


print(np.max(img_post_16))
print(np.mean(img_post_16))
print(np.min(img_post_16))

# In[81]:


import skimage.io as skio
from tifffile import imread, imwrite
import matplotlib.pyplot as plt


image_folder =root_path + r'/01_GT/SEG_16/'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

for i in range(img_post.shape[0]):
    image_path = os.path.join(image_folder, f'man_seg{i:04d}.tif')
    imwrite(image_path,img_post_16[i,:,:,0])


#plt.imshow(img_post_16[14,:,:,0])
#plt.show()

# In[82]:


# import skimage.io as skio
# from tifffile import imread, imwrite
# import matplotlib.pyplot as plt
# image_folder =root_path + r'/01_GT/SEG_8/'
# if not os.path.exists(image_folder):
#     os.makedirs(image_folder)

# for i in range(img_post.shape[0]):
#     image_path = os.path.join(image_folder, f'man_seg{i:03d}.tif')
#     imwrite(image_path,img_post[i,:,:,0])


# plt.imshow(img_post[14,:,:,0])
# plt.show()

# In[83]:


# plt.imshow(img_post[14,:,:,0])
# plt.show()

# In[84]:


# import cv2
# import numpy as np
# import skimage.io as skio
# from tifffile import imread, imwrite
# import matplotlib.pyplot as plt
# import os

# root_path = r'/data/sunrui/celldata/r03c03f01/'
# file_name = r'r03c03f01.tif'

# #读取一系列图像
# raw_path = root_path + '/01_GT/SAMSEG/'#原始图像路径
# imgfiles = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if f.endswith('.tif') or f.endswith('.tiff')]
# imgfiles.sort()

# img_raw = []

# print(len(imgfiles))
# for i in range(len(imgfiles)):
#     img_raw.append(skio.imread(imgfiles[i]).astype(np.uint16))
# print(len(img_raw))
# img_raw = np.array(img_raw)
# print(img_raw.shape)

# img_post = np.zeros(img_raw.shape, dtype= np.uint8)
# print(img_post.shape)

# image = img_raw[0,:,:,0]
# image_post = img_post[0,:,:,0]
# # 读取灰度图像
# # image = cv2.imread("gray_image.jpg", cv2.IMREAD_GRAYSCALE)

# # 获取图像的最小灰度值和最大灰度值
# min_intensity = np.min(image)
# max_intensity = np.max(image)

# # 设置图像框的大小
# figure_size = (20, 20)  # 设置为(宽度, 高度)

# # image_post = np.zeros(image.shape, dtype = np.uint8)

# # 遍历灰度值范围
# for intensity in range(min_intensity, max_intensity + 1):
#     # 创建一个与原始图像大小相同的空白图像
#     gray_value_image = (image == intensity).astype('uint8') * 255

#     # 定义结构元素（这里使用3x3的正方形结构元素）
#     structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

#     # 进行腐蚀操作
#     eroded_image_0 = cv2.erode(gray_value_image, structuring_element)
#     eroded_image_1 = cv2.erode(eroded_image_0, structuring_element)
#     # 进行膨胀操作
#     dilated_image_0 = cv2.dilate(eroded_image_1, structuring_element)
#     dilated_image_1 = cv2.dilate(dilated_image_0, structuring_element)

#     # 计算非零像素的数量(PRE)
#     non_zero_pixels_pre = np.count_nonzero(gray_value_image)
#     print('pre:')
#     print(non_zero_pixels_pre)

#     # 计算非零像素的数量(POST)
#     non_zero_pixels_post = np.count_nonzero(dilated_image_1)
#     print('post:')
#     print(non_zero_pixels_post)

#     #结果
#     post_result = (dilated_image_1/255)*intensity
#     post_result = post_result.astype('uint8')
#     ss = np.max(post_result)
#     print('label:')
#     print(ss)
#     print('...................................................')


#     if non_zero_pixels_post > 1000:
#         image_post += post_result

#     # # 设置图像框的大小
#     # plt.figure(figsize=figure_size)

#     # # 显示当前灰度值图像
#     # plt.subplot(121)
#     # plt.imshow(gray_value_image, cmap='gray')
#     # plt.axis('off')
#     # plt.title("Original Image")

#     # # 显示腐蚀和膨胀后的图像
#     # plt.subplot(122)
#     # plt.imshow(dilated_image_1, cmap='gray')
#     # plt.axis('off')
#     # plt.title("Eroded Image")

#     # plt.show()
# # img_post[0,:,:,0] = image_post
