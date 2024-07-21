#!/usr/bin/env python
# coding: utf-8

# In[76]:


import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import logging

import numpy as np
from scipy import ndimage
from skimage import morphology
from skimage import segmentation
from skimage.feature import peak_local_max
from skimage.exposure import equalize_adapthist
from skimage.exposure import rescale_intensity
from skimage.measure import label
import os

from PIL import Image 
from scipy import signal
import cv2

import numpy as np

import imageio
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import os
import skimage.io as skio
import numpy as np 
import matplotlib.image as mpimg
from math import nan 


import numpy as np
import logging
import skimage.io as skio
import matplotlib.pyplot as plt
import cv2

# In[77]:
print('>>SAMPRE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

def histogram_normalization(image, kernel_size=None):
    """Pre-process images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE).

    If one of the inputs is a constant-value array, it will
    be normalized as an array of all zeros of the same shape.

    Args:
        image (numpy.array): numpy array of phase image data.
        kernel_size (integer): Size of kernel for CLAHE,
            defaults to 1/8 of image size.

    Returns:
        numpy.array: Pre-processed image data with dtype float32.
    """
    if not np.issubdtype(image.dtype, np.floating):
        logging.info('Converting image dtype to float')
    image = image.astype('float32')

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            X = image[batch, ..., channel]
            sample_value = X[(0,) * X.ndim]
            if (X == sample_value).all():
                # TODO: Deal with constant value arrays
                # https://github.com/scikit-image/scikit-image/issues/4596
                logging.warning('Found constant value array in batch %s and '
                                'channel %s. Normalizing as zeros.',
                                batch, channel)
                image[batch, ..., channel] = np.zeros_like(X)
                continue

            # X = rescale_intensity(X, out_range='float')
            X = rescale_intensity(X, out_range=(0.0, 1.0))
            X = equalize_adapthist(X, kernel_size=kernel_size)
            image[batch, ..., channel] = X
    return image

# In[78]:



# 腐蚀操作
def grayscale_erosion(image, structuring_element):

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            X = image[batch, ..., channel]
            h, w = X.shape
            h_se, w_se = structuring_element.shape
            eroded_image = np.zeros_like(X)
            for i in range(h):
                for j in range(w):
                    min_value = float('inf')  # 初始化为无穷大，确保会被任何像素值取代

                    # 遍历结构元素的所有像素
                    for m in range(h_se):
                        for n in range(w_se):
                            if structuring_element[m, n] == 1:
                                # 边界处理，如果越界则忽略该像素
                                if i + m >= 0 and i + m < h and j + n >= 0 and j + n < w:
                                    current_value = X[i + m, j + n]
                                    min_value = min(min_value, current_value)

            eroded_image[i, j] = min_value
            image[batch, ..., channel] = eroded_image
    return image

# In[79]:



#边缘检测操作
def edge_enhancement(input_image):
    # 定义Sobel算子
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # 计算图像梯度
    gradient_x = cv2.filter2D(input_image, -1, sobel_x)
    gradient_y = cv2.filter2D(input_image, -1, sobel_y)

    # 计算梯度幅值
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # 边缘增强
    enhanced_image = input_image + gradient_magnitude

    return enhanced_image




# In[80]:


#canny边缘增强
def edge_enhancement_canny(image, threshold1, threshold2):
    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            X = image[batch, ..., channel]
             # 边缘检测
            edges = cv2.Canny(X, threshold1, threshold2)

            # 边缘增强
            enhanced_image = X + 2*edges
            image[batch, ..., channel] = enhanced_image
    
    # # 边缘检测
    # edges = cv2.Canny(input_image, threshold1, threshold2)

    # # 边缘增强
    # enhanced_image = input_image + edges

    return image

# # 设置Canny算法的阈值
# threshold1 = 100
# threshold2 = 200

# # 边缘增强
# enhanced_image = edge_enhancement_canny(input_image, threshold1, threshold2)


# In[81]:


# 步骤一 进口轮子 编写函数


# 加载函数
# 去掉高于平均背景水平的像素
def background_noise(image,noise_level= 3):
    if not np.issubdtype(image.dtype, np.floating):
        logging.info('Converting image dtype to float')
    image = image.astype('float32')

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            # img[img<noise_level] = noise_level
            img[img<noise_level] = 0
            image[batch, ..., channel] = img
    return image
# 均值滤波
def blur_proc(image):
    if not np.issubdtype(image.dtype, np.floating):
        logging.info('Converting image dtype to float')
    image = image.astype('float32')

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            normal_image = cv2.blur(img,(20,20))
            image[batch, ..., channel] = normal_image
    return image

def remove_hight_variance_pixels(image):
    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            max_intensity = 10*img.mean()#10
            img[img>max_intensity] = img.mean()
            image[batch, ..., channel] = img
    return image

#均衡化
def his(image):


    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            normal_image = cv2.equalizeHist(img)
            image[batch, ..., channel] = normal_image
    return image

def float32_to_uint8(float_value):
    # 将float32的值限制在0到1之间
    # float_value = np.clip(float_value, 0.0, 1.0)
    
    # # 将float32乘以255，并进行舍入
    # uint8_value = np.round(float_value * 255.0, decimals=0).astype(np.uint8)
    
    scaled_image = (float_value - np.min(float_value)) / (np.max(float_value) - np.min(float_value))
    uint8_value = np.round(scaled_image * 255.0).astype(np.uint8)

    return uint8_value


def uint8_to_uint16(uint8_array):
    # 将float32的值限制在0到1之间
    # float_value = np.clip(float_value, 0.0, 1.0)
    
    # # 将float32乘以255，并进行舍入
    # uint8_value = np.round(float_value * 255.0, decimals=0).astype(np.uint8)
        
    scaled_array = (uint8_array.astype(np.float32) / 255.0) * 65535.0
    uint16_array = scaled_array.astype(np.uint16)

    return uint16_array


def uint16_to_uint8(uint16_array):
    # 将uint16数组转换为uint8数组
    uint8_array = (uint16_array / 65535.0 * 255).astype(np.uint8)
    return uint8_array


def adaptive_threshold_enhancement(image):
    # 自适应阈值处理
    thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresholded

def contrast_enhancement(image):
    # 局部对比度增强 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return enhanced

def process_frame(image):

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            # 自适应阈值处理
            thresholded = adaptive_threshold_enhancement(img)
            # 局部对比度增强
            enhanced = contrast_enhancement(thresholded)
            image[batch, ..., channel] = enhanced
    return image
    


def average_intensity(image):
    intensity_sum = 0
    mean_intensity = 0

    # 计算图像集合的总强度
    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            intensity_sum += np.mean(img)
    
    mean_intensity = intensity_sum / (image.shape[0] * image.shape[-1])

    # 根据平均强度进行像素强度调整
    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            intensity_local = np.mean(img)
            if intensity_local < mean_intensity:
                img_diff = mean_intensity - intensity_local
                img = img + img_diff
            if intensity_local > mean_intensity:
                img_diff = intensity_local - mean_intensity
                img = img - img_diff
            image[batch, ..., channel] = img

    return image

def hist_match(source, template):  
    # 计算源图像和目标图像的直方图  
    source_hist, _ = np.histogram(source.ravel(), 256, [0, 256])  
    template_hist, _ = np.histogram(template.ravel(), 256, [0, 256])  
  
    # 归一化直方图  
    source_hist = source_hist.astype('float')  
    template_hist = template_hist.astype('float')  
    source_hist /= (source_hist.sum() + 1e-7)  # 避免除以零  
    template_hist /= (template_hist.sum() + 1e-7)  
  
    # 计算累积分布函数（CDF）  
    source_cdf = source_hist.cumsum()  
    template_cdf = template_hist.cumsum()  
  
    # 创建映射表  
    mapping = np.zeros(256)  
    for i in range(256):  
        # 找到最接近的累积分布值  
        diff = template_cdf - source_cdf[i]  
        idx = np.argmin(np.abs(diff))  
        mapping[i] = idx  
  
    # 应用映射表到源图像  
    matched = np.interp(source.ravel(), np.arange(256), mapping)  
    matched = matched.reshape(source.shape)  
  
    return matched.astype('uint8')  

# In[82]:


root_path = os.path.join(sys.argv[2],sys.argv[3])
file_name = sys.argv[3]+r'.tif'

# img_raw = skio.imread(root_path +'/01/'+ file_name, plugin="tifffile")
# img_raw = skio.imread(root_path + file_name, plugin="tifffile")
# print(img_raw.shape)


#读取一系列图像

# root_path = r'/data/sunrui/celldata/20230824_HBEC_test_DL/10%Laser_300ms_1x1bin/'
raw_path = root_path + r'/01/'#原始图像路径
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

target_image = x

saveDir_mul = root_path + r'/PRE/PRE_MUL/'
if not os.path.exists(saveDir_mul):
    os.makedirs(saveDir_mul)

for i in range(total_frames):
    img_raw0 = skio.imread(imgfiles[i]).astype(np.uint16)
    img_raw0 = np.array(img_raw0)
    # img_raw0 = np.expand_dims(img_raw0,axis=3)

    img_pre = img_raw0
    #img_pre = background_noise(img_pre)
    # img_pre = hist_match(img_pre, target_image)  
    
    image_path = os.path.join(saveDir_mul, f'test_{i:04d}.tif')
    imwrite(image_path, img_pre)







