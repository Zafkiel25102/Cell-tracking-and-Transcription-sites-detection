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



# ��ʴ����
def grayscale_erosion(image, structuring_element):

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            X = image[batch, ..., channel]
            h, w = X.shape
            h_se, w_se = structuring_element.shape
            eroded_image = np.zeros_like(X)
            for i in range(h):
                for j in range(w):
                    min_value = float('inf')  # ��ʼ��Ϊ�����ȷ���ᱻ�κ�����ֵȡ��

                    # �����ṹԪ�ص���������
                    for m in range(h_se):
                        for n in range(w_se):
                            if structuring_element[m, n] == 1:
                                # �߽紦�����Խ������Ը�����
                                if i + m >= 0 and i + m < h and j + n >= 0 and j + n < w:
                                    current_value = X[i + m, j + n]
                                    min_value = min(min_value, current_value)

            eroded_image[i, j] = min_value
            image[batch, ..., channel] = eroded_image
    return image

# In[79]:



#��Ե������
def edge_enhancement(input_image):
    # ����Sobel����
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # ����ͼ���ݶ�
    gradient_x = cv2.filter2D(input_image, -1, sobel_x)
    gradient_y = cv2.filter2D(input_image, -1, sobel_y)

    # �����ݶȷ�ֵ
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # ��Ե��ǿ
    enhanced_image = input_image + gradient_magnitude

    return enhanced_image




# In[80]:


#canny��Ե��ǿ
def edge_enhancement_canny(image, threshold1, threshold2):
    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            X = image[batch, ..., channel]
             # ��Ե���
            edges = cv2.Canny(X, threshold1, threshold2)

            # ��Ե��ǿ
            enhanced_image = X + 2*edges
            image[batch, ..., channel] = enhanced_image
    
    # # ��Ե���
    # edges = cv2.Canny(input_image, threshold1, threshold2)

    # # ��Ե��ǿ
    # enhanced_image = input_image + edges

    return image

# # ����Canny�㷨����ֵ
# threshold1 = 100
# threshold2 = 200

# # ��Ե��ǿ
# enhanced_image = edge_enhancement_canny(input_image, threshold1, threshold2)


# In[81]:


# ����һ �������� ��д����


# ���غ���
# ȥ������ƽ������ˮƽ������
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
# ��ֵ�˲�
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

#���⻯
def his(image):


    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            normal_image = cv2.equalizeHist(img)
            image[batch, ..., channel] = normal_image
    return image

def float32_to_uint8(float_value):
    # ��float32��ֵ������0��1֮��
    # float_value = np.clip(float_value, 0.0, 1.0)
    
    # # ��float32����255������������
    # uint8_value = np.round(float_value * 255.0, decimals=0).astype(np.uint8)
    
    scaled_image = (float_value - np.min(float_value)) / (np.max(float_value) - np.min(float_value))
    uint8_value = np.round(scaled_image * 255.0).astype(np.uint8)

    return uint8_value


def uint8_to_uint16(uint8_array):
    # ��float32��ֵ������0��1֮��
    # float_value = np.clip(float_value, 0.0, 1.0)
    
    # # ��float32����255������������
    # uint8_value = np.round(float_value * 255.0, decimals=0).astype(np.uint8)
        
    scaled_array = (uint8_array.astype(np.float32) / 255.0) * 65535.0
    uint16_array = scaled_array.astype(np.uint16)

    return uint16_array


def uint16_to_uint8(uint16_array):
    # ��uint16����ת��Ϊuint8����
    uint8_array = (uint16_array / 65535.0 * 255).astype(np.uint8)
    return uint8_array


def adaptive_threshold_enhancement(image):
    # ����Ӧ��ֵ����
    thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresholded

def contrast_enhancement(image):
    # �ֲ��Աȶ���ǿ (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return enhanced

def process_frame(image):

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            # ����Ӧ��ֵ����
            thresholded = adaptive_threshold_enhancement(img)
            # �ֲ��Աȶ���ǿ
            enhanced = contrast_enhancement(thresholded)
            image[batch, ..., channel] = enhanced
    return image
    


def average_intensity(image):
    intensity_sum = 0
    mean_intensity = 0

    # ����ͼ�񼯺ϵ���ǿ��
    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            intensity_sum += np.mean(img)
    
    mean_intensity = intensity_sum / (image.shape[0] * image.shape[-1])

    # ����ƽ��ǿ�Ƚ�������ǿ�ȵ���
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
    # ����Դͼ���Ŀ��ͼ���ֱ��ͼ  
    source_hist, _ = np.histogram(source.ravel(), 256, [0, 256])  
    template_hist, _ = np.histogram(template.ravel(), 256, [0, 256])  
  
    # ��һ��ֱ��ͼ  
    source_hist = source_hist.astype('float')  
    template_hist = template_hist.astype('float')  
    source_hist /= (source_hist.sum() + 1e-7)  # ���������  
    template_hist /= (template_hist.sum() + 1e-7)  
  
    # �����ۻ��ֲ�������CDF��  
    source_cdf = source_hist.cumsum()  
    template_cdf = template_hist.cumsum()  
  
    # ����ӳ���  
    mapping = np.zeros(256)  
    for i in range(256):  
        # �ҵ���ӽ����ۻ��ֲ�ֵ  
        diff = template_cdf - source_cdf[i]  
        idx = np.argmin(np.abs(diff))  
        mapping[i] = idx  
  
    # Ӧ��ӳ���Դͼ��  
    matched = np.interp(source.ravel(), np.arange(256), mapping)  
    matched = matched.reshape(source.shape)  
  
    return matched.astype('uint8')  

# In[82]:


root_path = os.path.join(sys.argv[2],sys.argv[3])
file_name = sys.argv[3]+r'.tif'

img_raw = skio.imread(root_path +'/01/'+ file_name, plugin="tifffile")
# img_raw = skio.imread(root_path + file_name, plugin="tifffile")
# print(img_raw.shape)


#��ȡһϵ��ͼ��

# root_path = r'/data/sunrui/celldata/20230824_HBEC_test_DL/10%Laser_300ms_1x1bin/'
# raw_path = root_path#ԭʼͼ��·��
# imgfiles = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if f.endswith('.tif') or f.endswith('.tiff')]
# imgfiles.sort()

# img_raw = []
# print(len(imgfiles))
# for i in range(len(imgfiles)):
#     img_raw.append(skio.imread(imgfiles[i]).astype(np.uint16))
# print(len(img_raw))
img_raw = np.array(img_raw)
# print(img_raw.shape)
# print(img_raw.shape[0])

# In[83]:


# plt.imshow(img_raw[2])
# x = img_raw[2]
# print(x.mean())
# print(x.max())
# print(x.sort)

# y = img_raw[2]
# print(y.mean())
# print(y.max())
# print(y.sort)
# plt.show()

# In[84]:


# import numpy as np
# import matplotlib.pyplot as plt

# # ���� img_raw ��ͼ������

# plt.subplot(2, 2, 1)
# plt.imshow(img_raw[1])
# x = img_raw[1]
# print("Mean of x:", x.mean())
# print("Max of x:", x.max())

# plt.subplot(2, 2, 2)
# plt.imshow(img_raw[2])
# y = img_raw[2]
# print("Mean of y:", y.mean())
# print("Max of y:", y.max())
# y = remove_hight_variance_pixels(img_raw)
# # max_row, max_col = np.where(y == y.max())
# # print(max_col)
# # print(max_row)
# # plt.plot(max_col, max_row, 'ro')  # �������ֵλ�õĺ�ɫԲ��

# # plt.subplot(2, 2, 4)
# # plt.imshow(img_raw[2])
# # y.sort()  # ������ y ��������
# # print("Sorted y:", y)
# plt.show()


# In[85]:


# import numpy as np
# import matplotlib.pyplot as plt

# # ����һ�����ͼ����Ϊʾ��
# image = img_raw[2]

# # �ҵ����Ҷ�ֵ��λ��
# max_row, max_col = np.unravel_index(np.argmax(image), image.shape)

# # ��ʾͼ������ֵ��
# plt.imshow(image, cmap='gray')
# plt.scatter(max_col, max_row, c='red', marker='o', s=50, label='Max Value')
# plt.legend()
# plt.show()

# print("Max Gray Value:", image[max_row, max_col])
# print("Max Value Location (Row, Col):", max_row, max_col)


# In[86]:


# �����
# һ����������
# ����ͼƬ Load Image (single)
# x = skio.imread('D:\/new_folder\human_f01_stack\max_human_stack_30_100.tif', plugin="tifffile")
#file_name = 'r07c05f02.tif kept stack.tif'
# x = skio.imread(r'D:\new_folder\deepcell-tf-master\mySegment\100 slices\/'+file_name, plugin="tifffile")
#x = skio.imread(r'D:\new_folder\deepcell-tf-master\my_post_process\new_cell_hesc\r06c02f01-ch1.tif', plugin="tifffile")

# root_path = r'/data/sunrui/celldata/r03c03f01/'
# file_name = r'r03c03f01.tif'

# x = skio.imread(root_path + file_name, plugin="tifffile")
# print(x.shape)


# x = skio.imread(r'/data/sunrui/celldata/20230414_hesc_Mixl1_yy/r03c03f01/r03c03f01.tif',plugin="tifffile")
# print(x.shape)
# x = np.expand_dims(x,axis=3)
# # x = np.expand_dims(x,axis=3)
# print(x.dtype)
#plt.figure(figsize=(20, 30))

x = img_raw.copy()
# print(x.shape)
# x = np.expand_dims(x,axis=2)
x = np.expand_dims(x,axis=3)
# print('x:')
# print(x.shape)
# print(x.dtype)
# print('.............................')
# x = x[1:20,:,:,:]
#plt.subplot(1,7,1)
#plt.imshow(x[0,...,0])

batch = 50
total_frames = x.shape[0]
# total_frames = 50
img_pre_save = np.zeros(x.shape,dtype='float32')
# print(img_pre_save.shape)
# print(img_pre_save.dtype)
# x = x.astype('uint8')
# img_pre = background_noise(x)
# img_pre = blur_proc(img_pre)
# img_pre = his(x)
# img_pre = his(x)
#ֱ��ͼ���⻯
# img_pre=histogram_normalization(x)
# print('histogram_normalization DONE!')
#��ʴ����
# structuring_element = np.ones((3, 3), np.uint8)
# img_pre = grayscale_erosion(img_pre, structuring_element)
# print('grayscale_erosion DONE!')
#Canny��Ե���

target_image = x[0]


## ���ڳ���ͼ����
for i in range(0, total_frames, batch):
    # mean_intensity = 0
    img_raw0 = x[i:i+batch,:,:,:]
    # img_raw = x[1:10,:,:]
    # print(i)
    # print('img_raw:')
    # print(img_raw.dtype)
    # print(img_raw.shape)
    #plt.subplot(1,7,2)
    #plt.imshow(img_raw0[0,...,0])  
    # print('.....................')


    img_pre = img_raw0
    #img_pre = background_noise(img_pre)
    #img_pre = hist_match(img_pre, target_image)  

    #img_pre = remove_hight_variance_pixels(img_pre)
    #img_pre = histogram_normalization(img_pre)
    # print('histogram_normalization DONE!')
    # print('img_his:')
    # print(img_pre.dtype)
    # print(img_pre.shape)
    #plt.subplot(1,7,3)
    #plt.imshow(img_pre[0,...,0])  
    # print('.....................')

    # img_pre = (img_pre * 255).astype(np.uint8)
    # img_pre = float32_to_uint8(img_pre)
    threshold1 = 100
    threshold2 = 200
    # print('img_32_TO_8:')
    # print(img_pre.shape)
    # print(img_pre.dtype)
    # print('.....................')

    #img_pre = edge_enhancement_canny(img_pre, threshold1, threshold2)
    # print('img_edge_enhance:')
    # print(img_pre.dtype)
    # print(img_pre.shape)
    # img_pre=process_frame(img_pre)
    #plt.subplot(1,7,4)
    #plt.imshow(img_pre[0,...,0])  
    # print('.....................')

    img_pre = uint8_to_uint16(img_pre)
    # print('img_8_TO_16:')
    # print(img_pre.dtype)
    # print(img_pre.shape)
    # print('.....................')
    
    # img_pre = background_noise(img_pre)
    # print('img_backgroundNoise:')
    # print(img_pre.dtype)
    # print(img_pre.shape)
    #plt.subplot(1,7,5)
    #plt.imshow(img_pre[0,...,0])  
    # print('.....................')

    # img_pre = blur_proc(img_pre)
    # print('img_blur_proc:')
    # print(img_pre.dtype)
    # print(img_pre.shape)

    
    #plt.subplot(1,7,6)
    #plt.imshow(img_pre[0,...,0])  
    # print('.....................')
    # for io in range(img_pre.shape[0]):
    #     plt.subplot(1,2,1)
    #     plt.imshow(img_pre[io,...,0])
    #     plt.show()

    img_pre_save[i:i+batch,:,:,:] = img_pre


# print(img_pre.dtype)
# img_pre = (img_pre * 255).astype(np.uint8)

# print(img_pre.shape)
# threshold1 = 100
# threshold2 = 200

# img_pre = edge_enhancement_canny(img_pre, threshold1, threshold2)
# print(img_pre.shape)


# img_pre = img_pre.astype('uint16')
# չʾ������

print('img_pre_save:')
print(img_pre_save.dtype)
print(img_pre_save.shape)
img_pre_save = float32_to_uint8(img_pre_save)
# img_pre_save = average_intensity(img_pre_save)
print(img_pre_save.dtype)
print(img_pre_save.shape)
#plt.subplot(1,7,7)
#plt.imshow(img_pre_save[2,...,0])
#plt.show()

# In[87]:



# plt.figure(figsize=(20, 30))

# x = img_raw.copy()
# x = np.expand_dims(x,axis=3)
# print(x.shape)
# img_raw0 = x
# # img_raw = x[1:10,:,:]
# # print(i)
# print('img_raw:')
# print(img_raw0.dtype)
# print(img_raw0.shape)
# # plt.subplot(2,7,1)
# # plt.imshow(img_raw0[2,...,0])  
# # plt.subplot(2,7,8)
# # plt.imshow(img_raw0[1,...,0]) 
# # print('.....................')


# img_pre = img_raw0
# img_pre = remove_hight_variance_pixels(img_pre)
# plt.subplot(2,7,1)
# plt.imshow(img_pre[2,...,0])  
# plt.subplot(2,7,8)
# plt.imshow(img_pre[1,...,0]) 

# img_pre=histogram_normalization(img_pre)
# # print('histogram_normalization DONE!')
# # print('img_his:')
# # print(img_pre.dtype)
# # print(img_pre.shape)
# plt.subplot(2,7,2)
# plt.imshow(img_pre[2,...,0])  
# plt.subplot(2,7,9)
# plt.imshow(img_pre[1,...,0])  
# # print('.....................')

# # img_pre = (img_pre * 255).astype(np.uint8)
# img_pre = float32_to_uint8(img_pre)
# threshold1 = 100
# threshold2 = 200
# # print('img_32_TO_8:')
# # print(img_pre.shape)
# # print(img_pre.dtype)
# # print('.....................')

# img_pre = edge_enhancement_canny(img_pre, threshold1, threshold2)
# # print('img_edge_enhance:')
# # print(img_pre.dtype)
# # print(img_pre.shape)
# plt.subplot(2,7,3)
# plt.imshow(img_pre[2,...,0])  
# plt.subplot(2,7,10)
# plt.imshow(img_pre[1,...,0])  
# # print('.....................')

# img_pre = uint8_to_uint16(img_pre)
# # print('img_8_TO_16:')
# # print(img_pre.dtype)
# # print(img_pre.shape)
# # print('.....................')

# img_pre = background_noise(img_pre)
# # print('img_backgroundNoise:')
# # print(img_pre.dtype)
# # print(img_pre.shape)
# plt.subplot(2,7,4)
# plt.imshow(img_pre[2,...,0])  
# plt.subplot(2,7,11)
# plt.imshow(img_pre[1,...,0])  
# # print('.....................')

# img_pre = blur_proc(img_pre)
# # print('img_blur_proc:')
# # print(img_pre.dtype)
# # print(img_pre.shape)
# plt.subplot(2,7,5)
# plt.imshow(img_pre[2,...,0])  
# plt.subplot(2,7,12)
# plt.imshow(img_pre[1,...,0])  
# plt.show()
# print(img_pre.shape[0])

# # for i in range(img_pre.shape[0]):
# #     plt.subplot(1,2,1)
# #     plt.imshow(img_pre[i,...,0])
# #     plt.show()
# # print('.....................')

# # img_pre_save[i:i+batch,:,:,:] = img_pre

# In[88]:


# import cv2
# import glob

# # ��ȡ�Ҷ�ͼ�񣨵�ͨ��ͼ��
# # tiff_files = glob.glob(r'D:\ai\data\cell\SUNR\01/*.tif')
# # tiff_files = uint16_to_uint8(x)
# tiff_files = img_pre_save

# # ���������Ƶ��֡�ʺͱ������
# frame_rate = 30.0
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# # ���������Ƶд�����
# output_video = None

# # ��֡����Ҷ�ͼ��д����Ƶ
# for i, tiff_file in enumerate(tiff_files):
#     # image = cv2.imread(tiff_file, cv2.IMREAD_GRAYSCALE)

#     # ��ͼ�����ֱ��ͼ���⻯
    
#     image = tiff_files[i].squeeze()
#     # enhanced_image = cv2.equalizeHist(image)
#     enhanced_image = image
#     # print(enhanced_image.shape)

#     # ����ǵ�һ֡������ݵ�һ��ͼ�񴴽������Ƶ����
#     if i == 0:
#         height, width = enhanced_image.shape
#         output_video = cv2.VideoWriter(root_path + r'test.mp4', fourcc, frame_rate, (width, height))

#     # д����ǿ���ͼ����Ƶ�ļ�
#     output_video.write(cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR))

# # �ͷ���Դ���ر������Ƶ�ļ�
# output_video.release()

# print('��Ƶ�ļ��ѵ����ɹ���')

# In[89]:


# ������
# ���沽����д������ͼƬ
from tifffile import imread, imwrite
saveDir = root_path + r'/PRE/'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
imwrite(saveDir+"test.tif",img_pre_save)

saveDir_mul = root_path + r'/PRE/PRE_MUL/'
if not os.path.exists(saveDir_mul):
    os.makedirs(saveDir_mul)
for i,ids in enumerate(img_pre_save):
    image_path = os.path.join(saveDir_mul, f'test_{i:03d}.tif')
    imwrite(image_path, ids)

# In[90]:


# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import cv2

# In[91]:


# def show_anns(anns):
#     if len(anns) == 0:
#         return
#     # �������С��ע�ͽ�������
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     img_shape = (sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1],1)


    
#     # ����һ���հ�ͼ��
#     # img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
#     img = np.zeros(img_shape, dtype=np.uint8)
#     img[:,:,0] = 0
#     for i, ann in enumerate(sorted_anns):
#         mask_thresholdU = 5000
#         mask_thresholdD = 1000
#         mask_area = ann['area']
#         m = ann['segmentation']
#         if(mask_area < mask_thresholdU and mask_area > mask_thresholdD):
#             # gray_value = 2*i + 1
#             gray_value = i + 1
#             # color_mask = np.concatenate([np.random.random(3), [0.35]])
#             img[m] = gray_value
        
        
#         # print(img.shape)
#     img = img
    
#     return img

# In[92]:


# import sys
# sys.path.append("..")
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# sam_checkpoint = "sam_vit_h_4b8939.pth"
# model_type = "vit_h"

# device = "cuda"

# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)

# mask_generator = SamAutomaticMaskGenerator(
#     sam,
#     pred_iou_thresh = 0.95,
#     crop_overlap_ratio=0,
#     min_mask_region_area = 1000,
#     )

# In[93]:


# # image = cv2.imread('images/dog.jpg')
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# import skimage.io as skio
# from tifffile import imread, imwrite

# # #��ȡһϵ��ͼ��
# # raw_path = root_path#ԭʼͼ��·��
# # imgfiles = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if f.endswith('.tif') or f.endswith('.tiff')]
# # imgfiles.sort()

# # img_raw = []
# # print(len(imgfiles))
# # for i in range(len(imgfiles)):
# #     img_raw.append(skio.imread(imgfiles[i]).astype(np.uint16))
# # print(len(img_raw))
# # img_raw = np.array(img_raw)
# # print(img_raw.shape)
# # print(img_raw.shape[0])


# img_raw = skio.imread(root_path + r'/PRE/test.tif',plugin="tifffile")
# # img_raw = skio.imread(root_path + r'/PRE/test_seg.tif',plugin="tifffile")
# print(img_raw.shape)
# # img_raw = np.expand_dims(img_raw,axis=3)
# # x = np.expand_dims(x,axis=3)
# print(img_raw.dtype)

# # #uint16תuint8
# # max_val = np.max(img_raw)
# # min_val = np.min(img_raw)

# # # ��һ������
# # arr_normalized = (img_raw - min_val) * 255 / (max_val - min_val)

# # # ����һ���������ת��Ϊ uint8 ����
# # arr_uint8 = arr_normalized.astype(np.uint8)

# img = img_raw


# In[94]:



# image_folder =root_path + r'/01_GT/SAMSEG/'
# if not os.path.exists(image_folder):
#     os.makedirs(image_folder)

# for i in range(img.shape[0]):
# # for i in range(1):
#     image = img[i]
#     #uint16ת��Ϊuint8
#     # uint16_img = image
#     # uint16_img -= image.min()
#     # uint16_img = uint16_img / (uint16_img.max()-uint16_img.min())
#     # uint16_img *= 255
#     # uint8_img = np.uint8(uint16_img)

#     image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

#     # image = image.astype('uint8')
#     # image = cv2.imread(r'/data/sunrui/celldata/20230414_hesc_Mixl1_yy/r03c03f01/01/t000.tif')
#     # image = imread(r'/data/sunrui/celldata/20230414_hesc_Mixl1_yy/r03c03f01/01/t000.tif')

#     # print(image.dtype)
#     # print(image.shape)
#     # print('-------------------------------------------------------')

#     #����masks
#     masks = mask_generator.generate(image)
#     mask_result = show_anns(masks)
#     # print(mask_result.shape)

#     # ͼƬ������ļ���·��
    
#     image_path = os.path.join(image_folder, f'man_seg{i:03d}.tif')
#     imwrite(image_path,mask_result)

# In[95]:


# print(mask_result.shape)
# print(mask_result.dtype)

# In[96]:



# import cv2
# import numpy as np
# import skimage.io as skio
# import os

# In[97]:



# # ��ȡһϵ��ͼ��
# raw_path = root_path + '/01_GT/SAMSEG/'  # ԭʼͼ��·��
# imgfiles = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if f.endswith('.tif') or f.endswith('.tiff')]
# imgfiles.sort()

# img_raw = []

# print(len(imgfiles))
# for i in range(len(imgfiles)):
#     img_raw.append(skio.imread(imgfiles[i]).astype(np.uint16))
# print(len(img_raw))
# img_raw = np.array(img_raw)
# print(img_raw.shape)

# # ������ͼ�����д�С��ͬ�Ŀհ�ͼ��
# img_post = np.zeros(img_raw.shape, dtype=np.uint8)
# print(img_post.shape)

# # ����ͼ������
# for i in range(img_raw.shape[0]):
#     image = img_raw[i, :, :, 0]

#     # ��ȡͼ�����С�Ҷ�ֵ�����Ҷ�ֵ
#     min_intensity = np.min(image)
#     max_intensity = np.max(image)

#     # �����Ҷ�ֵ��Χ
#     for intensity in range(min_intensity, max_intensity + 1):
#         # ����һ����ԭʼͼ���С��ͬ�Ŀհ�ͼ��
#         gray_value_image = (image == intensity).astype('uint8') * 255

#         # ����ṹԪ�أ�����ʹ��3x3�������νṹԪ�أ�
#         structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

#         # ���и�ʴ����
#         eroded_image_0 = cv2.erode(gray_value_image, structuring_element)
#         eroded_image_1 = cv2.erode(eroded_image_0, structuring_element)
#         # �������Ͳ���
#         dilated_image_0 = cv2.dilate(eroded_image_1, structuring_element)
#         dilated_image_1 = cv2.dilate(dilated_image_0, structuring_element)

#         # ����������ص�������PRE��
#         non_zero_pixels_pre = np.count_nonzero(gray_value_image)
#         # print('pre:')
#         # print(non_zero_pixels_pre)

#         # ����������ص�������POST��
#         non_zero_pixels_post = np.count_nonzero(dilated_image_1)
#         # print('post:')
#         # print(non_zero_pixels_post)

#         # ���
#         post_result = (dilated_image_1 / 255) * intensity
#         post_result = post_result.astype('uint8')
#         ss = np.max(post_result)
#         # print('label:')
#         # print(ss)
#         # print('...................................................')

#         # ����img_post
#         if non_zero_pixels_post > 1000:
#             img_post[i, :, :, 0] += post_result


# In[98]:


# print(img_post.shape)
# print(img_post.dtype)
# img_post_16 = img_post.astype(np.uint16)

# In[99]:


# print(np.max(img_post_16))
# print(np.mean(img_post_16))
# print(np.min(img_post_16))

# In[100]:


# import skimage.io as skio
# from tifffile import imread, imwrite
# import matplotlib.pyplot as plt

# image_folder =root_path + r'/01_GT/SEG/'
# if not os.path.exists(image_folder):
#     os.makedirs(image_folder)

# for i in range(img_post.shape[0]):
#     image_path = os.path.join(image_folder, f'man_seg{i:03d}.tif')
#     imwrite(image_path,img_post_16[i,:,:,0])


# plt.imshow(img_post_16[14,:,:,0])
# plt.show()
