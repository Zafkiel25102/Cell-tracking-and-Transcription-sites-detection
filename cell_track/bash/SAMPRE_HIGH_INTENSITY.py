#!/usr/bin/env python
# coding: utf-8


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

from tifffile import imread, imwrite

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
                    min_value = float('inf')  

     
                    for m in range(h_se):
                        for n in range(w_se):
                            if structuring_element[m, n] == 1:
                            
                                if i + m >= 0 and i + m < h and j + n >= 0 and j + n < w:
                                    current_value = X[i + m, j + n]
                                    min_value = min(min_value, current_value)

            eroded_image[i, j] = min_value
            image[batch, ..., channel] = eroded_image
    return image




def edge_enhancement(input_image):

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


    gradient_x = cv2.filter2D(input_image, -1, sobel_x)
    gradient_y = cv2.filter2D(input_image, -1, sobel_y)


    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)


    enhanced_image = input_image + gradient_magnitude

    return enhanced_image





#
def edge_enhancement_canny(image, threshold1, threshold2):
    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            X = image[batch, ..., channel]
             
            edges = cv2.Canny(X, threshold1, threshold2)

          
            enhanced_image = X + 2*edges
            image[batch, ..., channel] = enhanced_image
    
    # 
    # edges = cv2.Canny(input_image, threshold1, threshold2)

    #
    # enhanced_image = input_image + edges

    return image


# threshold1 = 100
# threshold2 = 200


# enhanced_image = edge_enhancement_canny(input_image, threshold1, threshold2)


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

    # float_value = np.clip(float_value, 0.0, 1.0)
    
    # uint8_value = np.round(float_value * 255.0, decimals=0).astype(np.uint8)
    
    scaled_image = (float_value - np.min(float_value)) / (np.max(float_value) - np.min(float_value))
    uint8_value = np.round(scaled_image * 255.0).astype(np.uint8)

    return uint8_value


def uint8_to_uint16(uint8_array):

    # float_value = np.clip(float_value, 0.0, 1.0)
 
    # uint8_value = np.round(float_value * 255.0, decimals=0).astype(np.uint8)
        
    scaled_array = (uint8_array.astype(np.float32) / 255.0) * 65535.0
    uint16_array = scaled_array.astype(np.uint16)

    return uint16_array


def uint16_to_uint8(uint16_array):
   
    uint8_array = (uint16_array / 65535.0 * 255).astype(np.uint8)
    return uint8_array


def adaptive_threshold_enhancement(image):

    thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresholded

def contrast_enhancement(image):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return enhanced

def process_frame(image):

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]

            thresholded = adaptive_threshold_enhancement(img)

            enhanced = contrast_enhancement(thresholded)
            image[batch, ..., channel] = enhanced
    return image
    


def average_intensity(image):
    intensity_sum = 0
    mean_intensity = 0


    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]
            intensity_sum += np.mean(img)
    
    mean_intensity = intensity_sum / (image.shape[0] * image.shape[-1])


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
 
    source_hist, _ = np.histogram(source.ravel(), 256, [0, 256])  
    template_hist, _ = np.histogram(template.ravel(), 256, [0, 256])  
  

    source_hist = source_hist.astype('float')  
    template_hist = template_hist.astype('float')  
    source_hist /= (source_hist.sum() + 1e-7) 
    template_hist /= (template_hist.sum() + 1e-7)  
  

    source_cdf = source_hist.cumsum()  
    template_cdf = template_hist.cumsum()  
  
  
    mapping = np.zeros(256)  
    for i in range(256):  

        diff = template_cdf - source_cdf[i]  
        idx = np.argmin(np.abs(diff))  
        mapping[i] = idx  
  

    matched = np.interp(source.ravel(), np.arange(256), mapping)  
    matched = matched.reshape(source.shape)  
  
    return matched.astype('uint8')  


def main():

    print('>>SAMPRE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    root_path = os.path.join(sys.argv[2],sys.argv[3])
    file_name = sys.argv[3]+r'.tif'

    img_raw = skio.imread(root_path +'/01/'+ file_name, plugin="tifffile")
    img_raw = np.array(img_raw)


    x = img_raw.copy()


    x = np.expand_dims(x,axis=3)

    batch = 50
    total_frames = x.shape[0]
    # total_frames = 50
    img_pre_save = np.zeros(x.shape,dtype='float32')


    target_image = x[0]



    for i in range(0, total_frames, batch):

        img_raw0 = x[i:i+batch,:,:,:]

        img_pre = img_raw0
        threshold1 = 100
        threshold2 = 200

        img_pre = uint8_to_uint16(img_pre)

        img_pre_save[i:i+batch,:,:,:] = img_pre


    print('img_pre_save:')
    print(img_pre_save.dtype)
    print(img_pre_save.shape)
    img_pre_save = float32_to_uint8(img_pre_save)
    print(img_pre_save.dtype)
    print(img_pre_save.shape)



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

if __name__ == "__main__":
    main()