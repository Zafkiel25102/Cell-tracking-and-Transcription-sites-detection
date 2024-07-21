#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

print('>>sam_pipe<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
#只需修改路径
path= os.path.join(sys.argv[2],sys.argv[3],'./PRE/test.tif')
path_output = os.path.join(sys.argv[2],sys.argv[3],'./01_GT/SAMSEG/')


if not os.path.exists(path_output):
    os.makedirs(path_output)


sam_checkpoint = "/home/wanyihanLab/sunrui03/cell_track/bash/segment-anything/notebooks/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

# sam = nn.DataParallel(sam)
sam.to(device=device)

predictor = SamPredictor(sam)#加参数
mask_generator = SamAutomaticMaskGenerator(sam,pred_iou_thresh=0.95)#

#
def prompts_of_disappear_masks(masks_old, masks_new_auto):
    prompts = []
    masks_new_auto_result = np.zeros((masks_new_auto[0]['segmentation'].shape[0], masks_new_auto[0]['segmentation'].shape[1]))       #1000*1000
    for mask in masks_new_auto:
        masks_new_auto_result = np.logical_or(masks_new_auto_result, mask['segmentation'])
    for mask in masks_old:
        mask_use = np.array(mask['bbox'])
        box=mask_use.astype(int)
        if not masks_new_auto_result[box[1]+box[3]//2, box[0]+box[2]//2]:
            prompts.append([box[0]+box[2]//2, box[1]+box[3]//2])
    return prompts

def get_masks_new(image, prompts, masks_new_auto):
    #masks_new = masks_new_auto.copy()
    predictor.set_image(image)
    for prompt in prompts:
        input_point = np.array([prompt])
        input_label = np.array([1])
        #mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

        mask, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        #mask_input=mask_input[None, :, :],
        multimask_output=False,
        )
        mask=mask.squeeze()
        xyxy = np.where(mask)
        # x1,y1,x2,y2 = xyxy[0].min(), xyxy[1].min(), xyxy[0].max(), xyxy[1].max()          #这里也是矩阵，到下面应该是图像中的坐标了
        x1,y1,x2,y2 = xyxy[1].min(), xyxy[0].min(), xyxy[1].max(), xyxy[0].max()

        bbox= [x1,y1,x2-x1,y2-y1]
        area= len(xyxy[0])

        masks_new_auto.append(
            {
                'segmentation':mask,
                'bbox':bbox,
                'area':area,
                'prompt':prompt
            }
        )
    return masks_new_auto

def save_anns(anns):
    if len(anns) == 0:
        return
    # 按面积大小对注释进行排序
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img_shape = (sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1],1)        #1000*1000*1
    
    # 创建一个空白图像
    # img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img = np.zeros(img_shape, dtype=np.uint16)
    img[:,:,0] = 0
    for i, ann in enumerate(sorted_anns):
        mask_thresholdU = 12000
        mask_thresholdD = 230
        mask_area = ann['area']
        m = ann['segmentation']         #python中1和True，0和False是一回事儿，m就是二值灰度图,假的！！！不是一回事儿
        # “is”和“==”的含义不同，“1”和“True”虽然数值相同，但是id不同。

        # #后处理
        # structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # # 进行腐蚀操作
        # eroded_image_0 = cv2.erode(m.astype('uint8'), structuring_element)              #得astype
        # eroded_image_1 = cv2.erode(eroded_image_0, structuring_element)
        # # 进行膨胀操作
        # dilated_image_0 = cv2.dilate(eroded_image_1, structuring_element)
        # dilated_image_1 = cv2.dilate(dilated_image_0, structuring_element).astype('bool')# 这也好像得astype

        if(mask_area < mask_thresholdU and mask_area > mask_thresholdD):
            # gray_value = 2*i + 1
            gray_value = i + 1
            # color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = gray_value                               #这是广播吗？
            #当前维度的值相等。
            # 当前维度的值有一个是 1。触发广播
    return img

# In[2]:


import skimage.io as skio
from tifffile import imread, imwrite

img_raw = skio.imread(path,plugin="tifffile")

img=img_raw[0]
index=0
img = img.squeeze()
img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
masks_old = mask_generator.generate(img)
masks_new = masks_old
output_result = save_anns(masks_new)
# 图片保存的文件夹路径 + 名字
image_path = os.path.join(path_output, f'man_seg{index:03d}.tif')
imwrite(image_path, output_result)
print(f'save successful man_seg{index:03d}.tif')
# 在训练迭代中释放不需要的内存
torch.cuda.empty_cache()

prompts=[]
for img in img_raw[1:,:, :, : ]:
    index+=1
    img = img.squeeze()
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    masks_new_auto = mask_generator.generate(img)

    prompts = prompts_of_disappear_masks(masks_old, masks_new_auto)
    masks_new=get_masks_new(img, prompts, masks_new_auto)
    masks_old=masks_new
    
    output_result = save_anns(masks_new)
    # 图片保存的文件夹路径 + 名字
    image_path = os.path.join(path_output, f'man_seg{index:03d}.tif')
    imwrite(image_path, output_result)
    print(f'save successful man_seg{index:03d}.tif')
    torch.cuda.empty_cache()
