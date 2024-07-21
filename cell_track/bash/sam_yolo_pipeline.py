###load
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1] #sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import json
 

import random
import json, os
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt


from mmdet.apis import init_detector, inference_detector

import sys
sys.path.append("..")
# from segment_anything import sam_model_registry, SamPredictor
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

import pandas as pd

import skimage.io as skio
from tifffile import imread, imwrite


###定义函数
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=25):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def true_ratio(matrix):
    total_elements = matrix.size  # 获取矩阵总元素数量
    true_elements = np.sum(matrix)  # 计算True的数量

    ratio = true_elements / total_elements  # 计算True的占比
    return ratio

import numpy as np

def calculate_overlap(bbox1, bbox2):
    """
    计算两个边界框之间的重叠程度（IoU）

    参数：
    - bbox1: 第一个边界框，格式为 [x1, y1, x2, y2]，其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标
    - bbox2: 第二个边界框，格式为 [x1, y1, x2, y2]

    返回值：
    - IoU: 重叠程度，范围在 0 到 1 之间，值越高表示重叠程度越高
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # 计算重叠区域的面积
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # 计算两个边界框的面积
    area_bbox1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    area_bbox2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    # 计算重叠程度（IoU）
    iou = intersection / float(area_bbox1 + area_bbox2 - intersection)

    return iou

def check_distense(box_raw, img, threshold_overlap=0.5, threshold_distance=25,threshold_overlap_single=0.8):
    # 初始化列表以存储筛选后的边界框
    bbx_save = []
    area_all = []
    xy_all = np.zeros((2, len(box_raw)))  

    # box_raw = box_raw.detach().numpy()  # 如果是PyTorch的Tensor
    # 或者
    box_raw = np.array(box_raw) # 如果是TensorFlow的Tensor

    
    # 遍历每个边界框
    for i, ids in enumerate(box_raw):
        area = 0
        box = ids
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        
        # 计算面积以备后用
        area = w * h
        
        if area<250:
            continue

        area_all.append(area)  
        
        # 计算边界框的中心点
        x = x0 + w / 2
        y = y0 + h / 2
        
        # 确保中心点在图像范围内
        h_img, w_img = img.shape[:2]
        x = min(x, w_img)
        y = min(y, h_img)
        
        # 存储中心点坐标
        xy_all[0, i] = x
        xy_all[1, i] = y
        
        # 检查边界框之间是否存在重叠,
        overlap = False
        for bbx in bbx_save:
            # 计算交集面积
            x_left = max(bbx[0], x0)
            y_top = max(bbx[1], y0)
            x_right = min(bbx[2], box[2])
            y_bottom = min(bbx[3], box[3])


            bbx_x = bbx[0] + (bbx[2] - bbx[0])/2
            bbx_y = bbx[1] + (bbx[3] - bbx[1])/2


            distance = distance = np.sqrt((x - bbx_x)**2 + (y - bbx_y)**2)
            if distance < threshold_distance:
                # 判断是否为嵌套关系，是否全为内部，是则保留嵌套中小的mask
                if x_right > x_left and y_bottom > y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)#相交面积
                    bbx_area = (bbx[2] - bbx[0])*(bbx[3] - bbx[1])#bbx面积
                    iou = intersection_area / (area + bbx_area - intersection_area)#两者交并比
                    rate_1 = intersection_area/bbx_area #交集比单个mask（bbx）
                    rate_2 = intersection_area/area #交集比单个mask（bbox）
                    area_1 = bbx_area
                    area_2 = area
                    
                    if rate_2 > threshold_overlap_single and bbx_area > area:
                        # print('rate_2:',rate_2)
                        # print('rate_1:',rate_1)
                        if (bbx_save == bbx).all():  # 使用 .all() 方法检查数组中的所有元素是否与当前边界框匹配
                            bbx_save.remove(bbx)
                        # bbx_save.remove(bbx)
                    elif rate_1 > threshold_overlap_single and bbx_area < area:
                        # print('rate_2:',rate_2)
                        # print('rate_1:',rate_1)
                        overlap = True
                        break
        
        # 如果边界框通过了以上两个条件，则将其添加到保存的筛选后边界框列表中
        if not overlap :
            bbx_save.append(ids)
    
    return bbx_save

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
        mask_thresholdD = 80
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



def mask_to_bbox(mask):
    # 检测边界
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # bboxes = []
    for contour in contours:
        # 计算边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 转换为标准边界框格式 [x1, y1, x2, y2]
        bbox = [x, y, x + w, y + h]
        
        # bboxes.append(bbox)

    return bbox
###创建保存文件夹

# root_path = r'/home/sunrui/cell_track/YOLO_SAM/1110V6_T358/'
root_path = os.path.join(sys.argv[2],sys.argv[3],'PRE/PRE_MUL/')
# root_path = os.path.join('/mnt/sda/cell_data/240120H9D3_F001/PRE/PRE_MUL/')

#test.tif --> multi test_id.tif: 写进sampre里面吧


# image_folder =root_path + r'/result/'
path_output = os.path.join(sys.argv[2],sys.argv[3],'01_GT/SAMSEG/')
# path_output = os.path.join('/mnt/sda/cell_data/240120H9D3_F001/01_GT/SAMSEG/')

if not os.path.exists(path_output):
    os.makedirs(path_output)
# if not os.path.exists(image_folder):
#     os.makedirs(image_folder)
device = "cuda"
###加载检测模型
config_file = '/home/sunrui/cell_track/cell_track/bash/config.py'
checkpoint_file = '/home/sunrui/cell_track/cell_track/bash/best_coco_bbox_mAP_epoch_180.pth'
model = init_detector(config_file, checkpoint_file, device=device)  # or device='cuda:0'


###加载分割器

sam_checkpoint = "/home/sunrui/cell_track/cell_track/bash/segment-anything/notebooks/sam_vit_h_4b8939.pth"
model_type = "vit_h"



sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


predictor = SamPredictor(sam)#加参数
# mask_generator = SamAutomaticMaskGenerator(sam,pred_iou_thresh=0.95)#


###读取数据文件
#所有pre图片的路径
data_path = [os.path.join(root_path, dir) for dir in os.listdir(root_path)]
data_path.sort()
#所有图片的名称
img_raw_name = [dir for dir in os.listdir(root_path)]
img_raw_name.sort()


#分割第一帧：
frame=0
img_path = data_path[0]
print('img_id:',frame)
f_name = 'test_'

t1 = inference_detector(model,img_path)
bbox_raw = t1.pred_instances.bboxes
score = t1.pred_instances.scores

# bbox_raw = np.array(bbox_raw) #YOLOV8 box
bbox_raw = bbox_raw.cpu().numpy()

bbox = []

for i in range(bbox_raw.shape[0]):
    if score[i]<0.09:
        continue
    bbox.append(bbox_raw[i])

# img=skio.imread(data_path[0],plugin="tifffile")
img = cv2.imread(data_path[0])
img = np.array(img)
# img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

# plt.figure(figsize=(10, 10))
# plt.subplot(1,2,1)
# plt.imshow(img)

mask_zero = np.zeros((img.shape[0],img.shape[1],1),dtype=np.uint16)
mask_zero[:,:,0] = 0
predictor.set_image(img)

bbox = check_distense(bbox,img)
bbox = np.array(bbox) #YOLOV8 box
bbox_save = []

if len(bbox) == 0 :
    print('Error! No Result')

for i in range(bbox.shape[0]):
    # if score[i]<0.09:
    #     continue

    bbx = torch.tensor(bbox[i],device=predictor.device)
    # show_box(bbx.cpu().numpy(),plt.gca())
    # bbx.cpu().numpy()
    # bbx_c = bbx.cpu().numpy()
    bbx = bbox[i]
    # score_l =score[i]
    # show_box(bbx,plt.gca())

    # [x,y]=[(bbx_c[0]+bbx_c[2])/2,(bbx_c[1]+(bbx_c[3])/2)]
    x0, y0 = bbx[0], bbx[1]
    w, h = bbx[2] - bbx[0], bbx[3] - bbx[1]
    x = x0 + w/2
    y = y0 + h/2

    h_img = img.shape[0]-1
    w_img = img.shape[1]-1

    x = min(x,w_img)
    y = min(y,h_img)

    input_point = np.array([[x, y]])
    input_label = np.array([1])

    # show_points(input_point,input_label, plt.gca())
    
    # print('x:',x,'y:',y)
    int_cent = img[int(y)][int(x)].sum()
    # int_cent = 4

    if int_cent <=3 :
        continue

    # print(bbx)
    # print(score_l)

    input_point = np.array([[x, y]])
    input_label = np.array([1])
    input_box = np.array(bbx)

    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=False,
    )
    
    if scores[0] < 0.1 :
    
        continue
    num_pixels = np.sum(masks[0])
    thresholdU = 12000
    thresholdD = 80
        
    if num_pixels > thresholdU or num_pixels < thresholdD :
        continue
    # show_mask(masks[0], plt.gca(),random_color=True)
         # gray_value = 2*i + 1
    # bbox_save.append(bbx)
    gray_value = int(i) + 1
    # color_mask = np.concatenate([np.random.random(3), [0.35]])
    mask_zero[masks[0]] = gray_value   
    mask_uint8 = masks[0].astype(np.uint8)
    bbox_save.append(mask_to_bbox(mask_uint8))
    


    # mask_zero[masks[0]]=True

# plt.subplot(1,2,2)
# plt.imshow(mask_zero)
image_path = os.path.join(path_output, f'man_seg{frame:04d}.tif')
imwrite(image_path, mask_zero)
print(f'save successful man_seg{frame:04d}.tif')
save_image_folder = path_output
# plt.subplot(1,2,2)
# plt.imshow(mask_zero)
# plt.savefig(save_image_folder+ 'plt' + f'{frame:03d}' +'.tif')

# rate = true_ratio(mask_zero)*100
# print(format(rate, '.4f'))
# print(rate)
# torch.cuda.empty_cache()
#分割后续帧数：
for frame,img_path in enumerate(data_path):
    if frame == 0 :
        continue
    # 首先生成bbox
    print('img_id:',frame)
    f_name = 'test_'

    t1 = inference_detector(model,img_path)
    bbox_raw = t1.pred_instances.bboxes
    score = t1.pred_instances.scores

    # bbox_raw = np.array(bbox_raw) #YOLOV8 box
    bbox_raw = bbox_raw.cpu().numpy()

    bbox = []

    for i in range(bbox_raw.shape[0]):
        if score[i]<0.09:
            continue
        bbox.append(bbox_raw[i])

    # img=skio.imread(img_path,plugin="tifffile")
    img = cv2.imread(img_path)
    img = np.array(img)
    #plt.figure(figsize=(10, 10))
    #plt.subplot(1,2,1)
    #plt.imshow(img)
    #img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    mask_zero = np.zeros((img.shape[0],img.shape[1],1),dtype=np.uint16)
    mask_zero[:,:,0] = 0

    predictor.set_image(img)
    bbox = check_distense(bbox,img)
    bbox = np.array(bbox)
    # filtered_bboxes = []
    # for bbox_temp in bbox:#所有第二帧的box
    #     filtered_bboxes.append(bbox_temp)
    #     overlap = False
    #     for prev_bbox in bbox_save: # sam结果里面第一帧的bbox
    #         # print(bbox_temp)
    #         # print(prev_bbox)
    #         iou = calculate_overlap(bbox_temp,prev_bbox) # 计算重叠率
    #         if iou > 0.5 :  #box重叠率
    #             overlap = True
    #             break
    #     if not overlap:
    #         filtered_bboxes.append(prev_bbox)
    filtered_bboxes = np.zeros_like(bbox)
    filtered_bboxes = bbox.copy()
    for prev_bbox in bbox_save:#第一帧
        overlap = False
        for bbox_temp in bbox:#第二帧
            iou = calculate_overlap(bbox_temp,prev_bbox)
            if iou > 0.2:
                overlap = True
                break
        if not overlap:
            # print(filtered_bboxes.shape)
            filtered_bboxes = np.concatenate((filtered_bboxes, [prev_bbox]))
            # print(filtered_bboxes.shape)
        pass

    bbox_save = [] # 清空保存文件
    # filtered_bboxes # 所有待分割的bbox
    if len(filtered_bboxes) == 0 :
        print('Error! No Result')
    # print(len(filtered_bboxes))

    filtered_bboxes = np.array(filtered_bboxes)
    # print(filtered_bboxes.shape[0])
    for i in range(filtered_bboxes.shape[0]):  #使用box分割图像并获得分割图的box
        # if score[i]<0.09:
        #     continue

        bbx = torch.tensor(filtered_bboxes[i],device=predictor.device)
        # show_box(bbx.cpu().numpy(),plt.gca())
        # bbx.cpu().numpy()
        # bbx_c = bbx.cpu().numpy()
        bbx = filtered_bboxes[i]
        # score_l =score[i]
        # show_box(bbx,plt.gca())

        # [x,y]=[(bbx_c[0]+bbx_c[2])/2,(bbx_c[1]+(bbx_c[3])/2)]
        x0, y0 = bbx[0], bbx[1]
        w, h = bbx[2] - bbx[0], bbx[3] - bbx[1]
        x = x0 + w/2
        y = y0 + h/2

        h_img = img.shape[0]-1
        w_img = img.shape[1]-1

        x = min(x,w_img)
        y = min(y,h_img)

        input_point = np.array([[x, y]])
        input_label = np.array([1])

        # show_points(input_point,input_label, plt.gca())
        
        # print('x:',x,'y:',y)
        int_cent = img[int(y)][int(x)].sum()
        # int_cent = 4

        if int_cent <=3 :
            continue

        # print(bbx)
        # print(score_l)

        input_point = np.array([[x, y]])
        input_label = np.array([1])
        input_box = np.array(bbx)

        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )
        
        if scores[0] < 0.1 :
        
            continue
            
        #print(masks[0].dtype)
        num_pixels = np.sum(masks[0])
        thresholdU = 12000
        thresholdD = 80
        #print(masks[0].dtype)
        if num_pixels > thresholdU or num_pixels < thresholdD :
            #print(num_pixels)
            continue
        #xiugai
        # show_mask(masks[0], plt.gca(),random_color=True)
            # gray_value = 2*i + 1
        # bbox_save.append(bbx)
        gray_value = int(i) + 1
        # color_mask = np.concatenate([np.random.random(3), [0.35]])
        
        mask_zero[masks[0]] = gray_value
        mask_uint8 = masks[0].astype(np.uint8)
        if np.any(mask_uint8):
            bbox_save.append(mask_to_bbox(mask_uint8))   
        

    #有新的bbox结果，需要和上一帧作对比，怎么排除
    # 从生成的结果图像里面找到bbox
    # 对比新旧的bbox重叠率，增加缺失的bbox
    # 使用新的bbox进行分割
    # 旧 = 新bbox
    #使用确定有bbox的框作对比
    image_path = os.path.join(path_output, f'man_seg{frame:04d}.tif')
    imwrite(image_path, mask_zero)
    print(f'save successful man_seg{frame:04d}.tif')
    #plt.subplot(1,2,2)
    #plt.imshow(mask_zero)
    #if not os.path.exists(save_image_folder+ '/plt/'):
    #    os.makedirs(save_image_folder+ '/plt/')
    #plt.savefig(save_image_folder+ '/plt/' + f'{frame:03d}' +'.tif')
    # torch.cuda.empty_cache()

    pass

