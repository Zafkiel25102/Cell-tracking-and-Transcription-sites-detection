#!/usr/bin/env python
# coding: utf-8

# In[137]:


import skimage.io as skio
import numpy as np
import cv2
from tifffile import imread, imwrite

import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage.measure import regionprops

# In[138]:


# x = skio.imread(r'D:\WORK\RNA\GAOSEG\GAOS\r04c06f01\SEG\post_seg.tif', plugin="tifffile")
# print("x:")
# print(x.shape)
# 1
import sys

print('>>single_fast<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
root_path = os.path.join(sys.argv[2],sys.argv[3])

GT_path = root_path + r'/01_GT/_RES/'#跟踪结果路径
files = [os.path.join(GT_path, f) for f in os.listdir(GT_path) if f.endswith('.tif') or f.endswith('.tiff')]
files.sort()

raw_path = root_path + r'/01/'#原始图像路径
imgfiles = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if f.endswith('.tif') or f.endswith('.tiff')]
imgfiles.sort()

# file_name =sys.argv[3]+ r'.tif'
# img_raw = skio.imread(root_path +'/' + file_name, plugin="tifffile")


#读取跟踪结果txt
track = np.genfromtxt(root_path + r"/01_GT/_RES/res_track.txt",dtype=[int, int, int, int])  # 将文件中数据加载到data数组里
# print(track.shape[0])
# print(track[0])
# print(track[0][2])
print(track.shape)
# print(np.expand_dims(track,axis=1).shape)
# track = np.expand_dims(track,axis=1)

# In[139]:


#img转换为Numpy数组
# io_num = 0
# print(track[io_num][2])
img = []
print(len(files))
for i in range(len(files)):
    img.append(skio.imread(files[i]).astype(np.uint32))
print(len(img))
img = np.array(img)
img = img.squeeze()
print(img.shape)
print(img.shape[0])
# # for i in range(img.shape[0]):
# #     #cv2.imshow(img[i])
# #     print(img[0])
# #     plt.imshow(img[0].astype("uint8"))
# plt.imshow(img[1].astype("uint8"))


# In[140]:


img_raw = []
print(len(imgfiles))
for i in range(len(imgfiles)):
    img_raw.append(skio.imread(imgfiles[i]).astype(np.uint16))
print(len(img_raw))
img_raw = np.array(img_raw)
img_raw = np.squeeze(img_raw)
print(img_raw.shape)
print(img_raw.shape[0])

# In[141]:


#生成CSV文件
cols = ["id","raw_id",
            "frame_num",
            "area",
            "min_row_bb", "min_col_bb", "max_row_bb", "max_col_bb",
            "centroid_row", "centroid_col",
            "major_axis_length", "minor_axis_length",
            "max_intensity", "mean_intensity", "min_intensity"]
print(cols)
num_labels = np.unique(img[0]).shape[0] - 1
df = pd.DataFrame(index=range(num_labels), columns=cols)

# In[142]:


# 生成CSV文件（包含单个细胞的数据）
for io_num in range(img.shape[0]):
    #print(io_num)
    result = img[io_num]
    result_raw = img_raw[io_num].copy()
    for ind, id_res in enumerate(np.unique(result)):
        row_ind = ind-1
        if id_res == 0:
            continue
        properties = regionprops(np.uint32(result == id_res), result_raw)[0]

        df.loc[row_ind, "id"] = id_res
        df.loc[row_ind, "raw_id"] = id_res
        df.loc[row_ind, "area"] = properties.area

        bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col = properties.bbox
        df.loc[row_ind, "min_row_bb"] = bbox_min_row
        df.loc[row_ind, "min_col_bb"] = bbox_min_col
        df.loc[row_ind, "max_row_bb"] = bbox_max_row
        df.loc[row_ind, "max_col_bb"] = bbox_max_col
        # df.loc[row_ind, "min_row_bb"], df.loc[row_ind, "min_col_bb"], \
        # df.loc[row_ind, "max_row_bb"], df.loc[row_ind, "max_col_bb"] = properties.bbox

        df.loc[row_ind, "centroid_row"], df.loc[row_ind, "centroid_col"] = \
            properties.centroid[0].round().astype(np.int32), \
            properties.centroid[1].round().astype(np.int32)

        df.loc[row_ind, "major_axis_length"], df.loc[row_ind, "minor_axis_length"] = \
            properties.major_axis_length, properties.minor_axis_length

        df.loc[row_ind, "max_intensity"], df.loc[row_ind, "mean_intensity"], df.loc[row_ind, "min_intensity"] = \
            properties.max_intensity, properties.mean_intensity, properties.min_intensity

    df.loc[:, "frame_num"] = int(io_num)
    df.to_csv(root_path + r'/01_GT/_RES/TRA_'+str(io_num)+'.csv', index=False)

# In[143]:


#track是跟踪结果txt，img是分割掩码，img_raw是原始图像
from tifffile import imread, imwrite
for io_num in range(track.shape[0]):
    #mask = x[io_num]
    #print(track[io_num][0])
    ti = track[io_num][1]
    te = track[io_num][2]+1
    #print("///")
    #print("cell_id:")
    #print(track[io_num][0])
    cell_id = track[io_num][0]
    #print("................")

    
#     wrow=0
#     wcol=0
#     padding_size = 15
    target = []
#     for t in range(ti,te):
#         print(t)
#         csv_f = pd.read_csv(r'D:\WORK\RNA\GAOSEG\GAOS\r04c06f01\r04c06f01\01_GT\SE_RES\TRA_'+str(t)+'.csv')
#         bb_col = csv_f[["id","min_row_bb","min_col_bb","max_row_bb","max_col_bb"]]
#         b_col = bb_col.loc[bb_col['id'] == cell_id]
#         bbx = np.array(b_col)
#         print(bbx)
#         if bbx[0,3]-bbx[0,1]>wrow:
#             wrow=bbx[0,3]-bbx[0,1]
#         if bbx[0,4]-bbx[0,2]>wcol:
#             wcol=bbx[0,4]-bbx[0,2]
#     wrow += 13
#     wcol += 13
#     print("w,h")
#     print(wrow)
#     print(wcol)
    wrow = 128
    wcol = 128
    #print("......................")
    empty_count = 0
    for t in range(ti,te):
        #print(t)
        #图像
        cur_frame = img_raw[t]
        cur_mask = img[t]
        #print(frame.shape)
        #覆盖掩膜
        curr_result_by_id = np.uint32(cur_mask == cell_id).copy()




        # 将 curr_result_by_id 转换为二值图像
        binary_image = np.uint16(curr_result_by_id > 0)

        # # 定义膨胀操作的核大小和迭代次数
        # kernel_size = 3
        # iterations = 1

        # # 创建膨胀操作的核
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # # 对二值图像进行膨胀操作
        # smoothed_image = cv2.dilate(binary_image, kernel, iterations=iterations)
        # smoothed_image_b = np.uint16(smoothed_image > 0)

        # frame = np.multiply(smoothed_image_b, cur_frame)
        
        frame = np.multiply(binary_image, cur_frame)
        
        #读取CSV文件
        csv_f = pd.read_csv(root_path + r'/01_GT/_RES/TRA_'+str(t)+'.csv')
        #print(csv_f)
        bb_col = csv_f[["id","min_row_bb","min_col_bb","max_row_bb","max_col_bb"]]
        b_col = bb_col.loc[bb_col['id'] == cell_id]
        bbx = np.array(b_col)
        if not np.any(curr_result_by_id):
            #print('empty1')
            image_size = (128, 128)
            black_image = np.zeros(image_size, dtype=np.uint16)
            target.append(np.expand_dims(black_image, axis=0))
            empty_count += 1
            continue
        # print(bbx)
        # 计算边框中心点坐标
        cy = (bbx[0, 2] + bbx[0, 4]) // 2
        cx = (bbx[0, 1] + bbx[0, 3]) // 2
            
        # 计算边框左上角和右下角坐标
        x1 = cx - wrow// 2
        y1 = cy - wcol// 2
        x2 = cx + wrow// 2
        y2 = cy + wcol// 2
        
        # # 判断边框是否超出图像边缘
        # if x1 < 0:
        #     continue
        #     x1 = 0
        #     x2 = wrow
        # if y1 < 0:
        #     continue
        #     y1 = 0
        #     y2 = wcol
        # if x2 > frame.shape[0]:
        #     continue
        #     x2 = frame.shape[0]
        #     x1 = x2 - wrow
        # if y2 > frame.shape[1]:
        #     continue
        #     y2 = frame.shape[1]
        #     y1 = y2 - wcol
         # 判断边框是否超出图像边缘
        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = 0
        if x1 < 0:
            pad_top = abs(x1)
            x1 = 0
        if y1 < 0:
            pad_left = abs(y1)
            y1 = 0
        if x2 > frame.shape[0]:
            pad_bottom = x2 - frame.shape[0]
            x2 = frame.shape[0]
        if y2 > frame.shape[1]:
            pad_right = y2 - frame.shape[1]
            y2 = frame.shape[1]

        # 提取图像块
        img1 = frame[x1:x2, y1:y2]
        
        #print(img1.shape)
        # 根据需要进行填充
        if pad_left or pad_right or pad_top or pad_bottom:
            img1 = np.pad(img1, ((pad_top,pad_bottom),(pad_left,pad_right)), mode='constant', constant_values=0)
        #print(img1.shape)
        # 提取图像
        # img1 = frame[x1:x2, y1:y2]
        target.append(np.expand_dims(img1, axis=0))
        
        #截取图像并加上边框
        #img1 = frame[bbx[0,1]-10:bbx[0,3]+10, bbx[0,2]-10:bbx[0,4]+10]
        #img1 = cv2.copyMakeBorder(img1, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0)
        # 调整图像大小为固定大小
        #     img1 = cv2.resize(img1, (128, 128))
        #     target.append(np.expand_dims(img1, axis=0))
            
#             plt.imshow(img1)
#             plt.show()
#             print(img1)
        #     img1 = frame[bbx[0,1]-10:bbx[0,3]+10,bbx[0,2]-10:bbx[0,4]+10]
        #     target.append(np.expand_dims(img1, axis=0))
    target = np.stack(target, axis=0)
    #print(target.shape)
    if not os.path.exists(root_path + r'/01_GT/maskOriginal/'):
        os.makedirs(root_path + r'/01_GT/maskOriginal/')
    imwrite(root_path + r'/01_GT/maskOriginal/cellraw_'+str(cell_id)+'.tif',target)

    # if len(target) > 10 and empty_count < 10:
    #     target = np.stack(target, axis=0)
    #     #print(target.shape)
    #     if not os.path.exists(root_path + r'/01_GT/maskOriginal/'):
    #         os.makedirs(root_path + r'/01_GT/maskOriginal/')
    #     imwrite(root_path + r'/01_GT/maskOriginal/cellraw_'+str(cell_id)+'.tif',target)
    # else:
    #     target = np.stack(target, axis=0)
    #     #print(target.shape)
    #     if not os.path.exists(root_path + r'/01_GT/maskOriginal_short/'):
    #         os.makedirs(root_path + r'/01_GT/maskOriginal_short/')
    #     imwrite(root_path + r'/01_GT/maskOriginal_short/cellraw_'+str(cell_id)+'.tif',target)


        #      for t in range(track[io_num][1],track[io_num][2]):
        #             print(t)
        #     #img = skio.imread(files[io_num]).astype(np.uint8)
            

# In[144]:


# image_size = (128, 128)
# black_image = np.zeros(image_size, dtype=np.uint8)
# print(black_image.shape)

# In[ ]:



