#!/usr/bin/env python
# coding: utf-8

# In[137]:


import skimage.io as skio
import numpy as np
import cv2
from tifffile import imread, imwrite

import csv  
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
# root_path = os.path.join(sys.argv[2],sys.argv[3])
root_path = r'/storage/wanyihanLab/sunrui03/sunrui/l3/PAX6_18_29_5h_76h_LLS/PAX6_18_29_5h_76h_LLS_01/spilt/'
spilt_files = [os.path.join(root_path, f) for f in os.listdir(root_path)]
spilt_files.sort()


batch_num = 200 #200帧
num_frames = batch_num + 1 #200帧加1拼接帧
end_frames = num_frames - 1
threshold_d = 75

  
import os  
import csv  
  
def save(lineage, filename):  
    """  
    将lineage列表保存到CSV文件中。  
      
    参数:  
    lineage (list): 包含元组的列表,每个元组包含两个id。  
    filename (str): 包含完整路径的文件名，例如'/path/to/folder/myfile.csv'。  
    """  
    # 检查文件夹是否存在，如果不存在则创建  
    folder_path = os.path.dirname(filename)  
    if not os.path.exists(folder_path):  
        os.makedirs(folder_path)  
      
    # 打开文件准备写入，使用newline=''避免在Windows中出现空行  
    with open(filename, mode='w', newline='') as file:  
        writer = csv.writer(file)  
          
        # 写入表头  
        writer.writerow(['id_0', 'id_1'])  
          
        # 写入lineage列表中的数据  
        for id_0, id_1 in lineage:  
            writer.writerow([id_0, id_1])  
      
    print(f'Lineage data saved to {filename}')

def check_distance(box_0,box_1):
    cy0 = (box_0[0, 2] + box_0[0, 4]) // 2
    cx0 = (box_0[0, 1] + box_0[0, 3]) // 2
    
    cy1 = (box_1[0, 2] + box_1[0, 4]) // 2
    cx1 = (box_1[0, 1] + box_1[0, 3]) // 2

    distances = np.sqrt((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2)  
    return distances
    # 

#0-200 200-400 400-600

for i in range(len(spilt_files)-1):

    #读取跟踪结果txt
    track_0 = np.genfromtxt(spilt_files[i] + r"/01_GT/_RES/res_track.txt",dtype=[int, int, int, int])  # 将文件中数据加载到data数组里
    print(track_0.shape)

    csv_f_0 = pd.read_csv(spilt_files[i] + r'/01_GT/_RES/TRA_'+str(num_frames - 1)+'.csv')

    track_1 = np.genfromtxt(spilt_files[i+1] + r"/01_GT/_RES/res_track.txt",dtype=[int, int, int, int])  # 将文件中数据加载到data数组里
    print(track_1.shape)

    csv_f_1 = pd.read_csv(spilt_files[i+1] + r'/01_GT/_RES/TRA_'+str(0)+'.csv')

    cell_id_0 = []
    cell_id_1 = []
    lineage = []
    for io_num in range(track_0.shape[0]):
        ti = track_0[io_num][1]
        te = track_0[io_num][2]
        if te == end_frames :
            cell_id_0.append(track_0[io_num][0])
        

    for io_num in range(track_1.shape[0]):
        ti = track_1[io_num][1]
        te = track_1[io_num][2]
        if ti == 0:
            cell_id_1.append(track_1[io_num][0])
        

    for j,id_0 in enumerate(cell_id_0):
        
        bb_col_0 = csv_f_0[["id","min_row_bb","min_col_bb","max_row_bb","max_col_bb"]]
        b_col_0 = bb_col_0.loc[bb_col_0['id'] == id_0]
        bbx_0 = np.array(b_col_0)
        
        distances = []
        for k,id_1 in enumerate(cell_id_1):
            bb_col_1 = csv_f_1[["id","min_row_bb","min_col_bb","max_row_bb","max_col_bb"]]
            b_col_1 = bb_col_1.loc[bb_col_1['id'] == id_1]
            bbx_1 = np.array(b_col_1)

            distances.append((check_distance(bbx_0,bbx_1),id_1))

            
        if distances:  
            min_distance, min_id_1 = min(distances, key=lambda x: x[0]) 
            # min_distance, min_id_1 = min(distances)  # 找到最小距离和对应的id_1  
            if min_distance < threshold_d:  
                lineage.append((id_0,min_id_1))
                print(f'{id_0} ---> {min_id_1}')  

        
    if lineage:
        # 指定保存地址和文件名  
        save_path = root_path
        file_name = f'{i+1}_{i+2}_lineage_data.csv'  # 您想要的文件名  
        full_filename = os.path.join(save_path, file_name)  # 构建完整的文件路径  
        save(lineage, full_filename)
  

##concat single_cell



import os  
  
def find(file_paths, id_to_find):  
    """  
    在文件路径列表中查找包含特定ID的文件,并返回其完整路径。  
      
    参数:  
    file_paths (list): 包含文件完整路径的列表。  
    id_to_find (str): 要在文件名中查找的ID。  
      
    返回:  
    str: 包含ID的文件的完整路径,如果未找到则返回None。  
    """
    for file_path in file_paths:  
        if id_to_find in os.path.basename(file_path):  
            return file_path  
    return None

  
# 文件的完整路径  
file_names = [os.path.join(root_path, f) for f in os.listdir(root_path) if f.endswith('.csv')]
file_names.sort()
spilt_files = [os.path.join(root_path, f) for f in os.listdir(root_path)]
spilt_files.sort()

for i in range(len(file_names)):
    # 读取CSV文件  
    df = pd.read_csv(file_names[i])  
    
    # 假设列名是'id_0'和'id_1'  
    id_0_list = df['id_0'].tolist()  # 提取id_0列的所有值到列表中  
    id_1_list = df['id_1'].tolist()  # 提取id_1列的所有值到列表中
    # file_name = f'{i+1}_{i+2}_lineage_data.csv'  
    # root_path
    src = file_names[i].split('_')[0]
    dst = file_names[i].split('_')[1]
    
    #获得所有single cell 的tif地址
    src_path_0_ori = root_path + f'spilt_{src:03d}' + r'/01_GT/maskOriginal/'
    # src_path_0_short = root_path + f'spilt_{src:03d}' + r'/01_GT/maskOriginal_short/'
    src_path_0_ori_files = [os.path.join(src_path_0_ori, f) for f in os.listdir(src_path_0_ori) if f.endswith('.tif') or f.endswith('.tiff')]
    # src_path_0_short_files = [os.path.join(src_path_0_short, f) for f in os.listdir(src_path_0_short) if f.endswith('.tif') or f.endswith('.tiff')]
    # src_path_0_files = src_path_0_ori_files + src_path_0_short_files
    src_path_0_files = src_path_0_ori_files


    dst_path_1_ori = root_path + f'spilt_{dst:03d}' + r'/01_GT/maskOriginal/'
    # dst_path_1_short = root_path + f'spilt_{dst:03d}' + r'/01_GT/maskOriginal_short/'
    dst_path_1_ori_files = [os.path.join(dst_path_1_ori, f) for f in os.listdir(dst_path_1_ori) if f.endswith('.tif') or f.endswith('.tiff')]
    # dst_path_1_short_files = [os.path.join(dst_path_1_short, f) for f in os.listdir(dst_path_1_short) if f.endswith('.tif') or f.endswith('.tiff')]
    # dst_path_1_files = dst_path_1_ori_files + dst_path_1_short_files
    dst_path_1_files = dst_path_1_ori_files

    for id0,id1 in id_0_list,id_1_list:
        print(f'{id0}-->cell_raw{id0}.tif')
        path0 = find(src_path_0_files,id0)
        print(f'{id1}-->cell_raw{id1}.tif')
        path1 = find(dst_path_1_files,id1)

        img0 = skio.imread(path0, plugin="tifffile")
        img0 = img0[:-1]
        img1 = skio.imread(path1, plugin="tifffile")
        img_save = np.concatenate((img0,img1),axis=0)

        
        imwrite(dst_path_1_files + r'cellraw_'+str(id1)+'.tif',img_save)

        # imwrite(root_path + r'/all_frames/maskOriginal/cellraw_'+str(id1)+'.tif',img_save)



    pass


  


    

