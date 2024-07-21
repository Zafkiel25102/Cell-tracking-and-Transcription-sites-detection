#!/usr/bin/env python
# coding: utf-8

# In[2]:

import sys
import os
import shutil

# 获取当前目录
current_dir = sys.argv[2]
# os.getcwd()

# 获取以.tif为结尾的文件
tif_files = [file for file in os.listdir(current_dir) if file.endswith(".tif")]

for file in tif_files:
    # 获取文件名（不包含扩展名）
    file_name = os.path.splitext(file)[0]

    # 创建与文件名称相同的文件夹
    new_dir = os.path.join(current_dir, file_name)
    os.makedirs(new_dir, exist_ok=True)

    # 创建01和01_GT文件夹
    sub_dir_01 = os.path.join(new_dir, "01")
    os.makedirs(sub_dir_01, exist_ok=True)

    sub_dir_01_gt = os.path.join(new_dir, "01_GT")
    os.makedirs(sub_dir_01_gt, exist_ok=True)

    # 移动文件到01文件夹下
    src_file = os.path.join(current_dir, file)
    dst_file = os.path.join(sub_dir_01, file)
    shutil.move(src_file, dst_file)


# In[ ]:



