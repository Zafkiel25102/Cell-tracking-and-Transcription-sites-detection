import os
import shutil
import sys

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

# root_path = os.path.join(sys.argv[2],sys.argv[3])

# current_dir = sys.argv[2]


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# root_path = os.path.join(sys.argv[2],sys.argv[3])

current_dir = '/mnt/sda/cell_data/l3/20240711/SOX2_wt_50h_20240717_002716/'

# 使用os.path.split()分割路径
head, tail = os.path.split(current_dir)
head, penultimate = os.path.split(head)

# 获取最后一个斜杠之后的所有字符
prj_name = penultimate

# print(prj_name)
result_path = os.path.join(current_dir,prj_name)


# 获取当前目录下的所有项目（文件和文件夹）
all_items = os.listdir(current_dir)

# 筛选出所有文件夹
folders = [item for item in all_items if os.path.isdir(os.path.join(current_dir, item))]

folders.sort()

os.makedirs(result_path, exist_ok=True)

# 打印所有文件夹
for folder in folders:
    # print(folder)
    if folder == prj_name:
        continue
    
    new_dir = os.path.join(result_path, folder)
    os.makedirs(new_dir, exist_ok=True)

    #move 
    old_dir = os.path.join(current_dir, folder)

    src_file_0 = os.path.join(old_dir,'01_GT/maskOriginal_short/')
    src_file_1 = os.path.join(old_dir,'01_GT/maskOriginal/')
    src_file_2 = os.path.join(old_dir,'01_GT/_RES/')
    print(src_file_1)
    dst_file = new_dir
    shutil.move(src_file_0, dst_file)
    shutil.move(src_file_1, dst_file)
    shutil.move(src_file_2, dst_file)
    # 获取当前目录下的所有项目（文件和文件夹）

    # all_items_del = os.listdir(old_dir)
    # folders_del = [item for item in all_items_del if os.path.isdir(os.path.join(old_dir, item))]
    # folders_del.sort()
    # for folder_del in folders_del:
    #     folder_path = os.path.join(old_dir, folder_del)
    #     if os.path.exists(folder_path):
    #         # shutil.rmtree(folder_path)
    #         print(f"Deleted: {folder_path}")
    #     else:
    #         print(f"{folder_path} does not exist.")
 
    pass

