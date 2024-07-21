# coding=utf-8
import os
import shutil
import sys

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

# root_path = os.path.join(sys.argv[2],sys.argv[3])

# current_dir = sys.argv[2]


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# root_path = os.path.join(sys.argv[2],sys.argv[3])

current_dir = '/mnt/sda/cell_data/l3/20240414_lightsheet-H9-V6/image/img_all/img_all/01_GT/SAMSEG/'


head, tail = os.path.split(current_dir)
head, penultimate = os.path.split(head)


prj_name = penultimate


imgfiles = [os.path.join(current_dir, f) for f in os.listdir(current_dir) if f.endswith('.tif') or f.endswith('.tiff')]


imgfiles.sort()

len_1 = 1541
len_2 = 955
# print(prj_name)
result_path = '/mnt/sda/cell_data/l3/20240414_lightsheet-H9-V6/img_all_2parts/part2/01_GT/SAMSEG/'
os.makedirs(result_path, exist_ok=True)

i=1542
j = 0
for t in range(1542):

    src_path = os.path.join(current_dir, f'man_seg{i:04d}.tif')
    dst_path = os.path.join(result_path, f'man_seg{j:04d}.tif')
    print(src_path,'--->',dst_path)
    shutil.copy2(src_path,dst_path)
    if j >= len_2:
        break
    i+=1
    j+=1


# for img in imgfiles:

    
    
    

#     #move 
#     src_path = img
    
#     dst_path = os.path.join(result_path, f't{i:04d}.tif')
#     print(img,'--->',dst_path)
#     shutil.copy2(src_path,dst_path)
#     i+=1

      

 

