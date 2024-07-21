# from aicsimageio import AICSImage
# import numpy as np
# import os
# import tifffile


# image_folder_01 = '/mnt/sda/cell_data/l3/20240709/raw_data/03sox2_nim_3h_lls_60h/sox2_nim_3h_lls_60h/01/'
# if not os.path.exists(image_folder_01):
#     os.makedirs(image_folder_01)


# img = AICSImage("/mnt/sda/cell_data/l3/20240709/sox2_nim_3h_lls_60h/03/sox2-nim-3h-LatticeLightsheet-3.czi")

# num_timepoints = img.dims['T']  

# # 检查维度信息  
# print("Dimensions of the image:")  
# for dim, size in img.dims.items():  
#     print(f"{dim}: {size}")  
# print(type(img.dims))

# # 初始化一个空列表来保存所有时间点的帧  
# frames = []
# t0 = 0
# print(num_timepoints[0])

# for i in range(num_timepoints[0]):
    


#     image_path = os.path.join(image_folder_01, f't{i+t0:04d}.tif')




#     # img = AICSImage("/storage/wanyihanLab/sunrui03/sunrui/l3/LLS/PAX6_18_29.5h_76h_LLS.czi")
#     img_virtual = img.get_image_dask_data("ZYX", T=i,C=0) # select channel
#     img_real = img_virtual.compute()  # read in-memory
#     # img_data = img.get_image_data("ZYX", T=i, C=0)  # 选择通道和时间点  
#     # img_real = img_data  
#     # img_MIP = np.max(img_real,0) # max intensity projection in z dimension


#     img_MIP = img_real

#     # frames.append(img_MIP)

#     tifffile.imwrite(image_path,img_MIP)
#     print(f"Saved MIP for timepoint {i} to {image_path}")

# # tiff_path = os.path.join(image_folder_01, 'all_frames_MIP.tif')  
# # tifffile.imsave(tiff_path, frames)  


#--------------------------------------------------->>>>>>Z轴<<<<<<<------------------------------------------------------------#

from aicsimageio import AICSImage
import numpy as np
import os
import tifffile


image_folder_01 = '/mnt/sda/cell_data/l3/20240709/raw_data/03sox2_nim_3h_lls_60h/sox2_nim_3h_lls_60h/01/'
if not os.path.exists(image_folder_01):
    os.makedirs(image_folder_01)


img = AICSImage("/mnt/sda/cell_data/l3/20240709/sox2_nim_3h_lls_60h/03/sox2-nim-3h-LatticeLightsheet-3.czi")

num_timepoints = img.dims['Z']  
print("Original dimensions:", img.dims)

# 检查维度信息  
print("Dimensions of the image:")  
for dim, size in img.dims.items():  
    print(f"{dim}: {size}")  
print(type(img.dims))

# 初始化一个空列表来保存所有时间点的帧  
frames = []
t0 = 0
print(num_timepoints[0])

for i in range(num_timepoints[0]):
    


    image_path = os.path.join(image_folder_01, f't{i+t0:04d}.tif')




    # img = AICSImage("/storage/wanyihanLab/sunrui03/sunrui/l3/LLS/PAX6_18_29.5h_76h_LLS.czi")
    img_virtual = img.get_image_dask_data("ZYX", Z=i) # select channel
    img_real = img_virtual.compute()  # read in-memory
    # img_data = img.get_image_data("ZYX", T=i, C=0)  # 选择通道和时间点  
    # img_real = img_data  
    # img_MIP = np.max(img_real,0) # max intensity projection in z dimension


    img_MIP = img_real

    # frames.append(img_MIP)

    tifffile.imwrite(image_path,img_MIP)
    print(f"Saved MIP for timepoint {i} to {image_path}")

# tiff_path = os.path.join(image_folder_01, 'all_frames_MIP.tif')  
# tifffile.imsave(tiff_path, frames)  