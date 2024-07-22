import numpy as np
import pandas as pd
import os
import tifffile as tiff
from skimage import measure

def get_mask_reg_coor(field, cell_folder):
    cell_idx = os.path.basename(cell_folder)
    cell_index_num = cell_idx.split('_')[1]

    csv_data = pd.DataFrame()
    csv_data['x'] = ''
    csv_data['y'] = ''
    csv_data['frame'] = ''

    # mask_path = cell_mask_reg.tif
    mask_path = os.path.join(cell_folder, 'imgs_raw_mask_reg_rcs.tif')
    mask = tiff.imread(mask_path)
    mask = mask[1, :, :, :]

    i=0
    for frame in range(mask.shape[0]):
        img = mask[frame].copy()

        # get centroid cordiante
        img_mask = measure.label(img)
        properties = measure.regionprops(img_mask)
        for prop in properties:
            csv_data.loc[i, 'x'] = prop.centroid[1]
            csv_data.loc[i, 'y'] = prop.centroid[0]
            csv_data.loc[i, 'frame'] = frame
            i += 1

    # save csv_data
    csv_target = os.path.join(cell_folder, 'cell_mask_reg.csv')
    csv_data.to_csv(csv_target, index=False)

    print(f'field {field:^4s}: cell {cell_index_num:^7s} coordinate extraction processing is done.')

if __name__=="__main__":
    data_path = r'D:\BaiduNetdiskDownload\20230807-sox2-31.5h trace'
    for field in os.listdir(data_path):
        field_path = os.path.join(data_path, field)
        for cell_idx in os.listdir(field_path):
            tiff_path = os.path.join(field_path, cell_idx)
            tiff_file = os.listdir(tiff_path)
            for file in tiff_file:
                get_mask_reg_coor(field_path, cell_idx)

         

