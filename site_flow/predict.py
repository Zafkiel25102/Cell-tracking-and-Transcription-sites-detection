import argparse
import os
from ..site_flow import *

def predict(data_path, model_weight):
    for field in os.listdir(data_path):
        field_path = os.path.join(data_path, field)  
        for site_num in os.listdir(field_path):
            field_site_path = os.path.join(field_path, site_num)  # ['0', '1', 'multi']
            for cell_idx_tif in os.listdir(field_site_path):
                if cell_idx_tif.endswith('.tif'):
                    cell_idx, _ = os.path.splitext(cell_idx_tif) 
                    cell_folder = os.path.join(field_site_path, cell_idx)

                    if not os.path.exists(cell_folder) or 'imgs_raw_mask.tif' not in os.listdir(cell_folder):
                        print(cell_idx_tif)
                        pred_torch(field, field_site_path, cell_idx_tif, model_weight, gpu='0')

                    if 'imgs_raw_mask_reg_rcs.tif' not in os.listdir(cell_folder):
                        reg_recursive(field, cell_folder)

    for field in os.listdir(data_path):
        field_path = os.path.join(data_path, field)  
        for site_num in os.listdir(field_path):
            field_site_path = os.path.join(field_path, site_num)  # ['0', '1', 'multi']
            for cell_idx in os.listdir(field_site_path):
                cell_folder = os.path.join(field_site_path, cell_idx)
                if not(cell_idx.endswith('.tif')):
                    get_mask_reg_coor(field, cell_folder)
                    spot_track_single(field, cell_folder,)

    
    for field in os.listdir(data_path):
        field_path = os.path.join(data_path, field)  
        for site_num in os.listdir(field_path):
            field_site_path = os.path.join(field_path, site_num)  # ['0', '1', 'multi']
            for cell_idx in os.listdir(field_site_path):
                cell_folder = os.path.join(field_site_path, cell_idx)
                if not(cell_idx.endswith('.tif')):
                    if 'dataAnalysis_tj_empty_withBg.csv' not in os.listdir(cell_folder) and 'dataAnalysis_tj_0_withBg.csv' not in os.listdir(cell_folder):
                        GetIntensity(field, cell_folder, is_filter=False)

def main():
    parser = argparse.ArgumentParser(description='Prediction script')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input folder')
    parser.add_argument('--model_weight', type=str, required=True, help='Path to the model weights')

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: {args.data_path} does not exist!")
        return

    if not os.path.exists(args.model_weight):
        print(f"Error: {args.model_weight} does not exist!")
        return
    
    predict(args.data_path, args.model_weight)

    print(f"Prediction is done!")

if __name__ == '__main__':
    main()
