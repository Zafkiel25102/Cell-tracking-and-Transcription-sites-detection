import os
import shutil
from pathlib import Path
import logging
import argparse

from .cell_obj import CellFolder

def main():

    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Prediction script')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to the input folder')
    parser.add_argument('--model_weight_path', type=str, required=True,
                       help='Path to the directory containing model weights')

    args = parser.parse_args()

    if not os.path.isdir(args.data_path):  
        print(f"Error: {args.data_path} is not a valid directory!")
        return

    if not os.path.isdir(args.model_weight_path):  
        print(f"Error: {args.model_weight_path} is not a valid directory!")
        return

    model_weight_path = os.path.abspath(args.model_weight_path)

    for field in sorted(os.listdir(args.data_path)):
        field_path = os.path.join(args.data_path, field)
            
        for site_num in sorted(os.listdir(field_path)):
            field_site_path = os.path.join(field_path, site_num)

            if os.path.isfile(field_site_path):
                continue

            if not site_num in ['0', '1']:
                continue

            cell_path_list = [os.path.join(field_site_path, f) for f in os.listdir(field_site_path) if f.endswith('.tif')]

            for cell_path in sorted(cell_path_list):
                print(f'Current cell dir: {cell_path}')

                cell_path = Path(cell_path)
                stem = cell_path.stem
                parent = cell_path.parent

                # make a new directory for the cell
                cell_dir = parent / stem
                cell_dir.mkdir(exist_ok=True)
                shutil.copy2(cell_path, cell_dir, follow_symlinks=True)

                cell_folder = CellFolder(cell_dir)

                ## begin process pipeline 
                if not (cell_folder.root / 'imgs_raw_mask_reg_rcs.tif').exists():
                    spotlearn_model_path = os.path.join(model_weight_path, 'spotlearn', 'epoch40.pt')

                    cell_folder.spotlearn_pred(spotlearn_model_path)
                    cell_folder.registration_recursive()

                if not ((cell_folder.root / 'raw_stack_with_label.tif').exists() \
                        or (cell_folder.root / 'dataAnalysis_tj_empty_withBg.csv').exists()):
                    rf_classifier_path = os.path.join(model_weight_path, 'rf_classifier', 'random_forest_model.pkl')
                    nn_classifier_path = os.path.join(model_weight_path, 'nn_classifier', 'tut1-model.pt')

                    if site_num == '1':
                        cell_folder.get_mask_coor_reg(rf_classifier_path, nn_classifier_path)
                        cell_folder.site_track(chose_longest=True, frame_filter=True)

                    cell_folder.compute_intensity(double=False)
                    cell_folder.get_raw_stack_with_label()

                    del cell_folder

if __name__ == '__main__':
    main()