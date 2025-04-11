import os
import re
from pathlib import Path
from typing import Tuple, Union, Literal, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
import scipy.optimize
from skimage import measure
import trackpy as tp

from pipeline.spotlearn import SpotlearnNet
from pipeline.utils import *


__class = ['Tracker', 'Celler', 'Trajer']

def read_csv(csv_path: PathType) -> pd.DataFrame:
    '''Read csv from given path'''
    return pd.read_csv(csv_path, index_col=False)


class CellFolder:
    '''An object for a single cell stack'''

    EXTENSIONS = ('csv')
    COLUMNS = [
        'x', 'y', 'frame', 'particle'
    ]
    File_Dict = {
            'det_raw': 'imgs_raw_mask.tif',
            'det_reg': 'imgs_raw_mask_reg_rcs.tif',
            'det_coor_reg': 'cell_mask_reg.csv',
            'traj_coor_reg': 'trajectories_data.csv',
            'patch_coor_reg': 'trajectories_data_raw.csv',
            'registration_transform': 'rigid_transforms_series.pkl'
    }

    def __init__(
            self,
            dir_path: Path,
            gpu = '0',
        ):

        self.root = dir_path
        assert self.root.is_dir() # and self.root.exists()
        self.dir_name = self.root.parts[-1]#.split('-')[-1]   # cellraw_xxx
        self.cell_idx = self.dir_name.split('_')[-1]

        # file
        self.raw_path = self.root / (self.dir_name + '.tif')  # raw cell stack 
        self.label_path = self.root / (self.dir_name + '_mask.tif')  # mask label stack
        self.det_raw_path = self.root / self.File_Dict['det_raw']  # raw mask stack
        self.det_reg_path = self.root / self.File_Dict['det_reg']  # reg mask stack
        self.det_coor_reg_path = self.root / self.File_Dict['det_coor_reg']  # reg mask site coordinates
        self.patch_coor_reg_path = self.root / self.File_Dict['patch_coor_reg']  # site coordinate of patch
        self.traj_coor_reg_path = self.root / self.File_Dict['traj_coor_reg']  # site coordinate of traj
        self.reg_transform_path = self.root / self.File_Dict['registration_transform']  # global reg transform

        self.raw_stack = tiff.imread(self.raw_path).squeeze()
        assert self.raw_stack.ndim == 3
        self.shape = self.raw_stack.shape
        self.num_frame, self.height, self.width = self.shape
        # del raw_stack

    # ===============
    # pipeline Method
    # ===============

    def spotlearn_pred(self, model_path):
        def _pred_img(net, img, device, out_threshold=0.5):
            net.eval()
            img = spotlearn_norm(img)  # norm
            img = torch.from_numpy(img)
            img = img.unsqueeze(0)
            img = img.unsqueeze(0)
            img = img.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                output = net(img).cpu().numpy()
                output = np.squeeze(output)
                output[output <= out_threshold] = 0
                output[output > out_threshold] = 1

                def _mask_filter(mask):
                    label_image = measure.label(mask, connectivity=2)
                    regions = measure.regionprops(label_image)
                    for region in regions:
                        if region.area <= 4: # or region.area >= 12:
                            mask[label_image == region.label] = 0
                    return mask
                
                output = _mask_filter(output)

            return output

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'=======Current device: {device}')
        net = SpotlearnNet(1, 1).to(device)
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint["model_state_dict"])
        # net = torch.nn.DataParallel(net)
        # net.load_state_dict(checkpoint["model_state_dict"])

        mask_stack = np.zeros_like(self.raw_stack)
        raw_stack = tiff.imread(self.raw_path).squeeze().astype('float32')
        for i in range(self.num_frame):
            mask = _pred_img(net, raw_stack[i], device, out_threshold=0.9)
            mask_stack[i] = mask

        mask_stack = mask_stack * 65000
        det_raw = np.stack((self.raw_stack, mask_stack))
        tiff.imwrite(self.det_raw_path, det_raw, imagej=True)

    def registration_recursive(self):
        
        raw_stack = tiff.imread(self.raw_path).squeeze().astype('float32')
        mask_stack = tiff.imread(self.det_raw_path)[1]
        assert raw_stack.ndim == 3 and mask_stack.ndim == 3
        
        last_composite_transform = []  # final composite transform
        final_cell_registered_stack = []  # save registered img
        final_cell_registered_stack.append(self.raw_stack[0])
        final_mask_registered_stack = []  # save registered mask
        final_mask_registered_stack.append(mask_stack[0])

        composite_transform = sitk.CompositeTransform(2)

        for i in range(1, self.num_frame):
            fixed_image = sitk.GetImageFromArray(raw_stack[i - 1])  # moving_registered_stack only resample once
            moving_frame = sitk.GetImageFromArray(raw_stack[i])  # reg_stack
            moving_frame.SetSpacing(fixed_image.GetSpacing())
            moving_frame.SetOrigin(fixed_image.GetOrigin())
            moving_frame.SetDirection(fixed_image.GetDirection())

            # 执行配准并得到配准后的图像
            moving_registered, final_transform = rigid_registration_centroid(fixed_image, moving_frame)

            composite_transform.AddTransform(final_transform)

        last_composite_transform.append(composite_transform)  # for revovery

        # image recursive registration
        final_cell_registered_stack = image_registration_recursive(self.raw_stack, composite_transform, final_cell_registered_stack)
        reg_stack = np.array(final_cell_registered_stack)
        # tiff.imwrite(os.path.join(cell_folder, 'cell_reg.tif'), reg_img)

        # mask recursive registration
        final_mask_registered_stack = image_registration_recursive(mask_stack, composite_transform, final_mask_registered_stack)
        final_mask_registered_stack = np.array(final_mask_registered_stack)
        # final_mask_registered_stack *= 65000
        threshold = 1
        final_mask_registered_stack[final_mask_registered_stack >= threshold] = 65000
        final_mask_registered_stack[final_mask_registered_stack < threshold] = 0
        final_mask_registered_stack = final_mask_registered_stack.astype(np.uint16)
        # tiff.imwrite(os.path.join(cell_folder, 'cell_mask_reg.tif'), final_mask_registered_stack)

        det_reg = np.stack((reg_stack, final_mask_registered_stack))
        tiff.imwrite(self.det_reg_path, det_reg, imagej=True)

        with open(self.reg_transform_path, 'wb') as file:
            pickle.dump(last_composite_transform, file)

    def get_mask_coor_reg(self, rf_classifier_path, nn_classifier_path, min_region = 4, max_region = 12):

        mask_reg_stack = tiff.imread(self.det_reg_path)[1]
        assert mask_reg_stack.ndim == 3

        coor_pd = pd.DataFrame(columns=['x', 'y', 'frame', 
                                    # 'region', 'region_filer', 
                                    'gaussian_sigma', 'gaussian_filer',
                                    'rf_filer', 'nn_filer'])
        
        rf_model = load_rf_classifier(rf_classifier_path)
        nn_model = load_nn_classifier('cnn', nn_classifier_path)
        global_transform = get_global_transform(self.reg_transform_path, self.num_frame)

        i = 0
        for frame, mask in enumerate(mask_reg_stack):
            mask = measure.label(mask)
            properties = measure.regionprops(mask)
            for prop in properties:
                coor_pd.loc[i, 'x'] = prop.centroid[1]
                coor_pd.loc[i, 'y'] = prop.centroid[0]
                coor_pd.loc[i, 'frame'] = frame

                # region filter
                # coor_pd.loc[i, 'region'] = prop.area
                # if prop.area <= min_region or prop.area >= max_region:
                #     coor_pd.loc[i, 'region_filer'] = 1

                x_value, y_value = prop.centroid[0], prop.centroid[1]
                if frame != 0:    
                    (y_value, x_value) = global_transform[frame - 1].TransformPoint((y_value, x_value))
                pic_gaussian, _, _ = get_pic(self.raw_stack[frame], x_value, y_value)
                pic_fp, _, _ = get_pic(self.raw_stack[frame], x_value, y_value, 5)
            
                is_gaussian_filter, sigma = gaussian_filter(pic_gaussian)
                rf_res, nn_res = fp_filter(pic_fp, rf_model, nn_model)

                if is_gaussian_filter:
                    coor_pd.loc[i, 'gaussian_filer'] = 1
                else:
                    coor_pd.loc[i, 'gaussian_sigma'] = sigma
                if rf_res == 1:
                    coor_pd.loc[i, 'rf_filer'] = 1

                thred = 0.69
                if nn_res.item() > thred:
                    coor_pd.loc[i, 'nn_filer'] = 1

                i += 1
        
        coor_pd.to_csv(self.det_coor_reg_path, index=False)

    def site_track(self, search_range=9, memory=5, threshold=2, chose_longest=False, frame_filter=False):
        coor_pd = read_csv(self.det_coor_reg_path)
        new_columns = {'y [px]': 'y', 'x [px]': 'x', 'z': 'frame'}
        coor_pd.rename(columns=new_columns, inplace=True)

        if coor_pd.empty:
            coor_pd['particle'] = None
            patch_res = None
            traj_res = None
        else:
            # track filter fp
            patch_res = tp.link_df(coor_pd, search_range=search_range, memory=memory)
            # normal fitler
            patch_res = tp.filter_stubs(patch_res, threshold)
            patch_res = patch_res.reset_index(drop=True)
            return_patch_res = patch_res.copy()  # befor track filter for record data
            # track filter
            patch_res = track_filter(patch_res, search_range=6, memory=3)
            patch_res = patch_res.reset_index(drop=True)

            if patch_res.empty:
                coor_pd['particle'] = None
                patch_res = None
                traj_res = None
            else:
                has_duplicates = patch_res['frame'].duplicated().any()

                traj_res = link_patches(patch_res.copy(), search_range=40, memory=20)
                traj_res = link_patches(traj_res, search_range=20, memory=50)
                traj_res = link_patches(traj_res, search_range=15, memory=100)
                traj_res = link_patches(traj_res, search_range=10, memory=self.num_frame)
                
                if frame_filter:
                    # traj_res = traj_res.loc[traj_res.groupby('frame')['particle'].idxmin()]
                    traj_res = frame_filter_(traj_res)

                if chose_longest:
                    traj_res = tp.link_df(traj_res, search_range=150, memory=self.num_frame)
        
                return_patch_res.to_csv(self.patch_coor_reg_path, index=False)
                traj_res.to_csv(self.traj_coor_reg_path, index=False)

    def site_track_label(self, search_range=9, memory=5, threshold=2, chose_longest=False, frame_filter=False):
        coor_pd = read_csv(self.det_coor_reg_path)
        new_columns = {'y [px]': 'y', 'x [px]': 'x', 'z': 'frame'}
        coor_pd.rename(columns=new_columns, inplace=True)

        if coor_pd.empty:
            coor_pd['particle'] = None
            patch_res = None
            traj_res = None
        else:
            # track filter fp
            patch_res = tp.link_df(coor_pd, search_range=search_range, memory=memory)
            return_patch_res = patch_res.copy()

            if patch_res.empty:
                coor_pd['particle'] = None
                patch_res = None
                traj_res = None
            else:
                has_duplicates = patch_res['frame'].duplicated().any()

                # if has_duplicates:
                #     raise ValueError('has duplicates')

                traj_res = tp.link_df(patch_res, search_range=150, memory=self.num_frame)

                particle_frame_counts = traj_res.groupby('particle')['frame'].count()
                longest_particle = particle_frame_counts.idxmax()
                longest_trajectory = traj_res[traj_res['particle'] == longest_particle]
                longest_trajectory = longest_trajectory.reset_index(drop=True)
        
                return_patch_res.to_csv(self.patch_coor_reg_path, index=False)
                longest_trajectory.to_csv(self.traj_coor_reg_path, index=False)

    def site_cluster(self, search_range=9, memory=5, threshold=2):
        coor_pd = read_csv(self.det_coor_reg_path)
        new_columns = {'y [px]': 'y', 'x [px]': 'x', 'z': 'frame'}
        coor_pd.rename(columns=new_columns, inplace=True)
        
        if coor_pd.empty:
            coor_pd['particle'] = None
            patch_res = None
            traj_res = None
        else:
            # track filter fp
            patch_res = tp.link_df(coor_pd, search_range=search_range, memory=memory)
            # normal fitler
            patch_res = tp.filter_stubs(patch_res, threshold)
            patch_res = patch_res.reset_index(drop=True)
            return_patch_res = patch_res.copy()  # befor track filter for record data

            if patch_res.empty:
                coor_pd['particle'] = None
                patch_res = None
                traj_res = None
            else:
                traj_res, cluster_centers = link_cluster(patch_res)

            return_patch_res.to_csv(self.patch_coor_reg_path, index=False)
            traj_res.to_csv(self.traj_coor_reg_path, index=False)

        reg_stack = tiff.imread(self.det_reg_path)[0]
        reg_stack_max_proj = max_projection(reg_stack)
        if cluster_centers is not None:
            plt.figure()
            plt.imshow(reg_stack_max_proj, cmap='gray')
            coordinates = np.array(cluster_centers)
            plt.scatter(coordinates[:, 0], coordinates[:, 1], color='red')
            plt.axis('off')
            plt.savefig(self.root / 'imgs_raw_reg_rcs_max_projection.png', bbox_inches='tight', pad_inches=0, dpi=300)

    def compute_intensity(self, double=False, plot_intensity=True):

        def _plot(df, dst_path, col='photon_number'):
            fig, ax = plt.subplots()
            df[col].plot(kind='line', xlabel='Frame', ylabel=col, ax=ax)
            plt.savefig(dst_path, format='png')
            ax.cla()

        def empty_compute_intensity(raw_stack, rigid_transform, num_frame, csv_suffix):
            traj_res, _ = empty_compute(raw_stack, rigid_transform, num_frame)
            dst_path = self.root / f'{csv_suffix}'
            traj_res.to_csv(dst_path, index=False)

            if plot_intensity:
                dst_path = dst_path.with_suffix('.png')
                _plot(traj_res, dst_path)

        raw_stack = tiff.imread(self.det_raw_path)[0]
        rigid_transform = get_global_transform(self.reg_transform_path, self.num_frame)

        if not self.traj_coor_reg_path.exists():
            # empty computation
            if not double:
                empty_compute_intensity(raw_stack, rigid_transform, self.num_frame, 'dataAnalysis_tj_empty_withBg.csv')
            else:
                empty_compute_intensity(raw_stack, rigid_transform, self.num_frame, 'dataAnalysis_tj_empty_0_withBg.csv')
                empty_compute_intensity(raw_stack, rigid_transform, self.num_frame, 'dataAnalysis_tj_empty_1_withBg.csv')

        else:
            track_res = read_csv(self.traj_coor_reg_path)

            if track_res['particle'].empty:
                # empty computation
                if not double:
                    empty_compute_intensity(raw_stack, rigid_transform, self.num_frame, 'dataAnalysis_tj_empty_withBg.csv')
                else:
                    empty_compute_intensity(raw_stack, rigid_transform, self.num_frame, 'dataAnalysis_tj_empty_0_withBg.csv')
                    empty_compute_intensity(raw_stack, rigid_transform, self.num_frame, 'dataAnalysis_tj_empty_1_withBg.csv')

            else:
                traj_num = int(track_res['particle'].max())
                for traj_id in range(traj_num + 1):
                    # Create a boolean mask for rows with the specified particle value
                    mask = track_res['particle'] == traj_id
                    # Filter the DataFrame using the mask
                    traj_tp_data = track_res[mask]
                    traj_tp_data = traj_tp_data.reset_index(drop=True)

                    traj_res, _ = traj_compute(traj_tp_data, raw_stack, rigid_transform, self.num_frame)
                    dst_path = self.root / ('dataAnalysis_tj_' + str(traj_id) + '_withBg.csv')
                    traj_res.to_csv(dst_path, index=False)

                    if plot_intensity:
                        dst_path = dst_path.with_suffix('.png')
                        _plot(traj_res, dst_path)
                
                if double:
                    empty_compute_intensity(raw_stack, rigid_transform, self.num_frame, 'dataAnalysis_tj_empty_withBg.csv')

    # =============
    # Helper Method
    # =============

    def get_mask_from_track(
            self,
            opt: Literal['traj', 'patch'] = 'patch',
        ) -> np.ndarray:
        
        if not self.patch_coor_reg_path.exists():
            return 

        patch_coor_reg = pd.read_csv(self.patch_coor_reg_path, index_col=False, na_values='NAN')
        traj_coor_reg = pd.read_csv(self.traj_coor_reg_path, index_col=False, na_values='NAN')

        mask = np.zeros(self.shape)
        site_data = patch_coor_reg if opt == 'patch' else traj_coor_reg
        grouped_data = site_data.groupby('frame')
        for frame, group in grouped_data:
                coords = group[['y', 'x']].to_numpy().astype(int)
                mask[frame] += coordinate_to_mask(coords, self.height, n=1)

        return mask
    
    def get_raw_stack_with_label(self, area=4):

        if not self.patch_coor_reg_path.exists():
            return
        
        raw_stack_with_label = self.raw_stack.copy()
        rigid_transform = get_global_transform(self.reg_transform_path, self.num_frame)
        patch_coor_reg = pd.read_csv(self.patch_coor_reg_path, index_col=False, na_values='NAN')
        traj_coor_reg = pd.read_csv(self.traj_coor_reg_path, index_col=False, na_values='NAN')

        traj_coords_set = set(tuple(x) for x in traj_coor_reg[['y', 'x']].to_numpy())
        grouped_data = patch_coor_reg.groupby('frame')
        for frame, group in grouped_data:
            coords = group[['y', 'x']].to_numpy().astype(float)

            for r, c in coords:

                label_pixel = 0 if (r, c) in traj_coords_set else 300
                if frame != 0:
                    transform = rigid_transform[frame - 1]
                    c, r = coord_reg_to_raw(c, r, transform)
                c = round(c)
                r = round(r)

                top, bottom = max(0, r - area), min(raw_stack_with_label.shape[1] - 1, r + area)
                left, right = max(0, c - area), min(raw_stack_with_label.shape[2] - 1, c + area)
                # print(frame)
                raw_stack_with_label[frame, top, left:right + 1] = label_pixel
                raw_stack_with_label[frame, bottom, left:right + 1] = label_pixel
                raw_stack_with_label[frame, top:bottom + 1, left] = label_pixel
                raw_stack_with_label[frame, top:bottom + 1, right] = label_pixel

        dst_path = self.root / 'raw_stack_with_label.tif'
        tiff.imwrite(dst_path, raw_stack_with_label)
    
    def evaluate(self, opt: Literal['det', 'track', 'traj'] = 'track') -> Tuple[int, int ,int]:

        if not(self.label_path.exists()):
            raise ValueError(f'{self.label} is not existing!')
        label_stack = tiff.imread(self.label).squeeze()
        assert label_stack.ndim == 3

        if opt == 'track':
            if os.path.exists(self.traj):
                det_stack = tiff.imread(self.traj)[1]
            else:
                det_stack = np.zeros(self.shape)
        elif opt == 'det':
            det_stack = self.det_data_raw_mask
        else:
            if os.path.exists(self.complete_traj):
                det_stack = tiff.imread(self.complete_traj)[1]
            else:
                det_stack = np.zeros(self.shape)

        label_stack = np.clip(label_stack, 0, 1)
        det_stack = np.clip(det_stack, 0, 1)

        cutoff = 3
        tp_list = []; fn_list = []; fp_list = []
        for i in range(self.num_frame):
            true = mask_to_coordinate(label_stack[i])
            pred = mask_to_coordinate(det_stack[i])

            if len(true) == 0:
                tp = 0; fn = 0; fp = len(pred)
            elif len(pred) == 0:
                tp = 0; fn = len(true); fp = 0
            else:
                matrix = scipy.spatial.distance.cdist(pred, true, metric="euclidean")
                pred_true_r, _ = linear_sum_assignment(matrix, cutoff)
                true_pred_r, true_pred_c = linear_sum_assignment(matrix.T, cutoff)

                # Calculation of tp/fn/fp based on number of assignments
                tp = len(true_pred_r)
                fn = len(true) - len(true_pred_r)
                fp = len(pred) - len(pred_true_r)

            tp_list.append(tp); fn_list.append(fn); fp_list.append(fp)

        tp = sum(tp_list); fn = sum(fn_list); fp = sum(fp_list)

        return tp, fn, fp
    
    @classmethod
    def init(cls, raw_stack_path: str):
        pass
    