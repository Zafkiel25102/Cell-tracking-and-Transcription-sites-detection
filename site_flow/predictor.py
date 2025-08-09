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

from spotlearn import SpotlearnNet
from utils import *


def read_csv(csv_path: PathType) -> pd.DataFrame:
    '''Read csv from given path'''
    return pd.read_csv(csv_path, index_col=False)


class SitePredictor:
    '''Predictor for a single cell sequence processing.

    Including site dection, cell registration, site linking and site intensity computation.
    Designed for single cell sequence which has no site, one site or two sites.

    Pay attention:
        - For no site and one site sequence, using `site_track` method to link the detected coordinates.
        - For two sites sequence, using `site_cluster` method to cluster the detected coordinates.
    '''

    EXTENSIONS = ('csv')
    COLUMNS = [
        'x', 'y', 'frame', 'particle'
    ]
    File_Dict = {
            'det_raw': 'imgs_raw_mask.tif',
            'det_reg': 'imgs_raw_mask_reg_rcs.tif',
            'det_coor_reg': 'cell_mask_reg.csv',
            'traj_coor_reg': 'trajectories_data.csv',
            'patch_coor_reg': 'trajectories_data_patch.csv',
            'registration_transform': 'rigid_transforms_series.pkl'
    }

    def __init__(
            self,
            site_dir: Path,
            device: Optional[str] = None,
        ):

        self.root = site_dir
        assert self.root.is_dir() # and self.root.exists()
        self.dir_name = self.root.parts[-1]#.split('-')[-1]   # cellraw_xxx
        self.cell_idx = self.dir_name.split('_')[-1]

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
        assert self.height == 128 and self.width == 128, \
            f"Expected image size 128x128, got {self.height}x{self.width}"
        # del raw_stack

        if not device:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

    @classmethod
    def init(cls, raw_stack_path: str):
        pass

    # region site dection

    @staticmethod
    def spotlearn_detect(
        raw_stack: np.ndarray,
        model_path: Union[str, Path],
        out_threshold: float = 0.9,
        min_region: int = 2,
        max_region: int = 12,
        device: Optional[str] = None,
    ) -> np.ndarray:
        '''Detect sites using Spotlearn model.'''
        assert raw_stack.ndim == 3, f"Expected 3D array, got {raw_stack.ndim}D array"
        assert raw_stack.dtype == np.uint16, f"Expected float16 array, got {raw_stack.dtype} array"
        mask_stack = np.zeros_like(raw_stack)

        net = SpotlearnNet(1, 1).to(device)
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint["model_state_dict"])
        # net = torch.nn.DataParallel(net)
        # net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()

        def _pred_img(net, img, device, out_threshold):
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
                        if region.area <= min_region: # or region.area >= max_region:
                            mask[label_image == region.label] = 0
                    return mask
                
                output = _mask_filter(output)

            return output

        for i in range(raw_stack.shape[0]):
            img = raw_stack[i].copy().astype(np.float32)
            mask_stack[i] = _pred_img(net, img, device, out_threshold)

        mask_stack = mask_stack * 65000
        return mask_stack

    def site_detect(self, model_path):
        """Pipeline method for site detection."""
        mask_stack = self.spotlearn_detect(
            self.raw_stack, model_path, device=self.device
        )

        det_raw = np.stack((self.raw_stack, mask_stack))
        tiff.imwrite(self.det_raw_path, det_raw, imagej=True)

    # region cell registration
    @staticmethod
    def registration_recursive(
        raw_stack: np.array,
        mask_stack: np.array
    ) -> Tuple[np.ndarray, list]:
        """Cell sequence recursive registration.
        
        Args:
            raw_stack (np.ndarray): The raw cell sequence stack.
            mask_stack (np.ndarray): The mask stack of the cell sequence.
        Returns:
            Tuple[np.ndarray, list]: A tuple containing the registered image stack and the last composite transform."""
        num_frame = len(raw_stack)
        assert raw_stack.ndim == 3 and mask_stack.ndim == 3
        assert raw_stack.shape == mask_stack.shape, \
            f"Expected raw_stack and mask_stack to have the same shape, got {raw_stack.shape} and {mask_stack.shape}"
        
        last_composite_transform = []  # final composite transform
        final_cell_registered_stack = []  # save registered img
        final_cell_registered_stack.append(raw_stack[0])
        final_mask_registered_stack = []  # save registered mask
        final_mask_registered_stack.append(mask_stack[0])

        composite_transform = sitk.CompositeTransform(2)

        for i in range(1, num_frame):
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
        final_cell_registered_stack = image_registration_recursive(raw_stack, composite_transform, final_cell_registered_stack)
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

        return det_reg, last_composite_transform

    def site_reg(self):
        """Pipeline method for cell registration."""
        raw_stack = tiff.imread(self.raw_path).squeeze().astype('float32')
        mask_stack = tiff.imread(self.det_raw_path)[1]

        det_reg, last_composite_transform = self.registration_recursive(raw_stack, mask_stack)
        
        tiff.imwrite(self.det_reg_path, det_reg, imagej=True)

        with open(self.reg_transform_path, 'wb') as file:
            pickle.dump(last_composite_transform, file)

    # region get site info

    def get_mask_coor_reg(
            self, 
            rf_classifier_path, 
            nn_classifier_path, 
            min_region = 4, 
            max_region = 12
    ):
        """Obtain the coordinates of the detected sites in the registered mask stack."""
        mask_reg_stack = tiff.imread(self.det_reg_path)[1]
        assert mask_reg_stack.ndim == 3, "Expected 3D array, got {mask_reg_stack.ndim}D array"

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
                coor_pd.loc[i, 'x'] = prop.centroid[1]  # x in fiji viewer
                coor_pd.loc[i, 'y'] = prop.centroid[0]  # y in fiji viewer
                coor_pd.loc[i, 'frame'] = frame

                # if prop.area <= min_region or prop.area >= max_region:
                #     # coor_pd.loc[i, 'region_filer'] = 1
                #     continue
                coor_pd.loc[i, 'area'] = prop.area
                
                # gaussian filter
                x_value, y_value = prop.centroid[0], prop.centroid[1]
                if frame != 0:    
                    (y_value, x_value) = global_transform[frame - 1].TransformPoint((y_value, x_value))
                pic_gaussian, _, _ = get_pic(self.raw_stack[frame], x_value, y_value)
                pic_fp, _, _ = get_pic(self.raw_stack[frame], x_value, y_value, 5)
                is_gaussian_filter, sigma = gaussian_filter(pic_gaussian)
                # trained model filter
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
    
    # region site linking
    @staticmethod
    def single_site_link(
            coor_pd: pd.DataFrame,
            num_frame: int,
            search_range: int = 9,
            memory: int = 5,
            patch_thres: int = 2,
            link_strategy: Literal['filter_patch', 'link_patch'] = 'link_patch',
            save_smallest_id: bool = False,
            save_longest_traj: bool = False
    )-> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Site tracking for cell sequence with single site.

        If tracking result is empty, return None.

        Args:
            coor_pd (pd.DataFrame): DataFrame containing the coordinates of detected sites.
                                    Required columns: 'x [px]', 'y [px]', 'z'.
            num_frame (int): Number of frames in the sequence.
            search_range (int): Search range for linking sites.
            memory (int): Memory parameter for linking sites.
            patch_thres (int): Minimum number of sites in a patch to be considered valid.
        """
        patch_res = None
        # patch generation and filter
        if not coor_pd.empty:
            patch_res = tp.link_df(coor_pd, search_range=search_range, memory=memory)
            
            # very short patch filter (length < patch_thres)
            patch_res = tp.filter_stubs(patch_res, patch_thres)
            patch_res = patch_res.reset_index(drop=True)
            return_patch_res = patch_res.copy()  # before track filter for record data

            # short patch filter based on distance and memory,
            # define short patch creteria here, which shoule be save
            patch_res = short_patch_filter(
                patch_res, patch_len=[2, 3], search_range=6, memory=3
            )
            patch_res = patch_res.reset_index(drop=True)

        if patch_res is None or patch_res.empty:
            return None, None

        # link patches to trajectories
        if link_strategy == 'link_patch':
            has_duplicates = patch_res['frame'].duplicated().any()
            # link patches
            traj_res = link_patches(patch_res, search_range=40, memory=20)
            traj_res = link_patches(traj_res, search_range=20, memory=50)
            traj_res = link_patches(traj_res, search_range=15, memory=100)
            traj_res = link_patches(traj_res, search_range=10, memory=num_frame)

            if save_smallest_id:
                # traj_res = traj_res.loc[traj_res.groupby('frame')['particle'].idxmin()]
                traj_res = save_smallest_id_filter(traj_res)    
            if save_longest_traj:
                traj_res = tp.link_df(traj_res, search_range=150, memory=num_frame)

        elif link_strategy == 'filter_patch':
            traj_res = filter_overlapping_frames(patch_res)
            traj_res['particle'] = 0

        else:
            raise ValueError("Invalid link strategy. Choose 'filter_patch' or 'link_patch'.")

        return return_patch_res, traj_res

    def site_track(
            self,
            search_range: int = 9, 
            memory: int = 5, 
            threshold: int = 2, 
            link_strategy: Literal['filter_patch', 'link_patch'] = 'link_patch',
            save_smallest_id: bool = True,
            save_longest_traj: bool = True
        ):
        coor_pd = read_csv(self.det_coor_reg_path)
        new_columns = {'y [px]': 'y', 'x [px]': 'x', 'z': 'frame'}
        coor_pd.rename(columns=new_columns, inplace=True)

        return_patch_res, traj_res = self.single_site_link(
            coor_pd, self.num_frame, search_range, memory, threshold, 
            link_strategy, save_smallest_id, save_longest_traj
        )
        if return_patch_res is not None and traj_res is not None:
            return_patch_res.to_csv(self.patch_coor_reg_path, index=False)
            traj_res.to_csv(self.traj_coor_reg_path, index=False)


    @staticmethod
    def single_site_link_label(
            coor_pd: pd.DataFrame,
            num_frame: int,
            search_range: int = 9,
            memory: int = 5,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Site tracking for cell sequence with single site and accurate site mask label."""
        patch_res = None
        if not coor_pd.empty:
            patch_res = tp.link_df(coor_pd, search_range=search_range, memory=memory)
        
        if patch_res is None or patch_res.empty:
            return None, None
        
        has_duplicates = patch_res['frame'].duplicated().any()
        # if has_duplicates:
        #     raise ValueError('There are duplicates in the patch results.')
        
        traj_res = tp.link_df(patch_res, search_range=150, memory=num_frame)
        particle_frame_counts = traj_res.groupby('particle')['frame'].count()
        longest_particle = particle_frame_counts.idxmax()
        longest_trajectory = traj_res[traj_res['particle'] == longest_particle]
        longest_trajectory = longest_trajectory.reset_index(drop=True)

        return patch_res, longest_trajectory

    def site_track_label(
            self, 
            search_range=9, 
            memory=5, 
    ):
        coor_pd = read_csv(self.det_coor_reg_path)
        new_columns = {'y [px]': 'y', 'x [px]': 'x', 'z': 'frame'}
        coor_pd.rename(columns=new_columns, inplace=True)

        patch_res, longest_trajectory = self.single_site_link_label(
            coor_pd, self.num_frame, search_range, memory
        )

        if patch_res is not None and longest_trajectory is not None:
            patch_res.to_csv(self.patch_coor_reg_path, index=False)
            longest_trajectory.to_csv(self.traj_coor_reg_path, index=False)


    def site_cluster(self, search_range=9, memory=5, threshold=2):
        """Site cluster based on the tracked coordinates.
        
        Only used for the single cell sequence which has **2 sites**.
        Args are uesd to filter the FPs from the detected coordinates.
        """
        coor_pd = read_csv(self.det_coor_reg_path)
        new_columns = {'y [px]': 'y', 'x [px]': 'x', 'z': 'frame'}
        coor_pd.rename(columns=new_columns, inplace=True)
        
        traj_res = None
        cluster_centers = None
        if not coor_pd.empty:
            # track filter fp
            patch_res = tp.link_df(coor_pd, search_range=search_range, memory=memory)
            patch_res = tp.filter_stubs(patch_res, threshold)
            patch_res = patch_res.reset_index(drop=True)
            coor_pd_filter = patch_res.copy()  # befor track filter for record data

            if not coor_pd_filter.empty:
                # cluster
                traj_res, cluster_centers = link_cluster(patch_res)
                # save cluster results
                coor_pd_filter.to_csv(self.patch_coor_reg_path, index=False)
                traj_res.to_csv(self.traj_coor_reg_path, index=False)

        # plot cluster centers
        reg_stack = tiff.imread(self.det_reg_path)[0]
        reg_stack_max_proj = max_projection(reg_stack)
        if cluster_centers is not None:
            plt.figure()
            plt.imshow(reg_stack_max_proj, cmap='gray')
            coordinates = np.array(cluster_centers)
            plt.scatter(coordinates[:, 0], coordinates[:, 1], color='red')
            plt.axis('off')
            plt.savefig(self.root / 'imgs_raw_reg_rcs_max_projection.png', bbox_inches='tight', pad_inches=0, dpi=300)

    # region site stastic computation

    @staticmethod
    def compute_bg_intensity(
            raw_stack: np.ndarray,
            rigid_transform: list,
            num_frame: int,
            res_dst_path: Path,
            is_plot_intensity: bool = True,
    ):
        """Compute the background intensity of fake tracked sites."""
        traj_res, _ = empty_compute(raw_stack, rigid_transform, num_frame)
        traj_res.to_csv(res_dst_path, index=False)

        if is_plot_intensity:
            plot_intensity(traj_res, res_dst_path.with_suffix('.png'))
    
    @staticmethod
    def compute_tracked_intensity(
        raw_stack: np.ndarray,
        rigid_transform: list,
        num_frame: int,
        track_res: pd.DataFrame, 
        res_dst_dir: Path,
        random_sample_when_zero: bool = True,
        is_plot_intensity: bool = True,
    ):
        """Compute the intensity of the tracked sites."""
        assert not track_res['particle'].empty, "The track_res DataFrame is empty."
        assert res_dst_dir.is_dir(), f"Output directory {res_dst_dir} does not exist."

        traj_num = int(track_res['particle'].max())
        for traj_id in range(traj_num + 1):
            # Create a boolean mask for rows with the specified particle value
            mask = track_res['particle'] == traj_id
            # Filter the DataFrame using the mask
            traj_tp_data = track_res[mask]
            traj_tp_data = traj_tp_data.reset_index(drop=True)

            traj_res, _ = traj_compute(traj_tp_data, raw_stack, rigid_transform, num_frame, random_sample_when_zero)
            res_dst_path = res_dst_dir / ('dataAnalysis_tj_' + str(traj_id) + '_withBg.csv')
            traj_res.to_csv(res_dst_path, index=False)

            if is_plot_intensity:
                plot_intensity(traj_res, res_dst_path.with_suffix('.png'))

    def compute_intensity(self, site2=False, is_plot_intensity=True, random_sample_when_zero=True):
        """Pipelien method to compute the intensity of the tracked sites."""
        raw_stack = tiff.imread(self.det_raw_path)[0]
        rigid_transform = get_global_transform(self.reg_transform_path, self.num_frame)

        fake_track: bool = True
        if self.traj_coor_reg_path.exists():
            track_res = read_csv(self.traj_coor_reg_path)
            if not track_res['particle'].empty:
                fake_track = False
        
        if fake_track:
            if not site2:
                self.compute_bg_intensity(
                    raw_stack, rigid_transform, self.num_frame, 
                    self.root / 'dataAnalysis_tj_empty_withBg.csv', is_plot_intensity=is_plot_intensity
                )
            else:
                self.compute_bg_intensity(
                    raw_stack, rigid_transform, self.num_frame, 
                    self.root / 'dataAnalysis_tj_empty_0_withBg.csv', is_plot_intensity=is_plot_intensity
                )
                self.compute_bg_intensity(
                    raw_stack, rigid_transform, self.num_frame, 
                    self.root / 'dataAnalysis_tj_empty_1_withBg.csv', is_plot_intensity=is_plot_intensity
                )
        else:
            self.compute_tracked_intensity(
                raw_stack, rigid_transform, self.num_frame,
                track_res, self.root,
                random_sample_when_zero=random_sample_when_zero,
                is_plot_intensity=is_plot_intensity
            )