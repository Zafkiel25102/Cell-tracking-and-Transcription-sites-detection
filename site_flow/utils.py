import os
import pickle
from pathlib import Path
from typing import Sequence, Iterable, List, Optional, Type, Union, Tuple
from functools import partial
import shutil

import pandas as pd
import numpy as np
import tifffile as tiff
import SimpleITK as sitk
import skimage
import scipy
from scipy.special import erf
from scipy.signal import convolve
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F

__class = ['utils', 'detection', 'registration', 'fp filter', 'trajectory', 'compute intensity']

# =====
# Utils
# =====

_EPSILON = 1e-10
PathType = Union[str, Path, Iterable[str], Iterable[Path]]

def nrrd_to_tif(nrrd_path: PathType, tif_path: PathType):
    """Turn `.nrrd` to `.tif`

    Args:
        nrrd_path (PathType): _description_
        tif_path (PathType): _description_
    """

    nrrd_image = sitk.ReadImage(nrrd_path)
    sitk.WriteImage(nrrd_image, tif_path)

def tif_to_nrrd(tif_path: PathType, nrrd_path: PathType):
    """Turn `.tif` to `.nrrd`

    Args:
        tif_path (PathType): _description_
        nrrd_path (PathType): _description_
    """

    tif_image = sitk.ReadImage(tif_path)
    sitk.WriteImage(tif_image, nrrd_path)

def delete_single_file_folders(root_folder):
    for folder_path, _, files in os.walk(root_folder, topdown=False):
        if len(files) == 1:
            print(f"Deleting folder with a single file: {folder_path}")
            shutil.rmtree(folder_path)

def delete_empty_folders(folder_path):
    items = os.listdir(folder_path)

    for item in items:
        item_path = os.path.join(folder_path, item)
        
        if os.path.isdir(item_path):
            delete_empty_folders(item_path)

    if not os.listdir(folder_path):
        print(f"Deleting empty folder: {folder_path}")
        os.rmdir(folder_path)

def delete_files_with_extension(directory, extension):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # if filename.endswith(extension):
            if extension in filename:
                file_path = os.path.join(root, filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

def delete_folders(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        # 删除文件
        # for name in files:
        #     file_path = os.path.join(root, name)
        #     os.remove(file_path)  
        # 删除文件夹
        for name in dirs:
            dir_path = os.path.join(root, name)
            shutil.rmtree(dir_path)

def delete_cell_folders(directory):
    for field in os.listdir(directory):
        field_path = os.path.join(directory, field)
        for site_num in os.listdir(field_path):
            field_site_path = os.path.join(field_path, site_num)
            delete_folders(field_site_path)

# ===========
# Helper Func
# ===========

def mask_to_coordinate(
    matrix: np.ndarray, probability: float = 0.5
) -> np.ndarray:
    """Convert the prediction matrix into a list of coordinates.

    NOTE - plt.scatter uses the x, y system. Therefore any plots
    must be inverted by assigning x=c, y=r!

    Args:
        matrix: Matrix representation of spot coordinates.
        image_size: Default image size the grid was layed on.
        probability: Cutoff value to round model prediction probability.

    Returns:
        Array of r, c coordinates with the shape (n, 2).
    """
    if not matrix.ndim == 2:
        raise ValueError("Matrix must have a shape of (r, c).")
    if not matrix.shape[0] == matrix.shape[1] and not matrix.shape[0] >= 1:
        raise ValueError("Matrix must have equal length >= 1 of r, c.")
    assert np.max(matrix) <= 1 and np.max(matrix) >= 0, 'Matrix must be prediction probability'

    # Turn prob matrix into binary matrix (0-1)
    binary_matrix = (matrix > probability).astype(int)

    # Label connected regions
    labeled_array = skimage.measure.label(binary_matrix)

    # Compute the centorid coordinates of each conneted regions
    properties = skimage.measure.regionprops(labeled_array)
    centers = [prop.centroid for prop in properties]
    coords = np.array(centers)

    return coords

def coordinate_to_mask(
    coords: np.ndarray, image_size: int = 128, n: int = 1, sigma: float = None, size_c: int = None
) -> np.ndarray:
    """Return np.ndarray of shape (n, n): r, c format.

    Args:
        coords: List of coordinates in r, c format with shape (n, 2).
        image_size: Size of the image from which List of coordinates are extracted.
        n: Size of the neighborhood to set to 1 or apply Gaussian filter.
        sigma: Standard deviation for Gaussian kernel. If None, no Gaussian filter is applied.
        size_c: If empty, assumes a squared image. Else the length of the r axis.

    Returns:
        The prediction matrix as numpy array of shape (n, n): r, c format.
    """
    nrow = ncol = image_size
    if size_c is not None:
        ncol = size_c

    prediction_matrix = np.zeros((nrow, ncol))

    for r, c in coords:
        # Consider bonuder
        r_min = max(0, r - n)
        r_max = min(nrow, r + n + 1)
        c_min = max(0, c - n)
        c_max = min(ncol, c + n + 1)

        # Assign values along pre人iction matrix 
        if sigma is None:
            prediction_matrix[r_min:r_max, c_min:c_max] = 255
        else:
            y, x = np.ogrid[-n:n+1, -n:n+1]
            gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            gaussian_kernel /= gaussian_kernel.sum()  
            prediction_matrix[r_min:r_max, c_min:c_max] += gaussian_kernel[:r_max-r_min, :c_max-c_min]

    return prediction_matrix

def euclidean_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return the euclidean distance between two the points (x1, y1) and (x2, y2)."""
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

def offset_euclidean(offset: List[tuple]) -> np.ndarray:
    """Calculates the euclidean distance based on row_column_offsets per coordinate."""
    return np.sqrt(np.sum(np.square(np.array(offset)), axis=-1))

def _get_offsets(
    pred: np.ndarray, true: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> List[tuple]:
    """Return a list of (r, c) offsets for all assigned coordinates.

    Args:
        pred: List of all predicted coordinates.
        true: List of all ground truth coordinates.
        rows: Rows of the assigned coordinates (along "true"-axis).
        cols: Columns of the assigned coordinates (along "pred"-axis).
    """
    return [
        (true[r][0] - pred[c][0], true[r][1] - pred[c][1]) for r, c in zip(rows, cols)
    ]

def linear_sum_assignment(
    matrix: np.ndarray, cutoff: float = None
) -> Tuple[list, list]:
    """Solve the linear sum assignment problem with a cutoff.

    A problem instance is described by matrix matrix where each matrix[i, j]
    is the cost of matching i (worker) with j (job). The goal is to find the
    most optimal assignment of j to i if the given cost is below the cutoff.

    Args:
        matrix: Matrix containing cost/distance to assign cols to rows.
        cutoff: Maximum cost/distance value assignments can have.

    Returns:
        (rows, columns) corresponding to the matching assignment.
    """
    # Handle zero-sized matrices (occurs if true or pred has no items)
    if matrix.size == 0:
        return [], []

    # Prevent scipy to optimize on values above the cutoff
    if cutoff is not None and cutoff != 0:
        matrix = np.where(matrix >= cutoff, matrix.max(), matrix)

    row, col = scipy.optimize.linear_sum_assignment(matrix)

    if cutoff is None:
        return list(row), list(col)

    # As scipy will still assign all columns to rows
    # We here remove assigned values falling below the cutoff
    nrow = []
    ncol = []
    for r, c in zip(row, col):
        if matrix[r, c] <= cutoff:
            nrow.append(r)
            ncol.append(c)
    return nrow, ncol

# =======
# Dection
# =======

def spotlearn_norm(img: np.array):

    # get　maxima and minima
    non_zero_indices = np.where(img != 0)
    if len(non_zero_indices[0])!=0:
        non_zero_values = img[non_zero_indices]
        max_val = np.max(img)
        min_val = np.min(non_zero_values)

        # normalize
        non_zero_indices = img != 0
        a = np.zeros_like(img)
        a[non_zero_indices] = (img[non_zero_indices] - min_val) / (max_val - min_val)
    else:
        a = np.zeros_like(img)

    return a


# ============
# Registration
# ============

def rigid_registration(fixed: np.ndarray, moving: np.ndarray):
    """Perform rigid registration

    Args:
        fixed (np.ndarray): fixed image
        moving (np.ndarray): moving image

    Returns:
        _type_: _description_
    """
    
    assert fixed.shape == moving.shape
    assert fixed.ndim == 2
    fixed = sitk.GetImageFromArray(fixed)
    moving = sitk.GetImageFromArray(moving)

    # 设置刚体变换参数
    initial_transform = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler2DTransform())

    # 创建刚体变换的图像配准器
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1, minStep=0.001,
                                                                 numberOfIterations=500)
    registration_method.SetInitialTransform(initial_transform)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # 执行刚体配准
    final_transform = registration_method.Execute(fixed, moving)

    # 对移动图像进行重采样得到配准后的图像
    moving_registered = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())

    return moving_registered, final_transform

def rigid_registration_centroid(fixed_img, moving_img):
    # 读取固定图像和移动图像
    fixed = fixed_img
    moving = moving_img
    # 获取fixed图像的尺寸
    fixed_size = fixed.GetSize()

    # 计算fixed图像的中心坐标
    center_index = [size // 2 for size in fixed_size]

    # 设置刚体变换参数，仅设置旋转参数
    initial_transform = sitk.Euler2DTransform()
    initial_transform.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint(center_index))  # 设置旋转中心
    initial_transform.SetTranslation((0.0, 0.0))  # 设置平移为零

    # 创建刚体变换的图像配准器
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1, minStep=0.001,
                                                                 numberOfIterations=500)
    registration_method.SetInitialTransform(initial_transform)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # 设置平移参数的缩放为0，以保持其值为0
    registration_method.SetOptimizerScales([0.000001, 1.0, 1.0])  # X, Y, Translation

    # 执行刚体配准
    final_transform = registration_method.Execute(fixed, moving)

    # 对移动图像进行重采样得到配准后的图像
    moving_registered = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())

    return moving_registered, final_transform

def image_registration_recursive(image_stack, composite_transform, registered_stack):
    fixed_image = sitk.GetImageFromArray(image_stack[0])
    for z in range(1, image_stack.shape[0]):
        img = sitk.GetImageFromArray(image_stack[z])
        rigid_transforms = sitk.CompositeTransform(2)
        for idx in range(0, z):
            rigid_transforms.AddTransform(composite_transform.GetNthTransform(idx))
        img = sitk.Resample(img, fixed_image, rigid_transforms, sitk.sitkLinear, 0.0, img.GetPixelID())
        registered_stack.append(sitk.GetArrayFromImage(img))
    return registered_stack

def image_registration(image_stack, transform, registered_stack):
    fixed_image = sitk.GetImageFromArray(image_stack[0])
    for z in range(1, image_stack.shape[0]):
        img = sitk.GetImageFromArray(image_stack[z])
        img = sitk.Resample(img, fixed_image, transform[z - 1], sitk.sitkLinear, 0.0, img.GetPixelID())
        registered_stack.append(sitk.GetArrayFromImage(img))
    return registered_stack

def reg_recursive(stack, composite_transform):
    '''Return registred stack with given transform'''

    registered_stack = []
    registered_stack.append(stack[0])
    fixed_image = sitk.GetImageFromArray(stack[0])
    for z in range(1, stack.shape[0]):
        img = sitk.GetImageFromArray(stack[z])
        rigid_transforms = sitk.CompositeTransform(2)

        for idx in range(0, z):
            rigid_transforms.AddTransform(composite_transform.GetNthTransform(idx))
        # only onece resample for each moving image
        img = sitk.Resample(img, fixed_image, rigid_transforms, sitk.sitkLinear, 0.0, img.GetPixelID())
        registered_stack.append(sitk.GetArrayFromImage(img))

    return registered_stack


def get_global_transform(transform_file: str, num_frame: int) -> list:
    '''return transform list corresponding to each time step
    
    Returns:
        global transform seqence, len == num_frame - 1 
    '''

    with open(transform_file, 'rb') as file:
        # read total local transform sequence, actually list[<sitk.CompositeTransform>]
        last_composite_transform = pickle.load(file)

    composite_transform = last_composite_transform[0]  # <sitk.CompositeTransform>
    rigid_transform = []
    for i in range(1, num_frame):
        tmp_transforms = sitk.CompositeTransform(2)

        for idx in range(0, i):
            tmp_transforms.AddTransform(composite_transform.GetNthTransform(idx))
        rigid_transform.append(tmp_transforms)

    return rigid_transform


def coord_reg_to_raw(reg_x, reg_y, transform):
    '''apply'''

    (org_x, org_y) = transform.TransformPoint((reg_x, reg_y))
    return org_x, org_y


# =========
# FP Filter
# =========

def gaussian_filter(pic):
    fit_flag, _, fit_res = gaussian_fit(pic, 3, fix_sigma=False, abs_bg=False)

    if fit_flag == False:
        return True, 0
    else:
        if fit_res['sigma'] < 0.4 or fit_res['sigma'] > 1.4:
            return False, fit_res['sigma']
        else:
            return False, fit_res['sigma']
        
class NN_CNN_Classifier(nn.Module):
    def __init__(self, input_channel, output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channel, 4, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.unsqueeze(2)

        x = self.relu1(self.bn1((self.conv1(x))))
        x = self.relu2(self.bn2((self.conv2(x))))
        h_2 = x.view(x.size(0), -1)
        # print(h_2.shape)
        h_2 = F.relu(self.fc1(h_2))
        h_2 = self.fc2(h_2)
        y_pred = self.sigmoid(h_2)

        return y_pred, h_2


def load_rf_classifier(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

def load_nn_classifier(model, path):
    INPUT_SIZE = 11
    INPUT_DIM = INPUT_SIZE * INPUT_SIZE
    OUTPUT_DIM = 1
    if model == 'cnn':
        model = NN_CNN_Classifier(1, OUTPUT_DIM)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(path))
    return model

def fp_filter(pic, rf_model=None, nn_model=None):
    if rf_model:
        X = []
        img = pic.reshape(-1)
        X.append(img)
        pred = rf_model.predict(X)
        rf_res = int(pred[0])
    else:
        rf_res = None

    if nn_model:
        img_min = np.min(pic)
        img_max = np.max(pic) 
        img = (pic - img_min) / (img_max - img_min)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.unsqueeze(0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img = img.to(device=device, dtype=torch.float32)
        
        y_pred, _ = nn_model(img)
        nn_res = y_pred.detach().cpu().numpy().squeeze()
    else:
        nn_res = None

    return rf_res, nn_res

# ==========
# Trajectory
# ==========

def track_filter(track_data: pd.DataFrame, patch_len = [2, 3], search_range=3, memory=3):
    
    valid_particles = []
    grouped = track_data.groupby('particle')

    for particle, group in grouped:
        group = group.sort_values(by='frame')

        if len(group) not in patch_len:
            valid_particles.append(particle)

        coords = group[['x', 'y']].to_numpy()
        times = group['frame'].to_numpy()

        distances = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
        time_diffs = np.diff(times)

        if np.all(distances <= search_range) and np.all(time_diffs <= memory):
            valid_particles.append(particle)

    filtered_data = track_data[track_data['particle'].isin(valid_particles)]
    return filtered_data

def frame_filter_(track_data: pd.DataFrame):
    invalid_particles = set()
    grouped = track_data.groupby('frame')

    for frame, group in grouped:
        particles_in_frame = group['particle'].unique()
        
        if len(particles_in_frame) > 1:
            min_particle = min(particles_in_frame)
            for particle in particles_in_frame:
                if particle != min_particle:  
                    invalid_particles.add(particle)

    filtered_data = track_data[~track_data['particle'].isin(invalid_particles)]

    return filtered_data

def reindex_particle(data):
    particle_counts = data['particle'].value_counts()
    particle_mapping = {particle: index for index, particle in enumerate(particle_counts.index)}
    data['particle'] = data['particle'].map(particle_mapping)
    
    return data

def link_patches(data, search_range, memory):
    linked_trajectories = {}  # 存储已链接的轨迹
    # {0: [[49.5, 60.5, 0], [46.388888888888886, 60.22222222222222, 1], [46.66666666666666, 61.0, 2]]}

    for i in range(len(data)):
        current_particle = data.loc[i, 'particle']
        current_frame = data.loc[i, 'frame']
        current_x = data.loc[i, 'x']
        current_y = data.loc[i, 'y']

        if current_particle not in linked_trajectories:
            linked_trajectories[current_particle] = []

        single_spot = []
        # [x, y, t]
        single_spot.append(current_x)
        single_spot.append(current_y)
        single_spot.append(current_frame)
        linked_trajectories[current_particle].append(single_spot)

    # 链接
    patcile_list = list(linked_trajectories.keys())
    track_particle = {}
    # {0: 1, 1: 2, 2: 4, 3: 4, 4: 6, 5: 6, 6: 7, 7: 8}
    for i in range(len(patcile_list)):
        old_index = patcile_list[i]
        last_x = linked_trajectories[old_index][-1][0]
        last_y = linked_trajectories[old_index][-1][1]
        last_frame = linked_trajectories[old_index][-1][2]
        for j in range(i + 1, len(patcile_list)):
            new_index = patcile_list[j]
            begin_x = linked_trajectories[new_index][0][0]
            begin_y = linked_trajectories[new_index][0][1]
            begin_frame = linked_trajectories[new_index][0][2]
            if ((begin_frame - last_frame < memory) & (begin_frame - last_frame > 0)) \
                    & ((begin_x - last_x) ** 2 + (begin_y - last_y) ** 2 <= search_range ** 2):
                track_particle[old_index] = new_index
                break

    repalce_index = list(track_particle.keys())
    for item in reversed(repalce_index):
        old_value = track_particle[item]
        new_value = item
        data.loc[data['particle'] == old_value, 'particle'] = new_value

    data = reindex_particle(data)

    return data

def max_projection(images):
    # Convert images to numpy array
    images_np = np.array(images)
    # Take the maximum pixel value along the depth axis (axis=0)
    max_proj = np.max(images_np, axis=0)
    return max_proj

def relative_filter(df, cluster_centers, percentage=0.9):

    for idx, cluster_center in enumerate(cluster_centers, start=0):
        distances = []
        for i in range(len(df)):
            if df.loc[i, 'particle'] == idx:
                site_x = df.loc[i, 'x']
                site_y = df.loc[i, 'y']
                distance = ((site_x - cluster_center[0]) ** 2 + (site_y - cluster_center[1]) ** 2) ** 0.5
                distances.append((i, distance))

        # 按距离升序排序
        distances.sort(key=lambda x: x[1])

        # 保留百分比的点
        num_points_to_keep = int(len(distances) * percentage)
        points_to_keep = [i for i, _ in distances[:num_points_to_keep]]

        # 删除不保留的点
        df = df.drop(df.index[points_to_keep])
        df = df.reset_index(drop=True)
    return df

def same_time_traj_filter(df, cluster_centers):
    filtered_rows = []

    for idx, cluster_center in enumerate(cluster_centers, start=0):
        cluster_center_x, cluster_center_y = cluster_center[0], cluster_center[1]

        for i in range(len(df)):
            if df.loc[i, 'particle'] == idx:
                site_x, site_y, frame = df.loc[i, 'x'], df.loc[i, 'y'], df.loc[i, 'frame']
                distance = ((site_x - cluster_center_x) ** 2 + (site_y - cluster_center_y) ** 2) ** 0.5

                same_time_traj_rows = df[(df['frame'] == frame) & (df['particle'] == idx)]
                if not same_time_traj_rows.empty:
                    min_distance_idx = same_time_traj_rows[['x', 'y']].apply(
                        lambda row: ((row['x'] - cluster_center_x) ** 2 + (row['y'] - cluster_center_y) ** 2) ** 0.5,
                        axis=1).idxmin()
                    if not pd.isnull(min_distance_idx):
                        min_distance_row = same_time_traj_rows.loc[min_distance_idx]
                        filtered_rows.append(min_distance_row)

    df_filtered = pd.DataFrame(filtered_rows)
    df_filtered = df_filtered.drop_duplicates()
    df_filtered = df_filtered.sort_index().reset_index(drop=True)

    return df_filtered

def link_cluster(spots_data, site_num=2, mode=None, threshold=None):
    if not spots_data.empty:
        X = np.array(spots_data[['x', 'y']])

        kmeans = KMeans(n_clusters=site_num)
        kmeans.fit(X)

        # 输出聚类结果
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        spots_data['particle'] = labels

        # if mode == 'r':
        #     # spots_data_filtered = relative_filter(spots_data, cluster_centers, threshold)
        #     spots_data_filtered = spots_data
        # else:
        #     raise ValueError

        # 删除同轨迹同时间点
        spots_data_filtered_final = same_time_traj_filter(spots_data, cluster_centers)

    else:
        spots_data['particle'] = None
        spots_data_filtered_final = spots_data

    return spots_data_filtered_final, cluster_centers

def get_coor_from_mask():
    pass

def traj_completion():
    pass

# =================
# Compute Intensity
# =================

# template utils functions

def create_tp_template():
    csv_data = pd.DataFrame()
    column_names = ['POSITION_T', 'Org_X', 'Org_Y',
                    'X', 'Y', 'sigma', 'Background', 'Signal']
    
    for column_name in column_names:
        csv_data[column_name] = pd.Series(dtype='float64')

    return csv_data

def create_total_template():
    csv_data = pd.DataFrame()
    column_names = ['particle_index', 'POSITION_T', 'Reg_X', 'Reg_Y', 'Org_X', 'Org_Y',
                    'local_maxima', 'Fit_X', 'Fit_Y', 'Fit_amp', 'Fit_offset', 'photon_number']
    
    for column_name in column_names:
        csv_data[column_name] = pd.Series(dtype='float64')

    return csv_data

# process funcitons

def link_linear_interpolation(csv_TPdata, csv_data, tiffFrame):
    for i in range(len(csv_TPdata)):
        if i == 0:
            NowFrame = int(csv_TPdata.loc[i, 'POSITION_T'])
            for j in range(int(NowFrame) + 1):
                csv_data.loc[j, 'POSITION_T'] = j
                csv_data.loc[j, 'Reg_X'] = csv_TPdata.loc[i, 'Reg_X']
                csv_data.loc[j, 'Reg_Y'] = csv_TPdata.loc[i, 'Reg_Y']
        # begin with i=1
        else:  # if (i > 0) and (i < len(csv_TPdata) - 1):
            BeforeFrame = int(csv_TPdata.loc[i - 1, 'POSITION_T'])
            NowFrame = int(csv_TPdata.loc[i, 'POSITION_T'])
            # NextFrame = int(csv_TPdata.loc[i + 1, 'POSITION_T'])

            if NowFrame == BeforeFrame + 1:
                csv_data.loc[NowFrame, 'POSITION_T'] = NowFrame
                csv_data.loc[NowFrame, 'Reg_X'] = csv_TPdata.loc[i, 'Reg_X']
                csv_data.loc[NowFrame, 'Reg_Y'] = csv_TPdata.loc[i, 'Reg_Y']
            else:
                x1 = float(csv_TPdata.loc[i - 1, 'Reg_X'])
                y1 = float(csv_TPdata.loc[i - 1, 'Reg_Y'])
                x2 = float(csv_TPdata.loc[i, 'Reg_X'])
                y2 = float(csv_TPdata.loc[i, 'Reg_Y'])
                Tstep = NowFrame - BeforeFrame
                Xstep = (x2 - x1) / Tstep
                Ystep = (y2 - y1) / Tstep
                n = 0
                # csv_data to NowFrame-1
                for j in range(BeforeFrame, NowFrame + 1):
                    csv_data.loc[j, 'POSITION_T'] = j
                    csv_data.loc[j, 'Reg_X'] = x1 + Xstep * n
                    csv_data.loc[j, 'Reg_Y'] = y1 + Ystep * n
                    n += 1

    lastindex = len(csv_TPdata) - 1
    lastline = int(csv_TPdata.loc[lastindex, 'POSITION_T'])
    for k in range(lastline, tiffFrame):
        csv_data.loc[k, 'POSITION_T'] = k
        csv_data.loc[k, 'Reg_X'] = csv_TPdata.loc[lastindex, 'Reg_X']
        csv_data.loc[k, 'Reg_Y'] = csv_TPdata.loc[lastindex, 'Reg_Y']

    for i in range(len(csv_TPdata)):
        TPindex = csv_TPdata.loc[i, 'POSITION_T']
        csv_data.loc[TPindex, 'TP_Flag'] = csv_TPdata.loc[i, 'TP_Flag']

    csv_data['particle_index'] = csv_TPdata.loc[0, 'particle']

    return csv_data

def trans_coor_reg2raw(csv_data, rigid_transform):
    for i in range(len(csv_data)):
        frame = int(csv_data.loc[i, 'POSITION_T'])
        if frame == 0:
            csv_data.loc[i, 'Org_X'] = csv_data.loc[i, 'Reg_X']
            csv_data.loc[i, 'Org_Y'] = csv_data.loc[i, 'Reg_Y']
        else:
            (reg_x, reg_y) = (csv_data.loc[i, 'Reg_X'], csv_data.loc[i, 'Reg_Y'])
            (org_x, org_y) = rigid_transform[frame - 1].TransformPoint((reg_x, reg_y))
            csv_data.loc[i, 'Org_X'] = org_x
            csv_data.loc[i, 'Org_Y'] = org_y
    return csv_data

# preprocess

def bpass(image, lnoise=1, lobject=3.4, field=False, noclip=False):
    # lnoise（噪声尺度）、lobject（目标物体尺度）
    nf = image.shape[0]
    height = 128
    width = 128

    b = float(lnoise)
    w = int(np.round(lobject > (2. * b))) # 根据目标物体尺度和噪声尺度，判断哪些频率成分应该被保留，生成一个布尔数组 w。
    N = 2 * w + 1 # 卷积核大小

    r = (np.arange(N) - w) / (2. * b) # 生成一个频率坐标数组 r，用于生成巴特沃斯滤波核。
    xpt = np.exp(-r ** 2) # 根据频率坐标 r 计算一个高斯型的滤波核
    xpt = xpt / np.sum(xpt) # 对滤波核进行归一化，确保其总和为 1
    factor = np.sum(xpt ** 2) - 1 / N # 计算一个用于归一化滤波结果的因子

    # 生成用于 x 和 y 方向卷积的核 gx 和 gy，以及用于 x 和 y 方向的滤波核 bx 和 by
    gx = xpt
    # gy = np.copy(gx)
    gy = gx.T
    kernel_g = np.outer(gx, gy)

    # bx = np.arange(N) - 1. / N
    bx = np.zeros(N, dtype=float) - 1./N
    # by = np.copy(bx)
    by = bx.T

    if field:
        # 根据频率核大小选择需要的频率坐标，并进行归一化
        if N % 4 == 1:
            indx = 2 * np.arange(w + 1, dtype=int)
        else:
            indx = 1 + (2 * np.arange(w, dtype=int))
        gy = gy[indx]
        gy = gy / np.sum(gy)
        nn = len(indx)
        # by = np.arange(nn) - 1. / nn
        by = np.zeros(nn, dtype=float) - 1. / nn

    res = np.copy(image)
    res = res.astype(np.float32)
    res_g = np.copy(image)
    res_g = res_g.astype(np.float32)
    res_b = np.copy(image)
    res_b = res_b.astype(np.float32)

    # do x and y convolutions
    for i in range(nf):
        g = np.apply_along_axis(lambda x: convolve(x, gx, mode='same'), axis=1, arr=image[i])
        g = np.apply_along_axis(lambda x: convolve(x, gy, mode='same'), axis=0, arr=g)

        b = np.apply_along_axis(lambda x: convolve(x, bx, mode='same'), axis=1, arr=image[i])
        b = np.apply_along_axis(lambda x: convolve(x, by, mode='same'), axis=0, arr=b)

        res[i] = g - b
        res_g[i] = g
        res_b[i] = b

    if noclip:
        return res / factor#, res_g, res_b
    else:
        return np.where(res / factor > 0, res / factor, 0)#, res_g, res_b

# gaussian fitting

def get_max_coordinate(img, x, y, range_value):
    # 设置边界，确保不会越界
    x_min = max(x - range_value, 0)
    x_max = min(x + range_value + 1, img.shape[0])
    y_min = max(y - range_value, 0)
    y_max = min(y + range_value + 1, img.shape[1])

    # 获取在指定范围内的子数组
    sub_array = img[x_min:x_max, y_min:y_max]

    if np.all(sub_array == 0):
        relative_max_index = (x, y)

    else:
        # 找到子数组中最大值的坐标
        max_index = np.unravel_index(np.argmax(sub_array), sub_array.shape)

        # 转换为相对于原始数组的坐标
        relative_max_index = (x_min + max_index[0], y_min + max_index[1])

    return relative_max_index

def get_pic(image: np.ndarray, x: float, y: float, pixel_window: int = 3) -> np.ndarray:

    xint = int(np.round(x))
    yint = int(np.round(y))
    # local maixima
    xint_raw, yint_raw = get_max_coordinate(image, xint, yint, 2)  # 最大像素值点坐标 (x,y)

    # assert 0<=xint_raw-pixelWindow<128 and 0<=yint_raw-pixelWindow<128 and 0<=xint_raw+pixelWindow+1<128 and 0<=yint_raw-pixelWindow<128, f'OUT OF RANGE: t{i+1} x:{yint_raw} y:{xint_raw}'
    x_range = range(xint_raw - pixel_window, xint_raw + pixel_window + 1)
    y_range = range(yint_raw - pixel_window, yint_raw + pixel_window + 1)

    if 0<=xint_raw-pixel_window<128 and 0<=yint_raw-pixel_window<128 and 0<=xint_raw+pixel_window+1<128 and 0<=yint_raw+pixel_window+1<128:
        try:
            pic = image[x_range][:, y_range]
        except IndexError:
            print(f'x_range: [{xint_raw-pixel_window}, {xint_raw+pixel_window+1}] y_range: [{yint_raw-pixel_window}, {yint_raw+pixel_window+1}]')
        pic_raw = pic.copy()
    else:
        pic = np.zeros((len(x_range), len(y_range)))
        for i_idx, x in enumerate(x_range):
            for j_idx, y in enumerate(y_range):
                if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                    pic[i_idx, j_idx] = image[x, y]
    
    # fill zero
    if np.count_nonzero(pic) > 0:
        # min_nonzero = np.min(pic[np.nonzero(pic)])
        min_nonzero = np.median(pic[np.nonzero(pic)])
        pic[pic == 0] = min_nonzero

    return pic, xint_raw, yint_raw

def local_background(pic: np.ndarray) -> np.ndarray:

    assert pic.ndim == 2
    pic_dim = pic.shape
    # pic is a numpy array, 1st-dim = rows, 2nd-dim = cols
    x_dim = pic_dim[1]
    y_dim = pic_dim[0]

    x_border = np.zeros(2 * x_dim, float)  # 定义x方向上的边界

    x_border[0:x_dim] = pic[0,]  # 将图像最上面一行的像素值赋值给边界
    x_border[x_dim:(2 * x_dim)] = pic[y_dim - 1,]  # 将图像最下面一行的像素值赋值给边界

    x = np.zeros(2 * x_dim, float)
    x[0:x_dim] = np.arange(x_dim)  # 生成一个长度为x_dim的一维数组，数组元素为0到x_dim-1
    x[x_dim:(2 * x_dim)] = np.arange(x_dim)  # 生成一个长度为x_dim的一维数组，数组元素为0到x_dim-1

    y_border = np.zeros(2 * y_dim, float)
    y_border[0:y_dim] = pic[:, 0]
    y_border[y_dim:(2 * y_dim)] = pic[:, x_dim - 1]

    y = np.zeros(2 * y_dim, float)
    y[0:y_dim] = np.arange(y_dim)
    y[y_dim:(2 * y_dim)] = np.arange(y_dim)

    # Following the method of Bevington, p. 96
    delta_x = 2 * x_dim * (x ** 2).sum() - (x.sum()) ** 2
    a = (1. / delta_x) * (((x ** 2).sum()) * (x_border.sum()) - (x.sum()) * ((x * x_border).sum()))
    b = (1. / delta_x) * (2 * x_dim * ((x * x_border).sum()) - (x.sum()) * (x_border.sum()))

    delta_y = 2 * y_dim * (y ** 2).sum() - (y.sum()) ** 2
    c = (1. / delta_y) * (((y ** 2).sum()) * (y_border.sum()) - (y.sum()) * ((y * y_border).sum()))
    d = (1. / delta_y) * (2 * y_dim * ((y * y_border).sum()) - (y.sum()) * (y_border.sum()))

    # The offset which is returned is averaged over each edge in x, and each edge in y.
    # The actual offset needs to be corrected for the tilt of the plane.
    # Then, the 2 offsets are averaged together to give a single offset.
    offset = (a - d * (y_dim - 1) / 2.0 + c - b * (x_dim - 1) / 2.0) / 2.0

    # Print some values
    # print('slope:', b, d)
    # print('offset:', offset)
    # print('adjusted: ', a - d * (y_dim - 1) / 2.0, c - b * (x_dim - 1) / 2.0)

    # now define the background plane in terms of the fit parameters 现在根据拟合参数定义背景平面
    plane = np.zeros((y_dim, x_dim), float)
    for i in range(0, x_dim):
        for j in range(0, y_dim):
            plane[j, i] = offset + b * float(i) + d * float(j)
    return plane

def gaussian_2d(xy, offset, amp, x0, y0, sigma):
    '''Define gaussian func template'''

    a = 1 / (2 * sigma * sigma)
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    inner = a * ((x - x0) ** 2 + (y - y0) ** 2)

    return offset + (amp * np.exp(-inner))

def gaussian_fit(
        pic: np.ndarray,
        pixel_window: int = 3,
        fix_sigma: bool = False,
        abs_bg: bool = True
    ):

    nx, ny = pic.shape
    blacklevel = np.median(pic)
    bn_width = 1.0  # background noise

    # linear fitting to abstract backgroud
    if abs_bg:
        pixelsRBkg = local_background(pic=pic)
        pixelsRBkgSub = pic - pixelsRBkg
        pixelsRBkgCorr = (pixelsRBkgSub > 0) * pixelsRBkgSub
        image = pixelsRBkgCorr
    else:
        image = pic

    # boundary condition.  border is set to zero
    image[0, :] = 0.0
    image[nx - 1, :] = 0.0
    image[:, 0] = 0.0
    image[:, ny - 1] = 0.0

    xint_local = pixel_window; yint_local = pixel_window
    x_range_local = range(xint_local - pixel_window, xint_local + pixel_window + 1)
    y_range_local = range(yint_local - pixel_window, yint_local + pixel_window + 1)

    # creating initial guess of parameters.
    sigma = 1
    initial_guess = (np.mean(image), np.max(image), xint_local, yint_local, sigma)

    try:
        
        yi, xi = np.meshgrid(y_range_local, x_range_local)
        xyi = np.vstack([xi.ravel(), yi.ravel()])
        weights = 1.0 / (np.abs(image) + bn_width ** 2)
        # 定义高斯中心点的取值范围
        x_limit = (xint_local - pixel_window, xint_local + pixel_window)
        y_limit = (yint_local - pixel_window, yint_local + pixel_window)

        if fix_sigma: 
            eps = 1e-6
            bounds = ([float('-inf'), float('-inf'), x_limit[0], y_limit[0], sigma - eps],
                        [float('inf'), float('inf'), x_limit[1], y_limit[1], sigma + eps])
        else:
            bounds = ([float('-inf'), float('-inf'), x_limit[0], y_limit[0], -1],
                        [float('inf'), float('inf'), x_limit[1], y_limit[1], 100])

        # calling curve_fit.
        popt, _ = curve_fit(gaussian_2d, xyi, image.ravel(), p0=initial_guess, sigma=weights.ravel(), bounds=bounds)

        # offset_fit, amp_fit, x_loacl_fit, y_loacl_fit = popt
        prop_key_list = ['offset', 'amp', 'x', 'y', 'sigma']

        # reforming resulting curve fit into a two-dimensional array
        zfit = gaussian_2d(xyi, *popt)
        zfit = zfit.reshape(nx, ny)

        return True, image, dict(zip(prop_key_list, popt))

    except RuntimeError as e:
        offset_fit, amp_fit, x_loacl_fit, y_loacl_fit = 0, 0, 3, 3
        return False, image, {
            'offset': 0, 
            'amp': 0, 
            'x': 3, 
            'y': 3, 
            'sigma': 1
        }

# compute intensity

def get_intensity(pic, x_loacl_fit, y_loacl_fit, psf_width=1.7):
        
    nx, ny = pic.shape

    # calculation of photon number using gaussian mask technique
    popt = np.zeros(3)
    popt[1] = x_loacl_fit
    popt[2] = y_loacl_fit
    array = np.arange(nx*ny)
    xarr = array % nx
    yarr = array / nx
    yarr = np.floor(yarr).astype(int)

    F = 1.0 / (np.sqrt(2.0) * psf_width)
    a = F * (yarr - 0.5 - popt[1])
    b = F * (yarr + 0.5 - popt[1])
    c = F * (xarr - 0.5 - popt[2])
    d = F * (xarr + 0.5 - popt[2])
    ls_mask = 0.25 * (erf(a) - erf(b)) * (erf(c) - erf(d))
    ls_mask_2d = ls_mask.reshape(pic.shape)
    sum_val = np.sum(ls_mask ** 2)
    N = np.sum(pic * ls_mask_2d)
    photon_number = N / sum_val

    return photon_number

def least_sqr_fit(traj_res, raw_stack, psf_width=1.7, bpass=False, random_sample_when_zero=False):
    if bpass:
        raw_stack = bpass(raw_stack, field=False, noclip=False)

    for i in range(len(traj_res)):
        index = int(traj_res.loc[i, 'POSITION_T'])
        img = raw_stack[index].copy()
        x = traj_res.loc[i, 'Org_Y']
        y = traj_res.loc[i, 'Org_X']

        pic, xint_raw, yint_raw = get_pic(img, x, y, pixel_window=3)
        fit_flag, pic, fit_res = gaussian_fit(pic, pixel_window=3, fix_sigma=True)
        intensity = get_intensity(pic, fit_res['x'], fit_res['y'], psf_width)

        # consider when 
        if intensity == 0 and random_sample_when_zero:
           pass
        
        else:
            traj_res.loc[i, 'Fit_X'] = yint_raw - 3 + fit_res['y']
            traj_res.loc[i, 'Fit_Y'] = xint_raw - 3 + fit_res['x']
            traj_res.loc[i, 'Fit_amp'] = fit_res['amp']
            traj_res.loc[i, 'Fit_offset'] = fit_res['offset']
            traj_res.loc[i, 'photon_number'] = intensity

    return traj_res, raw_stack

# site trace computation

def traj_compute(traj_tp_data, raw_stack, rigid_transform, total_frame, random_sample_when_zero=False):
    traj_tp_data['TP_Flag'] = 1
    new_columns = {'x': 'Reg_X', 'y': 'Reg_Y', 'frame': 'POSITION_T'}
    traj_tp_data.rename(columns=new_columns, inplace=True)
    traj_tp_data['Org_X'] = None
    traj_tp_data['Org_Y'] = None
    traj_tp_data['X'] = None
    traj_tp_data['Y'] = None

    traj_res = create_total_template()
    traj_res = link_linear_interpolation(traj_tp_data, traj_res, total_frame)
    traj_res = trans_coor_reg2raw(traj_res, rigid_transform)
    traj_res, bpass_stack = least_sqr_fit(traj_res, raw_stack, random_sample_when_zero=False)

    return traj_res, bpass_stack

# empty trace computation

def random_sample_with_gaussian(img, center, sigma):
    # np.random.seed(1015)
    distances = np.linalg.norm(np.argwhere(img != 0) - center, axis=1)
    probabilities = np.exp(-distances ** 2 / (2 * sigma ** 2))
    probabilities /= np.sum(probabilities)
    chosen_index = np.random.choice(len(probabilities), p=probabilities)
    chosen_coordinate = np.argwhere(img != 0)[chosen_index]

    return chosen_coordinate 

def empty_compute(raw_stack, rigid_transform, total_frame):
    traj_res = create_total_template()
    traj_tp_res = pd.DataFrame()
    column_names = ['x', 'y', 'frame', 'particle', 'TP_Flag']
    for column_name in column_names:
        traj_tp_res[column_name] = pd.Series(dtype='float64')

    # random get coor
    first_img = raw_stack[0].copy()
    random_coordinate = random_sample_with_gaussian(first_img, (63,63), sigma=10)

    # non_zero_indices = np.argwhere(first_img != 0)
    # random_index = np.random.choice(len(non_zero_indices))
    # random_coordinate = non_zero_indices[random_index]
    pixel_value = first_img[random_coordinate[0], random_coordinate[1]]  # reg_x,reg_y

    for i in range(total_frame):
        traj_tp_res.loc[i, 'Reg_X'] = random_coordinate[1]
        traj_tp_res.loc[i, 'Reg_Y'] = random_coordinate[0]
        traj_tp_res.loc[i, 'POSITION_T'] = i

    traj_res = link_linear_interpolation(traj_tp_res, traj_res, total_frame)
    traj_res = trans_coor_reg2raw(traj_res, rigid_transform)
    traj_res, bpass_stack = least_sqr_fit(traj_res, raw_stack)

    return traj_res, bpass_stack