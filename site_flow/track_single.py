import pandas as pd
import trackpy as tp
import tifffile as tiff
import os
import numpy as np


def link_trajectories(data, search_range, memory):
    linked_trajectories = {}

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

    patcile_list = list(linked_trajectories.keys())
    track_particle = {}
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

    particle_counts = data['particle'].value_counts()
    particle_mapping = {particle: index for index, particle in enumerate(particle_counts.index)}
    data['particle'] = data['particle'].map(particle_mapping)

    return data

def trackpy_link(spots_data, num_frames):
    new_columns = {'y [px]': 'y', 'x [px]': 'x', 'z': 'frame'}
    spots_data.rename(columns=new_columns, inplace=True)

    if not spots_data.empty:
        # linking
        traj = tp.link_df(spots_data, search_range=7, memory=5)
        traj_filter = tp.filter_stubs(traj, 2)
        traj_filter = traj_filter.reset_index(drop=True)

        # lingking again
        if not(traj_filter.empty):
            # check traces num
            has_duplicates = traj_filter['frame'].duplicated().any()

            # multi-patch
            if has_duplicates and len(traj_filter) > 2:

                # result_trajectories_tmp_tmp_1 = link_trajectories(traj_filter, search_range=40, memory=5)
                result_trajectories_tmp_tmp_2 = link_trajectories(traj_filter, search_range=40, memory=10)
                result_trajectories_tmp_tmp_3 = link_trajectories(result_trajectories_tmp_tmp_2, search_range=20, memory=100)
                result_trajectories_tmp_tmp_4 = link_trajectories(result_trajectories_tmp_tmp_3, search_range=10, memory=200)
                result_trajectories_tmp_tmp_5 = link_trajectories(result_trajectories_tmp_tmp_4, search_range=5, memory=num_frames)
                tmp_result_trajectories = result_trajectories_tmp_tmp_5.copy()

                # 1 site
                result_trajectories_tmp_tmp_5 = result_trajectories_tmp_tmp_5.loc[result_trajectories_tmp_tmp_5.groupby('frame')['particle'].idxmin()]
                result_trajectories = tp.link_df(result_trajectories_tmp_tmp_5, search_range=200, memory=num_frames)

            # single-patch
            else:
                result_trajectories = tp.link_df(traj_filter, search_range=120, memory=num_frames)
                tmp_result_trajectories = None
        else:
            traj_filter['particle'] = None
            result_trajectories = traj_filter
            tmp_result_trajectories = None
    else:
        spots_data['particle'] = None
        result_trajectories = spots_data
        tmp_result_trajectories = None

    return result_trajectories, tmp_result_trajectories

def spot_track_single(field, cell_folder, people):
    cell_idx = os.path.basename(cell_folder)
    cell_index_num = cell_idx.split('_')[1]

    raw_img = tiff.imread(os.path.join(cell_folder, 'imgs_raw_mask_reg_rcs.tif'))
    raw_img = raw_img[0, :, :, :]
    img_dim = raw_img.shape
    num_frames = img_dim[0]
    spots_data = pd.read_csv(os.path.join(cell_folder, 'cell_mask_reg.csv'), index_col=False)
    
    result_trajectories, tmp_result_trajectories = trackpy_link(spots_data, people, num_frames)
    if tmp_result_trajectories is not None:
        tmp_result_trajectories.to_csv(os.path.join(cell_folder, 'trajectories_data_raw.csv'))
    result_trajectories.to_csv(os.path.join(cell_folder, 'trajectories_data.csv'))

    if os.path.exists(os.path.join(cell_folder, 'filtered_cell_mask_reg.csv')):
        filtered_spots_data = pd.read_csv(os.path.join(cell_folder, 'filtered_cell_mask_reg.csv'), index_col=False)

        if not filtered_spots_data.empty:
            new_columns = {'y [px]': 'y', 'x [px]': 'x', 'z': 'frame'}
            filtered_spots_data.rename(columns=new_columns, inplace=True)

            result_trajectories = trackpy_link(filtered_spots_data, people, num_frames)
            result_trajectories.to_csv(os.path.join(cell_folder, 'filtered_trajectories_data.csv'))

    print(f'field {field:^4s}: cell {cell_index_num:^7s} track processing is done.')