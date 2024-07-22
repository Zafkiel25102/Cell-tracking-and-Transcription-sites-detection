import os
import numpy as np
import logging
import shutil

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

import tifffile as tiff
import SimpleITK as sitk
from skimage import measure

from .nets import SpotlearnNet

def normalize(imgs_train):
    imgs_train_normalize = np.zeros_like(imgs_train)
    for i in range(len(imgs_train)):
        img = imgs_train[i].copy()

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
        imgs_train_normalize[i] = a

    return imgs_train_normalize

def mask_filter(mask):
    label_image = measure.label(mask, connectivity=2)
    regions = measure.regionprops(label_image)
    threshold_area = 4
    for region in regions:
        if region.area <= threshold_area:
            mask[label_image == region.label] = 0
    return mask

def tif_to_nrrd(tif_path, nrrd_path):
    tif_image = sitk.ReadImage(tif_path)
    if '_raw_mask' in tif_path:
        tif_image = tif_image[2]
    sitk.WriteImage(tif_image, nrrd_path)

def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(full_img)
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu().numpy()

        output = np.squeeze(output)

        output[output <= out_threshold] = 0
        output[output > out_threshold] = 1

        output = mask_filter(output)

    return output


def predict(imgs_test, model_path, gpus='0'):
    imgs_result = np.zeros_like(imgs_test).astype(np.float32)

    assert len(imgs_result.shape) == 3

    device = torch.device('cuda:'+str(gpus) if torch.cuda.is_available() else 'cpu')

    # model setting
    net = SpotlearnNet(1, 1).to(device)
    checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(checkpoint["model_state_dict"])
    net = torch.nn.DataParallel(net)
    logging.info('Model loaded!')
    net = net.module

    for i in range(imgs_test.shape[0]):
        img = imgs_test[i, :, :].copy().astype(np.float32)
        mask = predict_img(net=net,
                           full_img=img,
                           out_threshold=0.94,
                           device=device)
        imgs_result[i, :, :] = mask

    return imgs_result

def pred_torch(field, field_site_path, cell_idx_tif, modelwtsfname, gpu):
    # make new repo for each cell
    cell_index = cell_idx_tif.split('.')[0]
    cell_index_num = cell_index.split('_')[1]
    new_path = os.path.join(field_site_path, cell_index)
    os.makedirs(new_path,exist_ok=True)

    # copy the raw tiff images
    old_path = os.path.join(field_site_path, cell_idx_tif)
    shutil.copy2(old_path, new_path, follow_symlinks=True)
    # new_nrrd_path = new_path + os.sep + 'imgs_raw.nrrd'
    # tif_to_nrrd(old_path, new_nrrd_path)

    # tif to npy
    new_tiff_path = os.path.join(new_path, cell_idx_tif)
    raw_stack = tiff.imread(new_tiff_path)
    raw_stack = np.squeeze(raw_stack)
    raw_stack_f32 = raw_stack.astype('float32')
    raw_stack_normalize = normalize(raw_stack_f32)
    # npy_path = os.path.join(new_path, 'imgs_test.npy')
    # np.save(npy_path, npy_data_normalize)
    print(f'field {field:^4s}: cell {cell_index_num:^7s} transfrom processing is done.')

    # predict
    # npy_path = cell_folder + os.sep + 'imgs_test.npy'
    # imgs_test = np.load(npy_path)
    imgs_test = raw_stack_normalize
    imgs_result = predict(imgs_test, modelwtsfname, gpu)
    # imgs_result = imgs_result.astype(np.uint16)
    # new_path = cell_folder + os.sep + 'imgs_result.npy'
    # np.save(new_path, imgs_result)

    # transform images to 16-bit binary images
    mask = imgs_result * 65000
    mask = mask.astype(np.uint16)
    # tif_path = cell_folder + os.sep + cell_idx + '_mask.tif'
    # tiff.imwrite(tif_path, mask)

    cell_folder = new_path
    imgs_raw_mask_path = os.path.join(cell_folder, r'imgs_raw_mask.tif')
    imgs_raw_mask = np.stack((raw_stack, mask))
    tiff.imwrite(imgs_raw_mask_path, imgs_raw_mask, imagej=True)

    new_nrrd_path = cell_folder + os.sep + 'imgs_mask_threshold.nrrd'
    # tif_to_nrrd(tif_threshold_path, new_nrrd_path)

    raw_nrrd_path = os.path.join(cell_folder, 'imgs_raw.nrrd')
    # tif_to_nrrd(raw_path, raw_nrrd_path)

    print(f'field {field:^4s}: cell {cell_index_num:^7s} prediction processing is done.')
 
