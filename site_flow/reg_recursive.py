import SimpleITK as sitk
import numpy as np
import tifffile as tiff
import os
import pickle
import math

def tif_to_nrrd(tif_path, nrrd_path):
    tif_image = sitk.ReadImage(tif_path)
    sitk.WriteImage(tif_image, nrrd_path)

def rigid_registration(fixed_img, moving_img):
    fixed = fixed_img
    moving = moving_img
    fixed_size = fixed.GetSize()
    center_index = [size // 2 for size in fixed_size]

    initial_transform = sitk.Euler2DTransform()
    initial_transform.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint(center_index))  # 设置旋转中心
    initial_transform.SetTranslation((0.0, 0.0)) 

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1, minStep=0.001,
                                                                 numberOfIterations=500)
    registration_method.SetInitialTransform(initial_transform)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerScales([0.000001, 1.0, 1.0])  # X, Y, Translation

    final_transform = registration_method.Execute(fixed, moving)
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

def reg_recursive(field, cell_folder):
    cell_idx = os.path.basename(cell_folder)
    cell_index_num = cell_idx.split('_')[1]

    imgs_raw_mask_path = os.path.join(cell_folder, r'imgs_raw_mask.tif')
    imgs_raw_mask = tiff.imread(imgs_raw_mask_path)

    # 读取时间序列图像
    raw_path = cell_folder + os.sep + r'cellraw_' + str(cell_index_num) + '.tif'
    reg_stack = tiff.imread(raw_path)
    reg_stack = np.squeeze(reg_stack)
    reg_stack = reg_stack.astype(np.float32)
    raw_img = reg_stack.copy()

    # get mask
    mask_img = imgs_raw_mask[1, :, :, :]

    # 循环处理每一帧移动图像
    # rigid_transforms = []   
    reg_tmp_transforms = []  # transform by each frame
    last_composite_transform = []  # final composite transform
    moving_registered_stack = []   # save registered img per frame
    moving_registered_stack.append(reg_stack[0])
    moving_mask_registered_stack = []
    moving_mask_registered_stack.append(mask_img[0])
    final_cell_registered_stack = []  # save registered img
    final_cell_registered_stack.append(reg_stack[0])
    final_mask_registered_stack = []  # save registered mask
    final_mask_registered_stack.append(mask_img[0])

    composite_transform = sitk.CompositeTransform(2)

    for i in range(1, reg_stack.shape[0]):
        fixed_image = sitk.GetImageFromArray(reg_stack[i - 1])  # moving_registered_stack only resample once
        moving_frame = sitk.GetImageFromArray(reg_stack[i])  # reg_stack
        moving_frame.SetSpacing(fixed_image.GetSpacing())
        moving_frame.SetOrigin(fixed_image.GetOrigin())
        moving_frame.SetDirection(fixed_image.GetDirection())

        moving_registered, final_transform = rigid_registration(fixed_image, moving_frame)

        composite_transform.AddTransform(final_transform)
        reg_tmp_transforms.append(final_transform)

        moving_registered_stack.append(sitk.GetArrayFromImage(moving_registered))

    last_composite_transform.append(composite_transform)  # for revovery

    # image registration
    reg_img_tmp = np.array(moving_registered_stack)
    # tiff.imwrite(os.path.join(cell_folder, 'cell_reg_tmp.tif'), reg_img_tmp)

    # mask registration
    moving_mask_registered_stack = image_registration(mask_img, reg_tmp_transforms, moving_mask_registered_stack)
    reg_mask_tmp = np.array(moving_mask_registered_stack)
    threshold = 1
    reg_mask_tmp[reg_mask_tmp >= threshold] = 65000
    reg_mask_tmp[reg_mask_tmp < threshold] = 0
    reg_mask_tmp = reg_mask_tmp.astype(np.uint16)
    # tiff.imsave(os.path.join(data_folder, 'cell_mask_reg_tmp.tif'), reg_mask_tmp)

    # image recursive registration
    final_cell_registered_stack = image_registration_recursive(raw_img, composite_transform, final_cell_registered_stack)
    reg_img = np.array(final_cell_registered_stack)
    # tiff.imwrite(os.path.join(cell_folder, 'cell_reg.tif'), reg_img)

    # mask recursive registration
    final_mask_registered_stack = image_registration_recursive(mask_img, composite_transform, final_mask_registered_stack)
    final_mask_registered_stack = np.array(final_mask_registered_stack)
    # final_mask_registered_stack *= 65000
    threshold = 1
    final_mask_registered_stack[final_mask_registered_stack >= threshold] = 65000
    final_mask_registered_stack[final_mask_registered_stack < threshold] = 0
    final_mask_registered_stack = final_mask_registered_stack.astype(np.uint16)
    # tiff.imwrite(os.path.join(cell_folder, 'cell_mask_reg.tif'), final_mask_registered_stack)

    reg_raw_mask_path = os.path.join(cell_folder, r'imgs_raw_mask_reg_rcs.tif')
    reg_raw_mask = np.stack((reg_img.astype(np.uint16), final_mask_registered_stack))
    tiff.imwrite(reg_raw_mask_path, reg_raw_mask, imagej=True)

    record_path = os.path.join(cell_folder, 'rigid_transforms_series.pkl')
    with open(record_path, 'wb') as file:
        pickle.dump(last_composite_transform, file)

    print(f'field {field:^4s}: cell {cell_index_num:^7s} registration processing is done.')
