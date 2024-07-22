import os
import SimpleITK as sitk
from skimage import measure
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import tifffile as tiff

import pickle
from scipy.special import erf
from scipy.signal import convolve

def local_background_python(pic):
    # print pic
    # Get the dimensions of the local patch

    pic_dim = pic.shape
    # pic is a numpy array, 1st-dim = rows, 2nd-dim = cols
    x_dim = pic_dim[1]  
    y_dim = pic_dim[0]

    x_border = np.zeros(2 * x_dim, float) 

    x_border[0:x_dim] = pic[0,] 
    x_border[x_dim:(2 * x_dim)] = pic[y_dim - 1,] 

    x = np.zeros(2 * x_dim, float)
    x[0:x_dim] = np.arange(x_dim) 
    x[x_dim:(2 * x_dim)] = np.arange(x_dim) 

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

    # now define the background plane in terms of the fit parameters 
    plane = np.zeros((y_dim, x_dim), float)
    for i in range(0, x_dim):
        for j in range(0, y_dim):
            plane[j, i] = offset + b * float(i) + d * float(j)
    return plane

def get_max_coordinate(img, x, y, range_value):
    x_min = max(x - range_value, 0)
    x_max = min(x + range_value + 1, img.shape[0])
    y_min = max(y - range_value, 0)
    y_max = min(y + range_value + 1, img.shape[1])

    sub_array = img[x_min:x_max, y_min:y_max]

    if np.all(sub_array == 0):
        relative_max_index = (x, y)
    else:
        max_index = np.unravel_index(np.argmax(sub_array), sub_array.shape)
        relative_max_index = (x_min + max_index[0], y_min + max_index[1])

    return relative_max_index

def createTPCSV():
    csv_data = pd.DataFrame()
    column_names = ['POSITION_T', 'Org_X', 'Org_Y',
                    'X', 'Y', 'sigma', 'Background', 'Signal']
    for column_name in column_names:
        csv_data[column_name] = pd.Series(dtype='float64')
    return csv_data

def createCSV():
    csv_data = pd.DataFrame()
    column_names = ['particle_index', 'POSITION_T', 'Reg_X', 'Reg_Y', 'Org_X', 'Org_Y',
                    'local_maxima', 'Fit_X', 'Fit_Y', 'Fit_amp', 'Fit_offset', 'photon_number']
    for column_name in column_names:
        csv_data[column_name] = pd.Series(dtype='float64')
    return csv_data

def bpass(image, lnoise=1, lobject=3.4, field=False, noclip=False):
    nf = image.shape[0]
    height = 128
    width = 128

    b = float(lnoise)
    w = int(np.round(lobject > (2. * b))) 
    N = 2 * w + 1 

    r = (np.arange(N) - w) / (2. * b)
    xpt = np.exp(-r ** 2)
    xpt = xpt / np.sum(xpt) 
    factor = np.sum(xpt ** 2) - 1 / N

    gx = xpt
    gy = gx.T
    kernel_g = np.outer(gx, gy)

    bx = np.zeros(N, dtype=float) - 1./N
    by = bx.T

    if field:
        if N % 4 == 1:
            indx = 2 * np.arange(w + 1, dtype=int)
        else:
            indx = 1 + (2 * np.arange(w, dtype=int))
        gy = gy[indx]
        gy = gy / np.sum(gy)
        nn = len(indx)
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

def gaussian_2d(xy, offset, amp, x0, y0):
    sigma = 1
    a = 1 / (2 * sigma * sigma)
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    inner = a * (x - x0) ** 2
    inner += a * (y - y0) ** 2
    return offset + (amp * np.exp(-inner))

# Gaussian Fitting to get more precise central coordinates
def least_sqr_fit(csv_data, tiff_data, psf_width=1.7):
    tiff_data_copy = tiff_data.copy()
    # tiff_data = bpass(tiff_data, field=False, noclip=False)
    for i in range(len(csv_data)):
        index = int(csv_data.loc[i, 'POSITION_T'])
        img = tiff_data[index].copy()

        x = csv_data.loc[i, 'Org_Y']
        y = csv_data.loc[i, 'Org_X']
        xint = int(np.round(x))
        yint = int(np.round(y))
        # local maixima
        max_coordinate = get_max_coordinate(img, xint, yint, 2)  
        xint_raw = max_coordinate[0]
        yint_raw = max_coordinate[1]

        pixelWindow = 3
        x_range = range(xint_raw - pixelWindow, xint_raw + pixelWindow + 1)
        y_range = range(yint_raw - pixelWindow, yint_raw + pixelWindow + 1)

        if 0<=xint_raw-pixelWindow<128 and 0<=yint_raw-pixelWindow<128 and 0<=xint_raw+pixelWindow+1<128 and 0<=yint_raw+pixelWindow+1<128:
            try:
                pic = img[x_range][:, y_range]
            except IndexError:
                print(f'frame: {index+1} x_range: [{xint_raw-pixelWindow}, {xint_raw+pixelWindow+1}] y_range: [{yint_raw-pixelWindow}, {yint_raw+pixelWindow+1}]')
            pic_raw = pic.copy()
        else:
            pic = np.zeros((len(x_range), len(y_range)))
            for i_idx, x in enumerate(x_range):
                for j_idx, y in enumerate(y_range):
                    if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
                        pic[i_idx, j_idx] = img[x, y]

        if np.count_nonzero(pic) > 0:
            # min_nonzero = np.min(pic[np.nonzero(pic)])
            min_nonzero = np.median(pic[np.nonzero(pic)])
            pic[pic == 0] = min_nonzero

        pixelsRBkg = local_background_python(pic=pic)
        pixelsRBkgSub = pic - pixelsRBkg
        pixelsRBkgCorr = (pixelsRBkgSub > 0) * pixelsRBkgSub

        maxvalue = pic[3][3]
        csv_data.loc[i, 'local_maxima'] = maxvalue

        bn_width = 1.0  # background noise
        # psf_width = psf_width
        s = pic.shape
        nx = s[0]
        ny = s[1]

        blacklevel = np.median(pic)
        image = pixelsRBkgCorr
        # image = pic

        # boundary condition.  border is set to zero
        image[0, :] = 0.0
        image[nx - 1, :] = 0.0
        image[:, 0] = 0.0
        image[:, ny - 1] = 0.0

        xint_local = 3
        yint_local = 3
        x_range_local = range(xint_local - pixelWindow, xint_local + pixelWindow + 1)
        y_range_local = range(yint_local - pixelWindow, yint_local + pixelWindow + 1)
        # creating initial guess of parameters.
        initial_guess = (np.mean(image), np.max(image), xint_local, yint_local)

        try:
            # calling curve_fit.
            yi, xi = np.meshgrid(y_range_local, x_range_local)
            xyi = np.vstack([xi.ravel(), yi.ravel()])
            weights = 1.0 / (np.abs(image) + bn_width ** 2)

            x_limit = (xint_local - 3, xint_local + 3)
            y_limit = (yint_local - 3, yint_local + 3)

            bounds = ([float('-inf'), float('-inf'), x_limit[0], y_limit[0]],
                      [float('inf'), float('inf'), x_limit[1], y_limit[1]])
            popt, _ = curve_fit(gaussian_2d, xyi, image.ravel(), p0=initial_guess, sigma=weights.ravel(), bounds=bounds)

            offset_fit, amp_fit, x_loacl_fit, y_loacl_fit = popt

            # reforming resulting curve fit into a two-dimensional array
            zfit = gaussian_2d(xyi, *popt)
            zfit = zfit.reshape(nx, ny)

        except RuntimeError as e:
            offset_fit, amp_fit, x_loacl_fit, y_loacl_fit = 0, 0, 3, 3

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
        ls_mask_2d = ls_mask.reshape(image.shape)
        sum_val = np.sum(ls_mask ** 2)
        N = np.sum(image * ls_mask_2d)
        photon_number = N / sum_val

        csv_data.loc[i, 'Fit_X'] = yint_raw - 3 + popt[2]
        csv_data.loc[i, 'Fit_Y'] = xint_raw - 3 + popt[1]
        csv_data.loc[i, 'Fit_amp'] = amp_fit
        csv_data.loc[i, 'Fit_offset'] = offset_fit
        csv_data.loc[i, 'photon_number'] = photon_number

    return csv_data, tiff_data


def trackID(csv_TPdata, csv_data, tiffFrame):
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

def corTrans(csv_data, rigid_transform):
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

def random_sample_with_gaussian(img, center, sigma):
    np.random.seed(1015)
    distances = np.linalg.norm(np.argwhere(img != 0) - center, axis=1)

    probabilities = np.exp(-distances ** 2 / (2 * sigma ** 2))
    probabilities /= np.sum(probabilities)

    chosen_index = np.random.choice(len(probabilities), p=probabilities)
    chosen_coordinate = np.argwhere(img != 0)[chosen_index]
    return chosen_coordinate

def emptyLink(cell_folder, reg_imgs, raw_img, rigid_transform):
    trajectory_null = createCSV()
    trajectory_reg_null = pd.DataFrame()
    column_names = ['x', 'y', 'frame', 'particle', 'TP_Flag']
    for column_name in column_names:
        trajectory_reg_null[column_name] = pd.Series(dtype='float64')

    # random get coor
    first_img = reg_imgs[0].copy()
    random_coordinate = random_sample_with_gaussian(first_img, (63,63), sigma=3)

    for i in range(reg_imgs.shape[0]):
        trajectory_reg_null.loc[i, 'Reg_X'] = random_coordinate[1]
        trajectory_reg_null.loc[i, 'Reg_Y'] = random_coordinate[0]
        trajectory_reg_null.loc[i, 'POSITION_T'] = i

    trajectory_null = trackID(trajectory_reg_null, trajectory_null, reg_imgs.shape[0])
    trajectory_null = corTrans(trajectory_null, rigid_transform)
    trajectory_null, bpass_imgs = least_sqr_fit(trajectory_null, raw_img)

    target_img_path = cell_folder + os.sep + r'cell_bpass.tif'
    # tiff.imwrite(target_img_path, bpass_imgs)

    return trajectory_null

def IntensityCompute(cell_folder, reg_imgs, raw_img, rigid_transform, field, cell_index_num, tiffFrame, is_filter=False):
    choices = ['filtered_'] if is_filter else ['']

    for choice in choices:

        trajectories_data_path = os.path.join(cell_folder, choice + 'trajectories_data.csv')
        if os.path.exists(trajectories_data_path):
            trajectories_data = pd.read_csv(os.path.join(cell_folder, choice + 'trajectories_data.csv'), index_col=False)

            if trajectories_data['particle'].empty:
                trajectory_data = emptyLink(cell_folder, reg_imgs, raw_img, rigid_transform)
                target_path = cell_folder + os.sep + choice + 'dataAnalysis_tj_empty_withBg.csv'
                trajectory_data.to_csv(target_path)

            else:
                trajectories_num = int(trajectories_data['particle'].max())
                # different traj process
                for i in range(trajectories_num + 1):
                    # Specify the particle value to get
                    specified_particle = i
                    # Create a boolean mask for rows with the specified particle value
                    mask = trajectories_data['particle'] == specified_particle
                    # Filter the DataFrame using the mask
                    trajectoryTP_data = trajectories_data[mask]
                    trajectoryTP_data = trajectoryTP_data.reset_index(drop=True)

                    trajectoryTP_data['TP_Flag'] = 1
                    new_columns = {'x': 'Reg_X', 'y': 'Reg_Y', 'frame': 'POSITION_T'}
                    trajectoryTP_data.rename(columns=new_columns, inplace=True)
                    trajectoryTP_data['Org_X'] = None
                    trajectoryTP_data['Org_Y'] = None
                    trajectoryTP_data['X'] = None
                    trajectoryTP_data['Y'] = None

                    trajectory_data = createCSV()
                    trajectory_data = trackID(trajectoryTP_data, trajectory_data, tiffFrame)
                    trajectory_data = corTrans(trajectory_data, rigid_transform)
                    trajectory_data, bpass_imgs = least_sqr_fit(trajectory_data, raw_img)

                    target_img_path = cell_folder + os.sep + r'cell_bpass.tif'
                    # tiff.imwrite(target_img_path, bpass_imgs)

                    target_path = cell_folder + os.sep + choice + 'dataAnalysis_tj_' + str(i) + '_withBg.csv'
                    trajectory_data.to_csv(target_path)

        else:
            trajectory_data = emptyLink(cell_folder, reg_imgs, raw_img, rigid_transform)
            target_path = cell_folder + os.sep + choice + 'dataAnalysis_tj_empty_withBg.csv'
            trajectory_data.to_csv(target_path)


    print(f'field {field:^4s}: cell {cell_index_num:^7s} intensity computation is done.')


def GetIntensity(field, cell_folder, is_filter):
    cell_idx = os.path.basename(cell_folder)
    cell_index_num = cell_idx.split('_')[1]

    tiff_path = cell_folder + os.sep + os.path.basename(cell_folder) + '.tif'  # raw stacks
    raw_img = tiff.imread(tiff_path)
    raw_img = np.squeeze(raw_img)
    img_dim = raw_img.shape
    tiffFrame = img_dim[0]
    reg_imgs_path = os.path.join(cell_folder, 'imgs_raw_mask_reg_rcs.tif')
    reg_imgs = tiff.imread(reg_imgs_path)
    reg_imgs = reg_imgs[0, :, :, :]

    record_path = os.path.join(cell_folder, 'rigid_transforms_series.pkl')
    with open(record_path, 'rb') as file:
        last_composite_transform = pickle.load(file)
    composite_transform = last_composite_transform[0]
    rigid_transform = []
    for i in range(1, raw_img.shape[0]):
        tmp_transforms = sitk.CompositeTransform(2)
        for idx in range(0, i):
            tmp_transforms.AddTransform(composite_transform.GetNthTransform(idx))
        rigid_transform.append(tmp_transforms)

    IntensityCompute(cell_folder, reg_imgs, raw_img, rigid_transform, field, cell_index_num, tiffFrame, is_filter)
