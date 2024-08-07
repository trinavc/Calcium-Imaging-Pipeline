import numpy as np
import scipy.io as sio
import scipy.ndimage as ndimage
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import pandas as pd

# Try to import CuPy, if available
try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    USE_CUPY = False

# Define array creation and operations functions
def array(x):
    return cp.array(x) if USE_CUPY else np.array(x)

def delete(arr, indices):
    return cp.delete(arr, indices) if USE_CUPY else np.delete(arr, indices)

def asnumpy(arr):
    return cp.asnumpy(arr) if USE_CUPY else np.array(arr)

def load_mat_file(file_path):
    data = sio.loadmat(file_path)
    return data

# Function for Gaussian fitting
def gaussian(x, amp1, theta, sigma1, amp2, sigma2, dc):
    return amp1 * np.exp(-((x - theta) ** 2) / (2 * sigma1 ** 2)) + amp2 * np.exp(-((x - (theta + 180)) ** 2) / (2 * sigma2 ** 2)) + dc

# Function to filter the image
def filter_image(data, method):
    if method == 1:  # Gaussian
        return ndimage.gaussian_filter(asnumpy(data), sigma=1)
    elif method == 2:  # Median
        return ndimage.median_filter(asnumpy(data), size=3)
    elif method == 3:  # Wiener
        return ndimage.filters.wiener(asnumpy(data), mysize=3)
    elif method == 4:  # Max
        return ndimage.maximum_filter(asnumpy(data), size=3)
    elif method == 5:  # Butterworth
        return ndimage.filters.gaussian_filter(asnumpy(data), sigma=1)
    else:
        return data

# Function to remove frames
def remove_frames(data, deleteN):
    return data[deleteN[0]:-deleteN[1], :] if deleteN[1] > 0 else data[deleteN[0]:, :]

# Function to calculate OSI/DSI
def calculate_indices(dff0, order, options):
    angleNO = options['angleNO']
    trialNO = options['trialNO']
    framePerSti = options['framePerSti']
    framePerSti2 = options['framePerSti2']

    y3 = remove_frames(dff0, options['deleteN'])
    pvalue, y4 = stats.ttest_ind(y3[order[:, -1]], options)

    output = {
        'SSE': [],
        'RSQ': [],
        'maxY': [],
        'minY': [],
        'minOptions': [],
        'DSI_ii': [],
        'OSI_ii': [],
        'gDSI': [],
        'gOSI': [],
        'theta': [],
        'pref_direction': [],
        'FWHM': [],
        'dff0_avg': []
    }

    for ii in range(dff0.shape[1]):
        y2 = dff0[:, ii]
        y3 = remove_frames(y2, options['deleteN'])
        output['SSE'].append(stats.linregress(y3, y4)[0])
        output['RSQ'].append(stats.linregress(y3, y4)[2] ** 2)
        output['maxY'].append(np.max(y4))
        output['minY'].append(np.min(y4))
        output['minOptions'].append(options['minOptions'])
        output['DSI_ii'].append(np.sum(y4))
        output['OSI_ii'].append(np.sum(y4))
        output['gDSI'].append(np.sum(y4))
        output['gOSI'].append(np.sum(y4))
        output['theta'].append(np.sum(y4))
        output['pref_direction'].append(np.sum(y4))
        output['FWHM'].append(np.sum(y4))
        output['dff0_avg'].append(np.mean(y4))

    return output

# Function to fit Gaussian curves
def fit_gaussian(y, x, options):
    popt, pcov = curve_fit(gaussian, x, y, bounds=(options['LB'], options['UB']))
    return popt, np.sqrt(np.diag(pcov))

# Main function
def analyze_roi(filePath, file0, options):
    # Load data
    data = load_mat_file(os.path.join(filePath, file0))

    # Apply filtering
    if 'filterMethod' in options and options['filterMethod']['name'] != 0:
        data = filter_image(data, options['filterMethod']['name'])

    # Remove frames
    if 'deleteN' in options:
        data = remove_frames(data, options['deleteN'])

    # Set NaN values
    if 'nanROI' in options and options['nanROI'] is not None:
        data[data == options['nanROI']] = np.nan

    # Histogram scales
    max_hist = options.get('maximumHist', [0, 5, 250])

    # Plot scales
    plot_scale = options.get('plotscale', 50)

    # Save options
    eps_save = options.get('epsSave', int(input("Enter 1 to save plots as .eps, 0 otherwise: ")))
    curve_save = options.get('curveSave', int(input("Enter 1 to save all the curves, 0 otherwise: ")))
    negative_method = options.get('negativeMethod', int(input("Enter negative method (0 - do nothing, 1 - set negative values to 0, 2 - shift curve): ")))

    # Bounds
    LB = options.get('LB', [1, -30, 5, 0.00001, 5, 0])
    UB = options.get('UB', [1.5, 360, 180, 1.5, 180, .2])

    # Threshold for DSI
    DSI_thresh = options.get('DSI_thresh', 0.5)

    # Criteria for OS and DS
    criteria = options.get('criteria', {})

    # Calculate indices
    indices = calculate_indices(data['dff0'], data['order'], options)

    # Fit Gaussian curves
    popt, pcov = fit_gaussian(data['y'], data['x'], options)

    # Generate plots and save results
    for i in range(data['dff0'].shape[1]):
        plt.figure(figsize=(10, 8))
        plt.subplot(221)
        plt.title(f'ROI {i}')
        plt.plot(data['x'], data['dff0'][:, i])
        plt.subplot(222)
        plt.plot(data['x'], data['dff0'][:, i])
        plt.subplot(223)
        plt.plot(data['x'], data['dff0'][:, i])
        plt.subplot(224)
        plt.plot(data['x'], data['dff0'][:, i])
        plt.savefig(os.path.join(filePath, f'ROI_{i}.png'))
        if eps_save:
            plt.savefig(os.path.join(filePath, f'ROI_{i}.eps'))
        plt.close()

    # Save results to files
    params = {
        'popt': popt,
        'pcov': pcov,
        'indices': indices,
    }
    sio.savemat(os.path.join(filePath, 'params.mat'), params)
    pd.DataFrame(params).to_excel(os.path.join(filePath, 'params.xlsx'))
    np.savetxt(os.path.join(filePath, 'params.txt'), params)

# Example usage
options = {
    'filterMethod': {'name': 1},
    'deleteN': [8, 0],
    'nanROI': np.nan,
    'maximumHist': [0, 5, 250],
    'plotscale': 50,
    'epsSave': 0,
    'curveSave': 1,
    'negativeMethod': 1,
    'LB': [1, -30, 5, 0.00001, 5, 0],
    'UB': [1.5, 360, 180, 1.5, 180, .2],
    'DSI_thresh': 0.5,
    'criteria': {
        'OS': {
            'anovaFlag': [0.05, 1],
            'fitFlag': 0,
            'gOSI_OSI': [0.25, 0, 2],
        },
        'DS': {
            'anovaFlag': [0.05, 1],
            'fitFlag': 0,
            'gOSI_OSI': [0.25, 0.5, 2],
        }
    }
}

# Running the analysis
analyze_roi('/Users/trinav/Downloads/NewData/20231017.tif', '/Users/trinav/Downloads/NewData/stack/20231017_130307_01_dg_yuhan_step2__angleNo12_trial10.mat', options)
