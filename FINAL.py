import cv2
import matplotlib.pyplot as plt
import numpy as np
import os.path
import logging
import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
from caiman.utils.utils import download_demo
import cv2
import tifffile as tiff
import os
import read_roi
from scipy import stats
import matplotlib.pyplot as plt
import tifffile
from scipy.io import savemat
import time

start_time = time.time()

def process_roi(roi_file_path):
    roi = read_roi.read_roi_file(roi_file_path)
    coordinates = []
    for key, value in roi.items():
        x = value['x']
        y = value['y']
        for i in range(len(x)):
            coordinates.append((x[i], y[i]))
    # Calculate centroid
    centroid = np.mean(coordinates, axis=0)
    return centroid, coordinates


# Directory containing ROI files
roi_directory = '/Users/trinav/Downloads/NewDataPython/ROISet'

# Process each ROI
centroids = []
masks = []
for roi_file in os.listdir(roi_directory):
    if roi_file.endswith('.roi'):
        roi_path = os.path.join(roi_directory, roi_file)
        centroid, mask_coords = process_roi(roi_path)
        centroids.append(centroid)
        masks.append(mask_coords)


# Convert to numpy arrays
centroids_matrix = np.array(centroids)
masks_matrix = np.array(masks, dtype=object)  # Using dtype=object for variable-length coordinates

# Output the matrices
#centroids_matrix, masks_matrix

# Function to calculate average intensity within the mask for a frame
def average_intensity(mask, frame):
    masked_frame = np.where(mask == 1, frame, 0)
    total_intensity = np.sum(masked_frame)
    count = np.count_nonzero(masked_frame)
    return total_intensity / count if count != 0 else 0

# Path to the original TIFF stack
tiff_stack_path = '/Users/trinav/Downloads/NewDataPython/20231017.tif'

# Load the TIFF stack
tiff_stack = tiff.imread(tiff_stack_path)

# Assuming masks_matrix is obtained from the previous script
F_raw = np.zeros((len(masks_matrix), 1440))

# Process each ROI and overlay its mask on each frame
# Process each ROI and overlay its mask on each frame
for i, mask_coords in enumerate(masks_matrix):
    # Create a binary mask from the coordinates
    mask = np.zeros_like(tiff_stack[0], dtype=np.uint8)
    for coord in mask_coords:
        x, y = coord
        # Convert coordinates to integer and adjust for zero-based indexing
        x_adj = int(x) - 1
        y_adj = int(y) - 1
        if 0 <= x_adj < mask.shape[1] and 0 <= y_adj < mask.shape[0]:
            mask[y_adj, x_adj] = 1  # Set the pixel at the ROI coordinate to 1
    ...


    # Calculate average intensity for each frame
    for j, frame in enumerate(tiff_stack):
        F_raw[i, j] = average_intensity(mask, frame)

# Output the F_raw matrix
#print(F_raw)

def create_neuropil_mask(centroid, image_shape, radius, roi_mask):
    neuropil_mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.circle(neuropil_mask, (int(centroid[0]), int(centroid[1])), radius, 1, -1)
    neuropil_mask = np.logical_and(neuropil_mask, np.logical_not(roi_mask)).astype(np.uint8)
    return neuropil_mask

radius = 30  # Radius for neuropil masks
neuropil_coordinates = []
F_neuropil = np.zeros((len(centroids_matrix), 4000))

for i, (centroid, mask_coords) in enumerate(zip(centroids_matrix, masks_matrix)):
    # Create the ROI mask
    roi_mask = np.zeros_like(tiff_stack[0], dtype=np.uint8)
    for coord in mask_coords:
        x, y = coord
        # Convert coordinates to integer and adjust for zero-based indexing
        x_adj = int(x) - 1
        y_adj = int(y) - 1
        if 0 <= x_adj < roi_mask.shape[1] and 0 <= y_adj < roi_mask.shape[0]:
            roi_mask[y_adj, x_adj] = 1

    # Rest of your code for neuropil mask creation and intensity calculation
    ...


    # Create the neuropil mask
    neuropil_mask = create_neuropil_mask(centroid, tiff_stack[0].shape, radius, roi_mask)
    neuropil_coordinates.append(np.argwhere(neuropil_mask == 1))

    # Calculate average intensity for each frame with neuropil mask
    for j, frame in enumerate(tiff_stack):
        F_neuropil[i, j] = average_intensity(neuropil_mask, frame)

# Convert neuropil coordinates to a numpy array
neuropil_coordinates_matrix = np.array(neuropil_coordinates, dtype=object)

# Output the F_neuropil matrix and neuropil coordinates
#print(F_neuropil)#, neuropil_coordinates_matrix

# Function to calculate F0 based on different methods
def calculate_F0(F_raw, method, num_tests=None, num_frames=None, num_repetitions=None):
    F0 = []
    for row in F_raw:
        if method == 'mode':
            F0.append(calculate_mode_with_histogram(row, bins=50))
        elif method in ['lowest_10_percent', 'lowest_20_percent']:
            # Filter out zero values
            filtered_row = row[row > 0]
            if len(filtered_row) > 0:
                # Determine the percentage to calculate
                percent = 0.1 if method == 'lowest_10_percent' else 0.2
                # Calculate the average of the lowest percent values
                F0.append(np.mean(np.sort(filtered_row)[:int(percent * len(filtered_row))]))
            else:
                # If there are no non-zero values, append zero or handle as needed
                F0.append(0)
        elif method == 'base_time_period':
            if num_tests and num_frames and num_repetitions:
                test_length = len(row) // num_tests
                repetition_length = test_length // num_repetitions
                base_frames = repetition_length // num_frames
                test_averages = [np.mean(row[i * test_length + j * repetition_length : i * test_length + j * repetition_length + base_frames])
                                 for i in range(num_tests) for j in range(num_repetitions)]
                F0.append(np.mean(test_averages))
            else:
                raise ValueError("Parameters for base time period calculation are missing.")
        else:
            raise ValueError("Invalid method chosen.")
    return np.array(F0)

def calculate_mode_with_histogram(data, bins):
    counts, bin_edges = np.histogram(data, bins=bins)
    max_count_index = np.argmax(counts)
    mode_value = (bin_edges[max_count_index] + bin_edges[max_count_index + 1]) / 2
    return mode_value

# User input for method selection
method = "mode"
#input("Choose the method for calculating F0 (mode/lowest_10_percent/lowest_20_percent/base_time_period): ")
num_tests = num_frames = num_repetitions = None
if method == 'base_time_period':
    num_tests = int(input("Enter the number of different tests: "))
    num_frames = int(input("Enter the number of frames to be tested: "))
    num_repetitions = int(input("Enter the number of repetitions: "))

# Assuming F_raw is already calculated
F0 = calculate_F0(F_raw, method, num_tests, num_frames, num_repetitions)

# Output the F0 values
#print(F0)

DFFO = np.zeros_like(F_raw)  # Initialize the DFFO matrix

for i in range(F_raw.shape[0]):  # Iterate over each ROI
    for j in range(F_raw.shape[1]):  # Iterate over each frame
        F = F_raw[i, j] - 0.7 * F_neuropil[i, j]  # Calculate F for the current frame
        DFFO[i, j] = ((F - F0[i]) / F0[i]) * 100 if F0[i] != 0 else 0  # Calculate ΔF/F0 and handle division by zero

# DFFO now contains the ΔF/F0 values for each ROI across all frames
print(DFFO)


# Create a dictionary of the data
data = {
    'masks_matrix': masks_matrix,
    'centroids_matrix': centroids_matrix,
    'F_raw': F_raw,
    'F_neuropil': F_neuropil,
    'F0': F0,
    'DFFO': DFFO
}

# Save to a .mat file
filename = '/Users/trinav/Downloads/NewDataPython/FluorescenceData.mat'
savemat(filename, data)

print(f"Data saved to {filename}")
print("--- %s seconds ---" % (time.time() - start_time))
