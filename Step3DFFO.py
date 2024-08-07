import zipfile
import tifffile
from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd
import read_roi
import os
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import wiener, butter, filtfilt
from scipy.stats import skew

# Optional GPU usage
try:
    import cupy as cp
    use_gpu = True
except ImportError:
    use_gpu = False

class Step3DFFO:
    def __init__(self, file_path, trial_file, save_path, roi_zip_path, options):
        self.file_path = file_path
        self.trial_file = trial_file
        self.save_path = save_path
        self.roi_zip_path = roi_zip_path
        self.options = options
        self.image_stack = None
        self.trial_data = None
        self.rois = {}
        self.intensity = None
        self.baseline = None
        self.f0s = None
        self.avg_img = None
        self.std_img = None
        self.dff0 = None
        self.bw = None
        self.xy = None
        self.cell_num = None
        self.qt = None
        self.order = None
        self.file4 = None
        self.intensity_raw = None
        self.intensity_neuropil = None
        self.baseline_neuropil = None
        self.neuropil_factor = None
        self.skew_raw = None
        self.neuropil_baseline_vector = None

    def load_data(self):
        # Load the TIFF image stack
        self.image_stack = tifffile.imread(self.file_path)
        
        # Load the experimental trial data
        self.trial_data = loadmat(self.trial_file)
        
        # Extract and load ROI files
        self.load_rois()

    def load_rois(self):
        # Extract ROI files from the ZIP archive
        with zipfile.ZipFile(self.roi_zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.save_path)
            
        # Load each ROI file according to the specified pipeline option
        for roi_key, roi_filename in self.options['roi_files'].items():
            roi_file_path = os.path.join(self.save_path, roi_filename)
            # Check the file format and load accordingly
            if roi_filename.endswith('.roi'):
                self.rois[roi_key] = read_roi.read_roi_file(roi_file_path)
            elif roi_filename.endswith('.npy'):
                self.rois[roi_key] = np.load(roi_file_path)
            elif roi_filename.endswith('.csv'):
                self.rois[roi_key] = pd.read_csv(roi_file_path)
            # Add other conditions based on file format as necessary

    def preprocess_images(self):
        if self.options['image_filter']['name'] == 1:
            self.image_stack = gaussian_filter(self.image_stack, self.options['image_filter']['sigma_number'])
        elif self.options['image_filter']['name'] == 2:
            self.image_stack = median_filter(self.image_stack, size=self.options['image_filter']['size_number'])
        elif self.options['image_filter']['name'] == 3:
            self.image_stack = wiener(self.image_stack, self.options['image_filter']['size_number'])
        elif self.options['image_filter']['name'] == 5:
            b, a = butter(self.options['image_filter']['order'], self.options['image_filter']['fcut'])
            self.image_stack = filtfilt(b, a, self.image_stack, axis=0)

    def extract_rois(self):
        # Implement ROI extraction based on the specified pipeline
        pipeline = self.options['roi_pipeline']
        if pipeline == 'cnmf':
            self.extract_rois_cnmf()
        elif pipeline == 'suite2p':
            self.extract_rois_suite2p()
        elif pipeline == 'imagej':
            self.extract_rois_imagej()
        elif pipeline == 'extract':
            self.extract_rois_extract()

    def extract_rois_cnmf(self):
        # CNMF extraction logic
        cnmf_file = self.options['roi_files']['cnmf']
        cnmf_data = loadmat(cnmf_file)
        self.intensity = cnmf_data['C']  # Extracted signal
        A = cnmf_data['A']  # Spatial components
        dims = cnmf_data['dims'][0]  # Dimensions of the image
        self.bw = A > 0  # Binary masks
        self.bw = self.bw.reshape((dims[0], dims[1], -1), order='F')
        self.xy = [np.column_stack(np.nonzero(self.bw[:, :, i])) for i in range(self.bw.shape[2])]
        self.cell_num = self.intensity.shape[1]

    def extract_rois_suite2p(self):
        # Suite2P extraction logic
        suite2p_file = self.options['roi_files']['suite2p']
        suite2p_data = loadmat(suite2p_file)
        self.intensity = suite2p_data['F']  # Extracted signal
        stat = suite2p_data['stat']
        dims = self.image_stack.shape[1:]
        self.bw = np.zeros((dims[0], dims[1], len(stat)), dtype=bool)
        for i, roi in enumerate(stat):
            ypix = roi['ypix'][0].astype(int)
            xpix = roi['xpix'][0].astype(int)
            self.bw[ypix, xpix, i] = True
        self.xy = [np.column_stack(np.nonzero(self.bw[:, :, i])) for i in range(self.bw.shape[2])]
        self.cell_num = self.intensity.shape[1]

    def extract_rois_imagej(self):
        # ImageJ extraction logic
        imagej_file = self.options['roi_files']['imagej']
        roi_set = read_roi.read_roi_zip(imagej_file)
        dims = self.image_stack.shape[1:]
        self.bw = np.zeros((dims[0], dims[1], len(roi_set)), dtype=bool)
        self.intensity = np.zeros((self.image_stack.shape[0], len(roi_set)))
        self.xy = []
        for i, (roi_name, roi) in enumerate(roi_set.items()):
            mask = np.zeros(dims, dtype=bool)
            mask[roi['y1']:roi['y2'], roi['x1']:roi['x2']] = True
            self.bw[:, :, i] = mask
            self.intensity[:, i] = np.mean(self.image_stack[:, mask], axis=1)
            self.xy.append(np.column_stack(np.nonzero(mask)))
        self.cell_num = self.intensity.shape[1]

    def extract_rois_extract(self):
        # EXTRACT extraction logic
        extract_file = self.options['roi_files']['extract']
        extract_data = loadmat(extract_file)
        self.intensity = extract_data['C']  # Extracted signal
        A = extract_data['A']  # Spatial components
        dims = extract_data['dims'][0]  # Dimensions of the image
        self.bw = A > 0  # Binary masks
        self.bw = self.bw.reshape((dims[0], dims[1], -1), order='F')
        self.xy = [np.column_stack(np.nonzero(self.bw[:, :, i])) for i in range(self.bw.shape[2])]
        self.cell_num = self.intensity.shape[1]

    def find_uniform_coefficient(self):
        # Calculate the uniform coefficient that minimizes the difference between neuropil and actual signal
        residuals = []
        coefficients = np.linspace(0, 1, 100)
        for coefficient in coefficients:
            neuropil_signal = coefficient * self.neuropil_signal()
            residual = np.sum((self.intensity_raw - neuropil_signal)**2)
            residuals.append(residual)
        return coefficients[np.argmin(residuals)]

    def find_roi_specific_coefficient(self):
        # Calculate ROI-specific coefficients that minimize the difference between neuropil and actual signal
        coefficients = []
        for i in range(self.intensity_raw.shape[1]):
            residuals = []
            coeff_range = np.linspace(0, 1, 100)
            for coeff in coeff_range:
                neuropil_signal = coeff * self.neuropil_signal_single_roi(i)
                residual = np.sum((self.intensity_raw[:, i] - neuropil_signal)**2)
                residuals.append(residual)
            coefficients.append(coeff_range[np.argmin(residuals)])
        return np.array(coefficients)

    def neuropil_signal(self):
        # Calculate the neuropil signal for all ROIs
        neuropil_signal = np.zeros_like(self.intensity_raw)
        for i in range(self.intensity_raw.shape[1]):
            neuropil_signal[:, i] = self.neuropil_signal_single_roi(i)
        return neuropil_signal

    def neuropil_signal_single_roi(self, roi_idx):
        # Calculate the neuropil signal for a single ROI, excluding the ROI area itself if exclusion is True
        roi_mask = self.bw[:, :, roi_idx]
        neuropil_mask = np.zeros_like(roi_mask, dtype=bool)
        for x in range(-self.options['neuropil_method']['size'], self.options['neuropil_method']['size']+1):
            for y in range(-self.options['neuropil_method']['size'], self.options['neuropil_method']['size']+1):
                if x**2 + y**2 <= self.options['neuropil_method']['size']**2:
                    neuropil_mask |= np.roll(np.roll(roi_mask, x, axis=0), y, axis=1)
        neuropil_mask &= ~roi_mask
        if self.options['neuropil_method']['exclusion']:
            for i in range(self.bw.shape[2]):
                if i != roi_idx:
                    neuropil_mask &= ~self.bw[:, :, i]
        return np.mean(self.image_stack[:, neuropil_mask], axis=1)

    def calculate_neuropil_signal(self, exclusion, size):
        # Calculate the neuropil signal based on the exclusion flag and neuropil size
        neuropil_signal = np.zeros_like(self.intensity_raw)
        for i in range(self.intensity_raw.shape[1]):
            neuropil_signal[:, i] = self.neuropil_signal_single_roi(i)
        return neuropil_signal

    def fixed_period_baseline(self):
        # Calculate the baseline using a fixed period specified by the user
        fixed_period = self.options['baseline_method']['fix']
        return np.mean(self.intensity_raw[fixed_period, :], axis=0)

    def percentile_baseline(self, percentile):
        # Calculate the baseline using the lowest X% values
        sorted_intensity = np.sort(self.intensity_raw, axis=0)
        num_frames = sorted_intensity.shape[0]
        percentile_index = int(num_frames * percentile / 100)
        return np.mean(sorted_intensity[:percentile_index, :], axis=0)

    def all_blank_periods_baseline(self):
        # Calculate the baseline using all blank periods identified by the user
        blank_periods = self.options['baseline_method']['gray_number']
        return np.mean(self.intensity_raw[blank_periods, :], axis=0)

    def sliding_window_baseline(self):
        # Calculate the baseline using a sliding window approach
        window_size = self.options['baseline_method']['n']
        baseline = np.zeros_like(self.intensity_raw)
        for i in range(self.intensity_raw.shape[0] - window_size + 1):
            window = self.intensity_raw[i:i+window_size, :]
            baseline[i+window_size//2, :] = np.mean(window, axis=0)
        return baseline

    def each_trial_blank_period(self):
        # Calculate the baseline for each trial blank period specified by the user
        blank_periods = self.options['baseline_method']['fix_per_sti']
        baseline = np.zeros((len(blank_periods), self.intensity_raw.shape[1]))
        for i, period in enumerate(blank_periods):
            baseline[i, :] = np.mean(self.intensity_raw[period, :], axis=0)
        return baseline

    def calculate_dff(self):
        # Implement Delta F/F0 calculation
        self.dff0 = (self.intensity_raw - self.baseline) / self.baseline
        self.skew_raw = skew(self.intensity_raw, axis=0)

    def generate_output(self):
        # Save the results following the specified nomenclature
        file_name = f"ROI_{self.file_path.split('/')[-1].split('.')[0]}_{self.options['roi_pipeline']}_NpMethod{self.options['neuropil_method']['method']}.mat"
        if self.options['neuropil_method']['method'] == 1:
            file_name = file_name.replace('.mat', f"_Coe{self.options['neuropil_method']['coefficient']}.mat")
        if not np.isnan(self.options['trace_filter']['method']):
            file_name = file_name.replace('.mat', f"_{self.options['trace_filter']['method']}.mat")
        if self.options['neuropil_method']['exclusion'] == 1 and self.options['neuropil_method']['method'] != 0:
            file_name = file_name.replace('.mat', '_Exclusion.mat')
        if self.options['neuropil_method']['method'] != 0:
            file_name = file_name.replace('.mat', f"_NpSize{self.options['neuropil_method']['size']}.mat")

        output_data = {
            'Intensity': self.intensity,
            'baseline': self.baseline,
            'f0s': self.f0s,
            'avgImg': self.avg_img,
            'stdImg': self.std_img,
            'dff0': self.dff0,
            'bw': self.bw,
            'xy': self.xy,
            'cellNum': self.cell_num,
            'qt': self.qt,
            'order': self.order,
            'file4': self.file4,
            'Intensity_raw': self.intensity_raw,
            'Intensity_neuropil': self.intensity_neuropil,
            'baseline_neuropil': self.baseline_neuropil,
            'options_step5': self.options,
            'neuropilFactor': self.neuropil_factor,
            'skew_raw': self.skew_raw,
            'neuropil_baseline_vector': self.neuropil_baseline_vector
        }
        
        savemat(os.path.join(self.save_path, file_name), output_data)

def get_options_from_user():
    options = {}
    options['roi_pipeline'] = input("Please Enter Analysis Pipeline (cnmf, suite2p, imagej, extract): ").lower()
    options['trace_filter'] = {
        'method': input("Please Enter Trace Filter Method (none, gaussian, movmean): ").lower(),
        'window_size': int(input("Enter window size for the trace filter (if applicable, else -1): "))
    }
    options['neuropil_method'] = {
        'method': int(input("Please Enter Neuropil Method (0 - no subtraction, 1 - user defined, 2 - uniform, 3 - ROI specific): ")),
        'coefficient': float(input("Enter neuropil coefficient (if applicable, else -1): ")),
        'exclusion': bool(input("Exclude signal from other ROIs? (1 - Yes, 0 - No): ")),
        'size': int(input("Enter neuropil size (if applicable, else -1): "))
    }
    options['image_filter'] = {
        'name': int(input("Please Enter Image Filter Method (0 - none, 1 - gaussian, 2 - median, 3 - wiener, 5 - butterworth): ")),
        'size_number': int(input("Enter filter size number (if applicable, else -1): ")),
        'sigma_number': float(input("Enter sigma number for gaussian (if applicable, else -1): ")),
        'order': int(input("Enter order for butterworth (if applicable, else -1): ")),
        'fcut': float(input("Enter cut-off frequency for butterworth (if applicable, else -1): "))
    }
    options['baseline_method'] = {
        'type': int(input("Please Enter Baseline Method (1 - mode, 2 - fixed period, 3 - lowest 10%, 4 - lowest 20%, 5 - all blank periods, 6 - auto with sliding window/mode, 7 - each trial blank period): ")),
        'gray_number': int(input("Enter gray number (if applicable for baseline methods 2/5/7, else -1): ")),
        'fix': int(input("Enter fixed frame number (if applicable for baseline method 2, else -1): ")),
        'fix_per_sti': [int(x) for x in input("Enter list of blank frames per trial, separated by spaces (if applicable for baseline method 7, else -1): ").split()]
    }
    return options

# Define file paths
file_path = '/Users/trinav/Downloads/NewDataPython copy/20231017.tif'
trial_file = '/Users/trinav/Downloads/NewData/stack/20231017_130307_01_dg_yuhan_step2__angleNo12_trial10.mat'
save_path = '/Users/trinav/Downloads/NormcorreGit/Calcium-Imagining-Pipeline/results'
roi_zip_path = '/Users/trinav/Downloads/NewDataPython/RoiSet.zip'

if not os.path.exists(save_path):
        os.makedirs(save_path)

# Get options from user
options = get_options_from_user()

# Initialize and run the pipeline
pipeline = Step3DFFO(
    file_path=file_path, 
    trial_file=trial_file, 
    save_path=save_path, 
    roi_zip_path=roi_zip_path, 
    options=options
)

pipeline.load_data()
pipeline.preprocess_images()
pipeline.extract_rois()
pipeline.correct_neuropil()
pipeline.calculate_baseline()
pipeline.calculate_dff()
pipeline.generate_output()
