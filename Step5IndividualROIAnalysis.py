import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def load_mat_file(file_path):
    data = sio.loadmat(file_path)
    return data

class Step5IndividualROIAnalysis:
    def __init__(self):
        deleteN_input = input("Enter deleteN (blank frames, post-stim blank frames) separated by comma: ")
        self.deleteN = [int(x) for x in deleteN_input.split(',')]
        self.greenDot = list(range(1, self.deleteN[0] + 1))
        self.negativeMethod = int(input("Enter negativeMethod (0, 1, or 2): "))
        self.LB = [1, -30, 5, 0.00001, 5, 0]
        self.UB = [1.5, 360, 180, 1.5, 180, .2]
        self.Trims = self.deleteN

def process_data(data, options):
    preStimBlanks = options.deleteN[0]
    postStimBlanks = options.deleteN[1]

    processed_data = {}
    
    for key in data.keys():
        if '__' in key:
            continue
        
        roi_data = data[key].flatten()

        if not np.issubdtype(roi_data.dtype, np.number):
            continue
        # Remove pre and post-stim blanks
        roi_data = roi_data[preStimBlanks: -postStimBlanks] if postStimBlanks > 0 else roi_data[preStimBlanks:]
        
        # Handle negative values based on negativeMethod
        if options.negativeMethod == 1:
            roi_data[roi_data < 0] = 0
        elif options.negativeMethod == 2:
            roi_data -= roi_data.min()
        
        processed_data[key] = roi_data
        
    return processed_data

def calculate_dff(intensity, f0s):
    if len(f0s) == 0:
        raise ValueError("f0s is empty, cannot calculate DFF")
    f0_mean = np.mean(f0s)
    if f0_mean == 0:
        raise ValueError("Mean of f0s is zero, cannot calculate DFF")
    return (intensity - f0_mean) / f0_mean

def generate_report(data, save_path, options):
    roi_number = 1
    for roi_key in data:
        roi_data = data[roi_key]
        
        # Dummy data for DFF0, trial-averaged traces, and tuning curve fittings
        dff0_data = calculate_dff(roi_data, roi_data[:options.deleteN[0]])
        trial_averaged_data = np.mean(dff0_data.reshape(-1, 12, 10), axis=2)
        tuning_curve_data = np.mean(trial_averaged_data, axis=0)
        
        plt.figure(figsize=(10, 8))
        plt.suptitle(f"ROI {roi_number}")
        
        # Intensity plot
        plt.subplot(2, 2, 1)
        plt.plot(roi_data)
        plt.title("Intensity")
        
        # DFF0 traces
        plt.subplot(2, 2, 2)
        plt.plot(dff0_data)
        plt.title("DFF0 Traces")
        
        # Trial averaged traces (per orientations)
        plt.subplot(2, 2, 3)
        plt.plot(trial_averaged_data)
        plt.title("Trial Averaged Traces")
        
        # Tuning curve fittings
        plt.subplot(2, 2, 4)
        plt.plot(tuning_curve_data)
        plt.title("Tuning Curve Fittings")
        
        # Adding parameters as text
        params_text = f"deleteN: {options.deleteN}\nnegativeMethod: {options.negativeMethod}\nLB: {options.LB}\nUB: {options.UB}"
        plt.gcf().text(0.02, 0.5, params_text, fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{save_path}/ROI_{roi_number}_report.png")
        plt.close()
        
        roi_number += 1

def main():
    file_path =  "/Users/trinav/Downloads/NewData/stack/20231017_130307_01_dg_yuhan_step2__angleNo12_trial10.mat" #input("Enter the path to the ROI_*.mat file: ")
    save_path = "/Users/trinav/Downloads/NormcorreGit/Calcium-Imagining-Pipeline/results" #input("Enter the directory to save reports: ")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data = load_mat_file(file_path)
    options = Step5IndividualROIAnalysis()
    processed_data = process_data(data, options)
    generate_report(processed_data, save_path, options)
    print("Reports generated successfully.")

main()
