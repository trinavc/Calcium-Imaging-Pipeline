import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import time
import caiman as cm
from caiman.motion_correction import MotionCorrect
import tifffile

def main():
    start = time.time()

    try:
        cv2.setNumThreads(0)
    except:
        pass

    logging.basicConfig(format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                        level=logging.DEBUG)

    fnames = "/home/trinav/NormCorrePipeline/Calcium-Imaging-Pipeline/20231017.tif"
    print(fnames)
    fnames = [fnames]     # Assuming the file exists, no need to download in this case

    # Load the movie
    m_orig = cm.load_movie_chain(fnames)
    downsample_ratio = .2
    m_orig.resize(1, 1, downsample_ratio).play(q_max=99.5, fr=30, magnification=2)

    # Motion correction parameters
    max_shifts = (6, 6)
    strides = (48, 48)
    overlaps = (24, 24)
    max_deviation_rigid = 3
    pw_rigid = False
    shifts_opencv = True
    border_nan = 'copy'

    # Start the cluster
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    mc = MotionCorrect(fnames, dview=dview, max_shifts=max_shifts,
                       strides=strides, overlaps=overlaps,
                       max_deviation_rigid=max_deviation_rigid,
                       shifts_opencv=shifts_opencv, nonneg_movie=True,
                       border_nan=border_nan)

    mc.motion_correct(save_movie=True)
    m_rig = cm.load(mc.mmap_file)
    #bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(int)
    #plt.imshow(mc.total_template_rig, cmap='gray')

    # Compute average of the motion corrected images
    average = m_rig.mean(axis=0)
    average.save('//home/trinav/NormCorrePipeline/Calcium-Imaging-Pipeline/avg_mc202.tif')

    # Compute Std Dev of the Motion Corrected Images
    stddev = m_rig.std(axis=0)
    stddev.save('/home/trinav/NormCorrePipeline/Calcium-Imaging-Pipeline/std_mc202.tif')

    # Grouped Z 20 avg of the motion corrected images
    converted = m_rig.copy()
    [z, x, y] = converted.shape
    groupSize = 20
    images = np.zeros((np.ceil(z / groupSize).astype(int), x, y))
    for i in range(0, z, groupSize):
        j = i // groupSize
        if i + groupSize < z:
            images[j, :, :] = np.mean(converted[i:i+groupSize, :, :], axis=0)
        else:
            images[j, :, :] = np.mean(converted[i:z, :, :], axis=0)

    fname = '/home/trinav/NormCorrePipeline/Calcium-Imaging-Pipeline/z20avg_mc202.tif'
    tifffile.imsave(fname, images.astype(np.float32))

    end = time.time()
    print("Elapsed Time is: ", end - start)

if __name__ == '__main__':
    main()
