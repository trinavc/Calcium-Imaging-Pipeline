import cv2
import matplotlib.pyplot as plt
import numpy as np
import os.path
import logging
import timeit

start = timeit.timeit()

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.DEBUG)

import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
from caiman.utils.utils import download_demo

fnames = "/Users/trinav/Downloads/NewDataPython copy/20231017.tif"
#input("What file would you like to read in?")
print(fnames)
#fnames = '20230818_01.tif'
fnames = [download_demo(fnames)]     # the file will be downloaded if it doesn't already exist
m_orig = cm.load_movie_chain(fnames)
downsample_ratio = .2  # motion can be perceived better when downsampling in time
m_orig.resize(1, 1, downsample_ratio).play(q_max=99.5, fr=30, magnification=2)   # play movie (press q to exit)

max_shifts = (6, 6)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
strides =  (48, 48)  # create a new patch every x pixels for pw-rigid correction
overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
max_deviation_rigid = 3   # maximum deviation allowed for patch with respect to rigid shifts
pw_rigid = False  # flag for performing rigid or piecewise rigid motion correction
shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)

#start the cluster (if a cluster already exists terminate it)
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

# create a motion correction object
mc = MotionCorrect(fnames, dview=dview, max_shifts=max_shifts,
                  strides=strides, overlaps=overlaps,
                  max_deviation_rigid=max_deviation_rigid, 
                  shifts_opencv=shifts_opencv, nonneg_movie=True,
                  border_nan=border_nan)

# correct for rigid motion correction and save the file (in memory mapped form)
mc.motion_correct(save_movie=True)

# load motion corrected movie
m_rig = cm.load(mc.mmap_file)
bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(int)
#visualize templates
#plt.figure(figsize = (20,10))
plt.imshow(mc.total_template_rig, cmap = 'gray')

import tifffile
# correct for rigid motion correction and save the file (in memory mapped form)
mc.motion_correct(save_movie=True)

# load motion corrected movie
m_rig = cm.load(mc.mmap_file)
bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(int)
#visualize templates
#plt.figure(figsize = (20,10))
plt.imshow(mc.total_template_rig, cmap = 'gray')

#Compute avg of the motion corrected images
average = m_rig.mean(axis=0)
#Visualize average of templates
#plt.imshow(average, cmap = 'gray')
#Save averaged template in chosen directory
#average.save("/Users/trinav/caiman_data/motion_corrected_results/avg_mc202.tif")
#file = input("Please enter the file you would like to save to: ")
#/Users/trinav/caiman_data/motion_corrected_results/avg_mc202.tif
#/Users/trinav/Downloads/NewDataPython/avg/avg_mc202.tif
average.save('/Users/trinav/Downloads/NewDataPython/avg/avg_mc202.tif')

#Compute Std Dev of the Motion Corrected Images
stddev = m_rig.std(axis=0)

#Visualize results
#plt.imshow(stddev, cmap = 'gray')

#Save file in chosen directory
#file = input("Please enter the file you would like to save to: ")
#input("Please enter the file you would like to save to: ")
#/Users/trinav/caiman_data/motion_corrected_results/stdev_mc202.tif
#/Users/trinav/Downloads/NewDataPython/std/std_mc202.tif
stddev.save('/Users/trinav/Downloads/NewDataPython/std/std_mc202.tif')


#Computing grouped Z 20 avg of the motion corrected images
#Convert m_rig images into 3d-array
converted = m_rig.copy()


#Compute Z20 average of the new array -> results in a condensed 3d-array
[z, x, y] = converted.shape
groupSize = 20
nextGroup = groupSize - 1
j = 0
images = np.zeros((np.ceil(z/ groupSize).astype(int), x, y))
for i in range(0, z, groupSize):
    #print(i)
    if j < len(images):
        if i + nextGroup < z:
            images[j, :, :] = np.mean(converted[i:i+nextGroup, :, :], axis=0)
        else:
            images[j, :, :] = np.mean(converted[i:z, :, :], axis=0)
    j = j + 1


#Convert 3d-array to image stack and save to directory of choice
fname = '/Users/trinav/Downloads/NewDataPython/avg/z20avg_mc202.tif'
tifffile.imsave(fname, images.astype(np.float32))
end = timeit.timeit()
print("Elapsed Time is: " + end - start)