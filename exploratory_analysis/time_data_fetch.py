


'''
Time how long it takes to open two images in nifti with nibabel.
The only difference is where the images are stored.
Do this several times and get an average.
'''

import nibabel as nib
import numpy as np
import time
import os
import sys

file1 = '/mnt/HDD18TB/lidfer/ukb_preprocessed/bids/sub-1000034/ses-2/T1_unbiased_brain_uint8.nii.gz'
file2 = '/mnt/CRAI-NAS/all/lidfer/Datasets/ukb/sub-1000034/ses-2/T1_unbiased_brain_uint8.nii.gz'

def time_open_nifti(file):
    start = time.time()
    image = nib.load(file).get_fdata()
    end = time.time()
    return end - start

# Run several times and get average
n = 10
times1 = [time_open_nifti(file1) for _ in range(n)]
times2 = [time_open_nifti(file2) for _ in range(n)]
print(f"Average time to open image in nifti with nibabel, {n} times:")
print(f"File 1: {np.mean(times1)}")
print(f"File 2: {np.mean(times2)}")

print(times1)
print(times2)


#File 1: 0.02285151481628418
#File 2: 0.033293724060058594
