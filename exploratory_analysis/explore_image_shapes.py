'''Open the first 100 images in root and get:
-mean shape for every axis
-max shape for every axis
-min shape for every axis
'''
import nibabel as nib
import numpy as np
import os
import sys
sys.path.append('../')
from config import get_paths

data_path, masks_path = get_paths()
data_path = os.path.join(data_path, 'bids')

# figure out how many subjects have ses-1 and ses-2
def explore_session(): 
    ses2 = 0
    ses3 = 0
    cnt = 0
    for sub in os.listdir(data_path):
        if os.path.exists(os.path.join(data_path, sub, 'ses-2')):
            ses2 += 1
        if os.path.exists(os.path.join(data_path, sub, 'ses-3')):
            ses3 += 1
        # check if there are other sessions not called ses-1, ses-2 or ses-3
        for ses in os.listdir(os.path.join(data_path, sub)):
            if ses not in ['ses-2', 'ses-3']:
                print(f'other session: {ses}')
        cnt += 1
        if cnt % 10000 == 0:
            print(f'cnt: {cnt}')
    print(f'ses-2: {ses2}')
    print(f'ses-3: {ses3}')
    print(f'total: {cnt}')

def get_image_shapes(root):
    shapes = []
    cnt = 0
    for sub in os.listdir(root)[:1000]:
        image_path2 = os.path.join(root, sub, 'ses-2', 'T1_unbiased_brain_uint8.nii.gz')
        image_path3 = os.path.join(root, sub, 'ses-3', 'T1_unbiased_brain_uint8.nii.gz')
        if os.path.exists(image_path2):
            image_path = image_path2
        elif os.path.exists(image_path3):
            image_path = image_path3
        shape = nib.load(image_path).get_fdata().shape
        shapes.append(shape)
        cnt += 1
        if cnt % 100 == 0:
            print(f'cnt: {cnt}')
    shapes = np.array(shapes)
    return shapes

explore_session()
image_shapes = get_image_shapes(data_path)

for axis in range(3):
    print(f'axis {axis}')
    print(f'mean: {image_shapes[:, axis].mean()}')
    print(f'max: {image_shapes[:, axis].max()}')
    print(f'min: {image_shapes[:, axis].min()}')
    # get shape that 90% of the images are equal or smaller than
    print(f'90%: {np.percentile(image_shapes[:, axis], 90)}')
    print(f'95%: {np.percentile(image_shapes[:, axis], 95)}')
    print(f'99%: {np.percentile(image_shapes[:, axis], 99)}')

