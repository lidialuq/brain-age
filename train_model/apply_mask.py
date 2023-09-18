import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import random

def adjust_mask_size(mask_data, image_data):
    '''Make the mask adhere to the data shape by cropping or padding.
    In this case, the mask is from brats and has shape (240,240,155), 
    the data is from the UKB and has shape (182,218,182).
    Input:
        mask_data: 3D numpy array
        image_data: 3D numpy array
    Output:
        mask_data: 3D numpy array, cropped or padded to match image_data
    '''
    mask_shape = mask_data.shape
    data_shape = image_data.shape
    # calculate the differences for each dimension
    diff = np.array(data_shape) - np.array(mask_shape)
    # determine padding or cropping for each dimension
    padding = np.where(diff > 0, diff, 0)
    cropping = np.where(diff < 0, -diff, 0)
    # evenly distribute the padding or cropping 
    pad_before = padding // 2
    pad_after = padding - pad_before
    
    crop_before = cropping // 2
    crop_after = cropping - crop_before
    # crop the mask if needed
    cropped_mask = mask_data[crop_before[0]:-crop_after[0] or None,
                             crop_before[1]:-crop_after[1] or None,
                             crop_before[2]:-crop_after[2] or None]
    # pad the cropped mask
    adjusted_mask = np.pad(cropped_mask, ((pad_before[0], pad_after[0]), 
                                          (pad_before[1], pad_after[1]), 
                                          (pad_before[2], pad_after[2])), mode='constant')

    return adjusted_mask


def apply_mask_to_nifti(image_folder, mask_folder, seed=42):
    '''Randomly select a NIFTI image and mask from the given 
    folders and apply the mask to the image.
    Input:
        image_folder: str, path to folder containing NIFTI images
        mask_folder: str, path to folder containing masks
        seed: int, random seed
    Output:
        masked_data: 3D numpy array, image with mask applied
    '''

    random.seed(seed)
    
    # Get list of patient subfolders
    patient_folders = [os.path.join(image_folder, d) for d in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, d))]

    # Randomly select a patient folder and construct the NIFTI image path
    random_patient_folder = random.choice(patient_folders)
    random_image_path = os.path.join(random_patient_folder, "T1", "T1_brain_to_MNI.nii.gz")
    
    # Randomly select a mask
    subfolders = [os.path.join(mask_folder, d) for d in os.listdir(mask_folder) if os.path.isdir(os.path.join(mask_folder, d))]
    random_subfolder = random.choice(subfolders)
    mask_files = [f for f in os.listdir(random_subfolder) if "seg.nii.gz" in f]
    random_mask_path = os.path.join(random_subfolder, mask_files[0])
    
    # Load the image and mask
    image_data = nib.load(random_image_path).get_fdata()
    mask_data = nib.load(random_mask_path).get_fdata()

    # make the mask (240,240,155) adhere to the data (182,218,182) by removing slices equally from both ends
    mask_data = adjust_mask_size(mask_data, image_data)
    # Apply mask
    masked_data = np.where(mask_data > 0, 0, image_data)
    
    return masked_data


def plot_masked_slices(data, num_slices=10):
    mask_indices = np.where(np.any(data == 0, axis=(1,2)))
    unique_indices = np.unique(mask_indices)
    print(f"Unique indices: {len(unique_indices)}")
    print(unique_indices[:10])

    if len(unique_indices) < num_slices:
        chosen_indices = unique_indices
    else:
        step = len(unique_indices) // num_slices
        chosen_indices = unique_indices[::step][:num_slices]
    print(f"Chosen indices: {chosen_indices}")
    fig, axes = plt.subplots(2, len(chosen_indices)//2, figsize=(15, 7))
    for i, ax in enumerate(axes.flat):
        slice = data[:,:,chosen_indices[i]]
        # rotate slice
        slice = np.rot90(slice)
        ax.imshow(slice, cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    IMAGE_FOLDER = "/home/lidia/CRAI-NAS/all/lidfer/brain-age/data"
    MASK_FOLDER = "/home/lidia/CRAI-NAS/all/BraTS/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"
    SEED = 43

    SEED = random.randint(0, 10000)
    
    masked_data = apply_mask_to_nifti(IMAGE_FOLDER, MASK_FOLDER, SEED)
    plot_masked_slices(masked_data)
