from pathlib import Path
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def get_paths():
    cwd = Path.cwd()
    # Set paths according to if running on server or locally
    # check if cwd path starts with /mnt
    if cwd.parts[1] == 'mnt':
        data_path = '/mnt/HDD18TB/lidfer/ukb_preprocessed'
        #data_path = '/home/lidfer/data/ukb_preprocessed/ukb_preprocessed'
        masks_path = '/mnt/CRAI-NAS/all/BraTS/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
    elif cwd.parts[1] == 'home':
        data_path = '/home/lidia/HDD18TB/ukb_preprocessed'
        masks_path = '/home/lidia/CRAI-NAS/all/BraTS/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
    return data_path, masks_path
