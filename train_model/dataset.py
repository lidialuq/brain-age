import os
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
import nibabel as nib
import ants
from apply_mask import adjust_mask_size


class UKB(Dataset):
    def __init__(self, root_data, root_masks, shape_transforms, transforms=None, 
            train=True, val_split=0.2):
        self.root = root_data
        self.root_masks = root_masks
        self.transform = transforms
        self.shape_transform = shape_transforms
        self.train = train

        # Load age data from CSV and standardize
        #age_df = pd.read_csv(os.path.join(self.root, 'ukb_TP_02_eid_age_sex.csv'))
        age_df = pd.read_csv(os.path.join('/mnt/HDD18TB/lidfer/ukb_preprocessed', 'ukb_TP_02_eid_age_sex.csv'))
        self.mean_age = age_df['examage'].mean()
        self.std_age = age_df['examage'].std()
        age_df['examage'] = (age_df['examage'] - self.mean_age) / self.std_age
        self.age_map = dict(zip(age_df['eid'], age_df['examage']))

        # Get list of subjects and divide into train and val
        subjects = list(self.age_map.keys())
        print(f'Number of subjects: {len(subjects)}')
        # make sure all subjects exist
        subjects = [s for s in subjects if os.path.exists(os.path.join(self.root, 'bids', 'sub-' + str(s)))]
        print(f'Number of subjects after checking: {len(subjects)}')
        random.shuffle(subjects)
        split_idx = int(len(subjects) * val_split)
        self.train_subjects, self.val_subjects = subjects[split_idx:], subjects[:split_idx]

        # Initialize mask variables and get list of mask folders
        self.cnt_mask_used = 0
        self.mask = None
        self.mask_folders = [os.path.join(self.root_masks, d) for d in os.listdir(self.root_masks) \
            if os.path.isdir(os.path.join(self.root_masks, d))]
        

    def _overlay_mask(self, image):
        # Set mask to None if the current one has been used 10 times
        if self.cnt_mask_used == 12:
            self.cnt_mask_used = 0
            self.mask = None
        # Select a new mask if the current one is None
        if self.mask is None:
            random_mask_folder = random.choice(self.mask_folders)
            mask_file = [f for f in os.listdir(random_mask_folder) if "seg.nii.gz" in f]
            mask_path = os.path.join(random_mask_folder, mask_file[0])
            mask = ants.image_read(mask_path, pixeltype='float').numpy()
            self.mask = adjust_mask_size(mask, image)
        # Overlay mask
        masked_data = np.where(self.mask > 0, 0, image)
        self.cnt_mask_used += 1
        return masked_data

    def __len__(self):
        return len(self.train_subjects) if self.train else len(self.val_subjects)

    def __getitem__(self, index):
        subjects = self.train_subjects if self.train else self.val_subjects
        subject = subjects[index]
        sessions = os.listdir(os.path.join(self.root, 'bids', 'sub-'+str(subject)))
        session = random.choice(sessions)
        img_path = os.path.join(self.root, 'bids', 'sub-'+str(subject), session, 'T1_unbiased_brain_uint8.nii.gz')

        # load with nibabel
        # image = nib.load(img_path).get_fdata()
        # load with ants
        image = ants.image_read(img_path, pixeltype='float').numpy()
        image = self.shape_transform(image)
        image = self._overlay_mask(image)
        label = self.age_map[subject]

        if self.transform:
            image = self.transform(image)

        return image, label

