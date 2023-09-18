import pandas as pd
import os
import numpy as np
import sys
sys.path.append('../')
from config import get_paths
data_path, masks_path = get_paths()


age_df = pd.read_csv(os.path.join(data_path, 'ukb_TP_02_eid_age_sex.csv'))
print(f'average age: {age_df["examage"].mean()}')
print(f'min age: {age_df["examage"].min()}')
print(f'max age: {age_df["examage"].max()}')
print(f'std age: {age_df["examage"].std()}')
print(f'number of subjects: {len(age_df)}')


