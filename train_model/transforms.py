import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import monai.transforms as trans

train_transforms = trans.Compose([
    #trans.CropForegroundd(keys=keys, source_key="image", margin=3, return_coords=False),
    # spatial transforms
    trans.AddChannel(),
    # trans.RandZoom(prob=0.2, min_zoom=0.9, max_zoom=1.1, mode='area'),
    # trans.RandRotate(range_x=0.2, range_y=0.2, range_z=0.2, prob=0.5, padding_mode='zeros', mode='bilinear'),
    # trans.RandRotate90(prob=0.5, max_k=3, spatial_axes=(0, 1)),
    # trans.RandRotate90(prob=0.5, max_k=3, spatial_axes=(0, 2)),
    # trans.RandRotate90(prob=0.5, max_k=3, spatial_axes=(1, 2)),
    # trans.RandFlip(prob=0.5, spatial_axis=0),
    # trans.RandFlip(prob=0.5, spatial_axis=1),
    # trans.RandFlip(prob=0.5, spatial_axis=2),
    # Intensity transforms
    #trans.RandAdjustContrastd(keys=["image"], gamma=(0.5, 1.5), prob=prob),
    #trans.RandHistogramShiftd(keys=["image"], num_control_points=(3, 10), prob=prob),
    #trans.NormalizeIntensity(nonzero=True, channel_wise=True),
    #trans.RandScaleIntensityd(keys=["image"], factors=0.2, prob=prob),
    # pad if needed and crop
    #trans.SpatialPadd(keys=keys, spatial_size=size, mode='constant'),
    #trans.RandSpatialCrop(roi_size=(188, 246, 239), random_size=False),
    trans.EnsureType(data_type='tensor', dtype=torch.float16),
])

val_transforms = trans.Compose([
    trans.NormalizeIntensity(nonzero=True, channel_wise=True),
    trans.EnsureType(data_type='tensor', dtype=torch.float16),
])

shape_transforms = trans.Compose([
    trans.AddChannel(),
    trans.CropForeground(),
    trans.SpatialPad((256, 256, 256), mode='constant'),
    trans.CenterSpatialCrop((256, 256, 256)),
    trans.SqueezeDim(dim=0),
])