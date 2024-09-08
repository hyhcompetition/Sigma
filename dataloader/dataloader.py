import cv2
import torch
import numpy as np
from torch.utils import data
import random
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize, normalize_tif
import random
def random_mirror(rgb, gt, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)

    return rgb, gt, modal_x
def copy_paste(rgb, gt, modal_x, target_classes, max_shift=50):
    if random.random() < 0.3:
        return rgb, gt, modal_x
    # Find the indices where the target class is located in the gt mask
    
    target_class = random.choice(target_classes)
    target_indices = np.where(gt == target_class)
    
    # If the target class does not exist in the mask, return the original images
    if len(target_indices[0]) == 0:
        return rgb, gt, modal_x

    # Get the bounding box of the target region
    y_min, y_max = target_indices[0].min(), target_indices[0].max()
    x_min, x_max = target_indices[1].min(), target_indices[1].max()
    
    # Extract the region from rgb and modal_x corresponding to the target class
    rgb_region = rgb[y_min:y_max + 1, x_min:x_max + 1, :]
    modal_x_region = modal_x[y_min:y_max + 1, x_min:x_max + 1, :]
    gt_region = gt[y_min:y_max + 1, x_min:x_max + 1]

    # Get the height and width of the extracted region
    region_height, region_width = rgb_region.shape[:2]
    
    # Calculate a random shift to paste the region at a new location
    shift_y = random.randint(-max_shift, max_shift)
    shift_x = random.randint(-max_shift, max_shift)
    
    # Calculate the new location to paste the region
    new_y_min = max(0, y_min + shift_y)
    new_x_min = max(0, x_min + shift_x)
    new_y_max = min(rgb.shape[0], new_y_min + region_height)
    new_x_max = min(rgb.shape[1], new_x_min + region_width)
    
    # Adjust the region size if it goes beyond the image boundaries
    new_y_min = new_y_max - region_height
    new_x_min = new_x_max - region_width

    # Paste the region onto the new location in rgb and modal_x
    rgb[new_y_min:new_y_max, new_x_min:new_x_max, :] = rgb_region
    modal_x[new_y_min:new_y_max, new_x_min:new_x_max, :] = modal_x_region
    gt[new_y_min:new_y_max, new_x_min:new_x_max] = gt_region
    
    return rgb, gt, modal_x

def random_rotate(rgb, gt, modal_x, angle_range=(-180, 180)):
    if random.random() > 0.5:
        return rgb, gt, modal_x
    # Randomly choose an angle within the specified range
    angle = random.uniform(angle_range[0], angle_range[1])
    
    # Get the image dimensions (assuming rgb, gt, and modal_x are of the same height and width)
    h, w = rgb.shape[:2]
    
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    
    # Initialize rotated images with the same shape as the input
    rgb_rotated = np.zeros_like(rgb)
    modal_x_rotated = np.zeros_like(modal_x)
    
    # Rotate each channel of the rgb image
    for c in range(rgb.shape[2]):
        rgb_rotated[:, :, c] = cv2.warpAffine(rgb[:, :, c], rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    
    # Rotate the gt image (assuming it is 2D)
    gt_rotated = cv2.warpAffine(gt, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
    
    # Rotate each channel of the modal_x image
    for c in range(modal_x.shape[2]):
        modal_x_rotated[:, :, c] = cv2.warpAffine(modal_x[:, :, c], rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    
    return rgb_rotated, gt_rotated, modal_x_rotated

def random_scale(rgb, gt, modal_x, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return rgb, gt, modal_x, scale

class TrainPre(object):
    def __init__(self, norm_mean, norm_std, config):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.config = config

    def __call__(self, rgb, gt, modal_x):
        rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)
        if self.config.train_scale_array is not None:
            rgb, gt, modal_x, scale = random_scale(rgb, gt, modal_x, self.config.train_scale_array)

        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        modal_x = normalize(modal_x, self.norm_mean, self.norm_std)

        crop_size = (self.config.image_height, self.config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)

        p_rgb = p_rgb.transpose(2, 0, 1)
        p_modal_x = p_modal_x.transpose(2, 0, 1)
        
        return p_rgb, p_gt, p_modal_x
class TrainPreTif(object):
    def __init__(self, norm_mean, norm_std, config):
        self.sar_norm_mean = norm_mean[0]
        self.msi_norm_mean = norm_mean[1]
        self.sar_norm_std = norm_std[0]
        self.msi_norm_std = norm_std[1]
        self.config = config

    def __call__(self, rgb, gt, modal_x):
        # rgb, gt, modal_x = copy_paste(rgb, gt, modal_x, [2,5,9])
        rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)
        rgb, gt, modal_x = random_rotate(rgb, gt, modal_x)
        
        
        rgb = normalize_tif(rgb, self.msi_norm_mean, self.msi_norm_std)
        modal_x = normalize_tif(modal_x, self.sar_norm_mean, self.sar_norm_std)

        rgb = rgb.transpose(2, 0, 1)
        modal_x = modal_x.transpose(2, 0, 1)
        
        return rgb, gt, modal_x
class ValPre(object):
    def __call__(self, rgb, gt, modal_x):
        return rgb, gt, modal_x

def get_train_loader(engine, dataset, config):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    train_preprocess = TrainPreTif([config.sar_norm_mean, config.msi_norm_mean], [config.sar_norm_std, config.msi_norm_std], config)

    train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler