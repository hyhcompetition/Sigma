import cv2
import torch
import numpy as np
from torch.utils import data
import random
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize, normalize_tif
import random
import numpy as np
import random
from scipy.ndimage import label
from skimage.measure import regionprops

def random_mirror(rgb, gt, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)

    return rgb, gt, modal_x

def random_patch_shuffle(rgb, gt, modal_x):
    if random.random() > 0.2:
        return rgb, gt, modal_x

    # 获取图像尺寸
    h, w = rgb.shape[:2]

    # 定义每个块的尺寸，这里是64x64
    patch_size = 64

    # 检查图像尺寸是否能被patch_size整除
    assert h % patch_size == 0 and w % patch_size == 0, "图像尺寸必须是64的倍数"

    # 计算每个维度上的块数
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size

    # 生成块的索引
    idxs = [(i, j) for i in range(num_patches_h) for j in range(num_patches_w)]
    random.shuffle(idxs)

    # 创建一个空的数组来存放新的图像
    new_rgb = np.zeros_like(rgb)
    new_gt = np.zeros_like(gt)
    new_modal_x = np.zeros_like(modal_x)

    # 使用索引重排块
    idx = 0
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size

            src_i, src_j = idxs[idx]
            src_y_start = src_i * patch_size
            src_y_end = src_y_start + patch_size
            src_x_start = src_j * patch_size
            src_x_end = src_x_start + patch_size

            new_rgb[y_start:y_end, x_start:x_end] = rgb[src_y_start:src_y_end, src_x_start:src_x_end]
            new_gt[y_start:y_end, x_start:x_end] = gt[src_y_start:src_y_end, src_x_start:src_x_end]
            new_modal_x[y_start:y_end, x_start:x_end] = modal_x[src_y_start:src_y_end, src_x_start:src_x_end]
            idx += 1

    return new_rgb, new_gt, new_modal_x

def random_copy_paste(rgb, gt, modal_x, class_indices):
    """
    对单个图像对进行随机复制-粘贴数据增强。

    参数：
    - rgb: RGB图像，形状为 (H, W, C) 的 numpy 数组。
    - gt: 掩码图像，形状为 (H, W) 的 numpy 数组，像素值为类别索引。
    - modal_x: 另一种模态的图像，形状为 (H, W, C') 的 numpy 数组。
    - class_indices: 需要增强的类别索引列表。

    返回：
    - rgb_aug: 增强后的 RGB 图像。
    - gt_aug: 增强后的掩码图像。
    - modal_x_aug: 增强后的模态图像。
    """
    # 随机决定是否进行增强
    if random.random() > 0.2:
        return rgb, gt, modal_x

    # 创建指定类别的掩码
    class_mask = np.isin(gt, class_indices).astype(np.uint8)
    # 标记连通区域
    labeled_mask, num_features = label(class_mask)
    regions = regionprops(labeled_mask)

    # 如果没有找到区域，返回原图像
    if num_features == 0:
        return rgb, gt, modal_x

    # 随机选择一些区域进行复制
    num_regions_to_copy = random.randint(len(regions)//2, len(regions)) 
    regions_to_copy = random.sample(regions, num_regions_to_copy)

    # 复制原始图像，用于修改
    rgb_aug = rgb.copy()
    

    gt_aug = gt.copy()
    modal_x_aug = modal_x.copy()

    h, w = gt.shape

    for region in regions_to_copy:
        # 获取区域的边界框
        min_row, min_col, max_row, max_col = region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col

        region_slice = (slice(min_row, max_row), slice(min_col, max_col))

        # 提取区域的图像和掩码
        object_rgb = rgb[region_slice]
        object_gt = gt[region_slice]
        object_modal_x = modal_x[region_slice]

        object_mask = (labeled_mask[region_slice] == region.label).astype(np.uint8)

        # 在图像内随机选择粘贴位置
        paste_row = random.randint(0, h - region_height)
        paste_col = random.randint(0, w - region_width)
        paste_slice = (slice(paste_row, paste_row + region_height), slice(paste_col, paste_col + region_width))

        # 检查粘贴区域是否与已有对象重叠
        existing_gt = gt_aug[paste_slice]
        existing_mask = (existing_gt != 0).astype(np.uint8)
        overlap = existing_mask * object_mask

        if np.any(overlap):
            continue  # 如果有重叠，跳过这次粘贴

        # 进行粘贴
        # 对 RGB 图像
        for c in range(rgb.shape[2]):
            channel = rgb_aug[paste_slice][:, :, c]
            channel[object_mask != 0] = object_rgb[:, :, c][object_mask != 0]
            rgb_aug[paste_slice][:, :, c] = channel

        # 对 modal_x 图像
        for c in range(modal_x.shape[2]):
            channel = modal_x_aug[paste_slice][:, :, c]
            channel[object_mask != 0] = object_modal_x[:, :, c][object_mask != 0]
            modal_x_aug[paste_slice][:, :, c] = channel

        # 对 gt 掩码
        gt_channel = gt_aug[paste_slice]
        gt_channel[object_mask != 0] = object_gt[object_mask != 0]
        gt_aug[paste_slice] = gt_channel

    return rgb_aug, gt_aug, modal_x_aug

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
        rgb, gt, modal_x = random_copy_paste(rgb, gt, modal_x, [2,6,8,5])
        rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)
        rgb, gt, modal_x = random_patch_shuffle(rgb, gt, modal_x)
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