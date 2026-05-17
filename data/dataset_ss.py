"""Mask-guided self-supervised pretraining dataset.

This dataset reads preprocessed image arrays and compressed auto masks from
``dataset.h5``. For each training sample it draws class-agnostic masks, builds
reference/target crops, applies paired augmentations, and returns the tensors
used by the MASS self-supervised objective.
"""

import os
import random
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import h5py

from utils.registry import register_dataset
from . import augmentation
from .split import (
    train_test_split,
    dataset_lab_map,
    dataset_weight,
)

@register_dataset("MaskGuidedSelfSupervisedDataset")
class MaskGuidedSelfSupervisedDataset(Dataset):
    """
    Dataset for self-supervised training using auto-generated masks without class labels.
    Generates two augmented views of the same image - one as reference, one as target.
    Only for training.

    Args:
        datasets: List of dataset names to include
        data_root: Root directory for data
        training_size: Crop size for training
        aug_device: Device for augmentation ('cpu' or 'gpu')
        max_in_batch: Maximum number of classes to process in a single batch
        augmentation_config: Dictionary of augmentation parameters
    """

    def __init__(
        self,
        data_root: str,
        datasets: List[str] = ["bcv"],
        training_size: Union[List[int], Tuple[int, int, int]] = (128, 128, 128),
        aug_device: str = "cpu",
        max_in_batch: int = 10,
        augmentation_config: Optional[Dict] = None,
        device: str = "cpu",
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.datasets = datasets
        self.training_size = training_size
        self.augmentation_config = augmentation_config or {}
        self.aug_device = aug_device
        self.max_in_batch = max_in_batch
        self.device = device
        self.dataset_lab_map = dataset_lab_map
        self.dataset_weight = dataset_weight

        self._prepare_file_list()
        self.weight_list = self.get_weight_list()

        self._setup_augmentation()

        logging.info(f"[MaskGuidedSelfSupervisedDataset] mode=Train - {len(self.file_list)} samples loaded")



    def _prepare_file_list(self):
        """ Builds self.file_list: a list of dicts with img/lab paths and metadata. """
        self.file_list = []

        for dataset in self.datasets:
            dataset_dir = self.data_root / dataset
            h5_file_path = dataset_dir / "dataset.h5"

            if not dataset_dir.exists():
                logging.warning(f"Dataset directory {dataset_dir} not found")
                continue

            if not h5_file_path.exists():
                logging.warning(f"HDF5 file {h5_file_path} not found")
                continue

            try:
                with h5py.File(h5_file_path, 'r') as h5f:
                    available_groups = set(h5f.keys())
            except Exception as e:
                logging.warning(f"Could not read HDF5 file {h5_file_path}: {e}")
                continue

            split_key = f"{dataset}_train"
            if split_key in train_test_split:
                file_names = train_test_split[split_key]
            else:
                file_names = sorted(available_groups)
                logging.warning(
                    f"Split key {split_key} not found in train_test_split; "
                    f"using all {len(file_names)} groups from {h5_file_path}"
                )

            for name in file_names:
                img_name = str(name)

                img_npy_path = dataset_dir / f"{img_name}_image.npy"

                if img_npy_path.exists() and img_name in available_groups:
                    try:
                        with h5py.File(h5_file_path, 'r') as h5f:
                            group = h5f[img_name]
                            has_auto_masks = 'auto_masks' in group

                            if has_auto_masks:
                                auto_masks = group['auto_masks']
                                if auto_masks.shape[0] == 0:
                                    logging.warning(f"Skipping {img_name}: auto_masks is empty")
                                    continue

                                self.file_list.append({
                                    "h5_file": h5_file_path,
                                    "img_npy_path": img_npy_path,
                                    "img_name": img_name,
                                    "weight": self.dataset_weight.get(dataset, 1.0),
                                    "dataset_idx": self.datasets.index(dataset),
                                    "ds_name": dataset,
                                })
                            else:
                                logging.warning(f"Auto masks not found in HDF5: {img_name}/auto_masks")
                    except Exception as e:
                        logging.warning(f"Error checking HDF5 group {img_name}: {e}")
                else:
                    if not img_npy_path.exists():
                        logging.warning(f"Image .npy file not found: {img_npy_path}")
                    if img_name not in available_groups:
                        logging.warning(f"Group not found in HDF5: {img_name}")


    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.file_list) * 10000

    def get_weight_list(self) -> np.ndarray:
        """Return one sampling weight per scan, using dataset-level weights."""
        weights = np.asarray([info.get("weight", 1.0) for info in self.file_list], dtype=np.float64)
        if weights.size == 0:
            return weights

        invalid = ~np.isfinite(weights) | (weights <= 0)
        if invalid.any():
            logging.warning("Found non-positive or invalid dataset weights; replacing them with 1.0")
            weights[invalid] = 1.0
        return weights

    def _setup_augmentation(self):
        """Setup augmentation parameters for weak and strong augmentation."""
        # Weak augmentation parameters
        self.weak_aug_config = {
            'affine_prob': self.augmentation_config.get('weak_affine_prob', 0.5),
            'scale': self.augmentation_config.get('weak_scale', [0.3, 0.3, 0.3]),
            'rotate': self.augmentation_config.get('weak_rotate', [30, 30, 30]),
            'translate': self.augmentation_config.get('weak_translate', [0, 0, 0]),
            'shear': self.augmentation_config.get('weak_shear', [0.0, 0.0, 0.0]),
            'brightness_range': self.augmentation_config.get('weak_brightness_range', [0.9, 1.1]),
            'brightness_additive_std': self.augmentation_config.get('weak_brightness_additive_std', 0.1),
            'gamma_range': self.augmentation_config.get('weak_gamma_range', [0.8, 1.2]),
            'contrast_range': self.augmentation_config.get('weak_contrast_range', [0.8, 1.2]),
            'blur_range': self.augmentation_config.get('weak_blur_range', [0.8, 1.2]),
            'gaussian_noise_std': self.augmentation_config.get('weak_gaussian_noise_std', 0.02),
            'aug_prob': self.augmentation_config.get('weak_aug_prob', 0.2)
        }

        # Strong augmentation parameters
        self.strong_aug_config = {
            'affine_prob': self.augmentation_config.get('strong_affine_prob', 0.8),
            'scale': self.augmentation_config.get('strong_scale', [0.3, 0.3, 0.3]),
            'rotate': self.augmentation_config.get('strong_rotate', [30, 30, 30]),
            'translate': self.augmentation_config.get('strong_translate', [0.0, 0.0, 0.0]),
            'shear': self.augmentation_config.get('strong_shear', [0.1, 0.1, 0.1]),
            'brightness_range': self.augmentation_config.get('strong_brightness_range', [0.8, 1.3]),
            'brightness_additive_std': self.augmentation_config.get('strong_brightness_additive_std', 0.15),
            'gamma_range': self.augmentation_config.get('strong_gamma_range', [0.8, 1.3]),
            'contrast_range': self.augmentation_config.get('strong_contrast_range', [0.8, 1.3]),
            'blur_range': self.augmentation_config.get('strong_blur_range', [0.7, 1.5]),
            'gaussian_noise_std': self.augmentation_config.get('strong_gaussian_noise_std', 0.04),
            'aug_prob': self.augmentation_config.get('strong_aug_prob', 0.2)
        }

    def _apply_augmentation(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        aug_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentation to image and mask.

        Args:
            image: Input image tensor with shape [D, H, W]
            mask: Input mask tensor with shape [N, D, H, W]
            aug_type: Type of augmentation ('weak' or 'strong')

        Returns:
            Tuple of augmented (image, mask) tensors
        """
        if aug_type == "weak":
            config = self.weak_aug_config
        elif aug_type == "strong":
            config = self.strong_aug_config
        else:
            return image, mask

        img = image.unsqueeze(0).unsqueeze(0).float()
        mask_tensor = mask.unsqueeze(0).float()

        if self.aug_device == "gpu" and torch.cuda.is_available():
            img = img.to(self.device)
            mask_tensor = mask_tensor.to(self.device)

        if random.random() < config['affine_prob']:
            img, mask_tensor = augmentation.random_scale_rotate_translate_3d(
                img, mask_tensor,
                scale=config['scale'],
                rotate=config['rotate'],
                translate=config['translate'],
                shear=config['shear'],
            )
            img, mask_tensor = augmentation.crop_3d(img, mask_tensor, self.training_size, mode="center")
        else:
            img, mask_tensor = augmentation.crop_3d(img, mask_tensor, self.training_size, mode="random")

        aug_prob = config['aug_prob']

        if random.random() < aug_prob:
            img = augmentation.brightness_multiply(img, config['brightness_range'])

        if random.random() < aug_prob:
            img = augmentation.brightness_additive(img, std=config['brightness_additive_std'])

        if random.random() < aug_prob:
            img = augmentation.gamma(img, config['gamma_range'])

        if random.random() < aug_prob:
            img = augmentation.contrast(img, config['contrast_range'])

        if random.random() < aug_prob:
            img = augmentation.gaussian_blur(img, config['blur_range'])

        if random.random() < aug_prob:
            img = augmentation.gaussian_noise(img, std=config['gaussian_noise_std'])

        if img.is_cuda:
            img = img.cpu()
            mask_tensor = mask_tensor.cpu()

        image_out = img.squeeze(0).squeeze(0).contiguous().clone()
        mask_out = mask_tensor.squeeze(0).contiguous().clone()

        return image_out, mask_out

    def _load_and_crop(self, file_info: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image and mask from HDF5 file.
        Then generate two cropped views of the image and mask.

        Args:
            file_info: Dictionary containing file paths

        Returns:
            Tuple of (img1, mask1, img2, mask2) tensors
        """

        # Images are kept as .npy for fast per-iteration reads; only the many
        # auto masks are stored compressed in HDF5.
        image = np.load(file_info["img_npy_path"], mmap_mode='r')

        with h5py.File(file_info["h5_file"], 'r') as h5f:
            group = h5f[file_info["img_name"]]


            mask_dataset = group['auto_masks']

            num_channels = mask_dataset.shape[0]

            # Each selected auto-mask channel becomes one in-context task slot.
            num_to_select = min(self.max_in_batch, num_channels)
            selected_indices = np.random.choice(num_channels, size=num_to_select, replace=False)
            selected_indices = np.sort(selected_indices)

            d, h, w = self.training_size
            crop_size = [d + 40, h + 40, w + 40]

            img_d, img_h, img_w = image.shape

            crop_size = [
                min(crop_size[0], img_d),
                min(crop_size[1], img_h),
                min(crop_size[2], img_w)
            ]

            z_min = crop_size[0] // 2
            y_min = crop_size[1] // 2
            x_min = crop_size[2] // 2
            z_max = img_d - crop_size[0] // 2
            y_max = img_h - crop_size[1] // 2
            x_max = img_w - crop_size[2] // 2

            # Sample one anchor point, then make two overlapping crops around it.
            z = np.random.randint(z_min, max(z_max, z_min + 1))
            y = np.random.randint(y_min, max(y_max, y_min + 1))
            x = np.random.randint(x_min, max(x_max, x_min + 1))
            coordinate = [z, y, x]

            overlap_ratio = self.augmentation_config.get('overlap_ratio', 0.67)
            img1, mask1, img2, mask2 = augmentation.np_crop_around_coordinate_two_views_with_mask_slicing_3d(
                np_img=image,
                np_lab=mask_dataset,
                crop_size=crop_size,
                coordinate=coordinate,
                overlap_ratio=overlap_ratio,
                selected_indices=selected_indices,
                mode='random'
            )

        img1 = torch.from_numpy(img1).float().contiguous().clone()
        img2 = torch.from_numpy(img2).float().contiguous().clone()
        mask1 = torch.from_numpy(mask1).float().contiguous().clone()
        mask2 = torch.from_numpy(mask2).float().contiguous().clone()

        ones_mask_prob = getattr(self, 'ones_mask_prob', 1) 
        
        if random.random() < ones_mask_prob:
            # An all-foreground channel teaches the model a global-context task
            # in addition to object-shaped auto masks.
            channel_idx = random.randint(0, mask1.shape[0] - 1)
            mask1[channel_idx] = 1.0
            mask2[channel_idx] = 1.0

        return img1, mask1, img2, mask2

    def _load_and_crop_merge(self, file_info: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image and mask from HDF5 file.
        Then generate two cropped views of the image and mask.

        This optional variant merges a small number of auto masks into
        composite pseudo-objects; the default training path uses _load_and_crop.

        Args:
            file_info: Dictionary containing file paths

        Returns:
            Tuple of (img1, mask1, img2, mask2) tensors
        """
        image = np.load(file_info["img_npy_path"], mmap_mode='r')

        with h5py.File(file_info["h5_file"], 'r') as h5f:
            group = h5f[file_info["img_name"]]

            mask_dataset = group['auto_masks']

            num_channels = mask_dataset.shape[0]

            num_to_select = min(self.max_in_batch, num_channels)

            # A small fraction of samples merge masks to mimic composite regions.
            merge_masks = random.random() < 0.05

            if merge_masks and num_channels >= num_to_select + 2:
                num_merged = random.randint(1, max(1, num_to_select // 2))
                num_single = num_to_select - num_merged

                min_indices_for_merged = num_merged * 2
                min_total_indices = num_single + min_indices_for_merged

                if min_total_indices <= num_channels:
                    indices_per_merged = []
                    remaining_indices = num_channels - num_single

                    for i in range(num_merged):
                        max_for_this = min(5, remaining_indices - (num_merged - i - 1) * 2)
                        if max_for_this < 2:
                            indices_per_merged.append(2)
                        else:
                            indices_per_merged.append(random.randint(2, max_for_this))
                        remaining_indices -= indices_per_merged[-1]

                    total_indices = num_single + sum(indices_per_merged)

                    all_selected = np.random.choice(num_channels, size=total_indices, replace=False)
                    selected_indices = all_selected

                    merge_groups = []
                    idx = 0

                    for _ in range(num_single):
                        merge_groups.append([idx])
                        idx += 1

                    for merge_count in indices_per_merged:
                        merge_groups.append(list(range(idx, idx + merge_count)))
                        idx += merge_count

                    assert len(merge_groups) == num_to_select, f"Expected {num_to_select} groups, got {len(merge_groups)}"
                else:
                    selected_indices = np.random.choice(num_channels, size=num_to_select, replace=False)
                    selected_indices = np.sort(selected_indices)
                    merge_groups = None
            else:
                selected_indices = np.random.choice(num_channels, size=num_to_select, replace=False)
                selected_indices = np.sort(selected_indices)
                merge_groups = None

            d, h, w = self.training_size
            crop_size = [d + 40, h + 40, w + 40]

            img_d, img_h, img_w = image.shape

            crop_size = [
                min(crop_size[0], img_d),
                min(crop_size[1], img_h),
                min(crop_size[2], img_w)
            ]

            z_min = crop_size[0] // 2
            y_min = crop_size[1] // 2
            x_min = crop_size[2] // 2
            z_max = img_d - crop_size[0] // 2
            y_max = img_h - crop_size[1] // 2
            x_max = img_w - crop_size[2] // 2

            z = np.random.randint(z_min, max(z_max, z_min + 1))
            y = np.random.randint(y_min, max(y_max, y_min + 1))
            x = np.random.randint(x_min, max(x_max, x_min + 1))
            coordinate = [z, y, x]

            overlap_ratio = self.augmentation_config.get('overlap_ratio', 0.67)
            img1, mask1, img2, mask2 = augmentation.np_crop_around_coordinate_two_views_with_mask_slicing_3d(
                np_img=image,
                np_lab=mask_dataset,
                crop_size=crop_size,
                coordinate=coordinate,
                overlap_ratio=overlap_ratio,
                selected_indices=selected_indices,
                mode='random'
            )

            if merge_groups is not None:
                final_mask1 = []
                final_mask2 = []

                for group in merge_groups:
                    assert len(group) > 0, "Empty merge group encountered"

                    if len(group) == 1:
                        final_mask1.append(mask1[group[0]])
                        final_mask2.append(mask2[group[0]])
                    else:
                        merged1 = mask1[group[0]].astype(bool)
                        merged2 = mask2[group[0]].astype(bool)
                        for i in group[1:]:
                            merged1 = np.logical_or(merged1, mask1[i])
                            merged2 = np.logical_or(merged2, mask2[i])
                        final_mask1.append(merged1.astype(np.float32))
                        final_mask2.append(merged2.astype(np.float32))

                mask1 = np.stack(final_mask1, axis=0)
                mask2 = np.stack(final_mask2, axis=0)

                assert mask1.shape[0] == num_to_select, f"Expected {num_to_select} masks, got {mask1.shape[0]}"
                assert mask2.shape[0] == num_to_select, f"Expected {num_to_select} masks, got {mask2.shape[0]}"

        img1 = torch.from_numpy(img1).float().contiguous().clone()
        img2 = torch.from_numpy(img2).float().contiguous().clone()
        mask1 = torch.from_numpy(mask1).float().contiguous().clone()
        mask2 = torch.from_numpy(mask2).float().contiguous().clone()

        return img1, mask1, img2, mask2

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get dataset item at index idx.

        Returns:
            Tuple of (tgt_img, tgt_mask, ref_img, ref_mask)
        """
        idx = idx % len(self.file_list)
        file_info = self.file_list[idx]

        img1, mask1, img2, mask2 = self._load_and_crop(file_info)

        # Reference crops define the task with weaker perturbations, while target
        # crops receive stronger perturbations for the prediction objective.
        ref_img, ref_mask = self._apply_augmentation(img1, mask1, "weak")
        tgt_img, tgt_mask = self._apply_augmentation(img2, mask2, "strong")

        ref_mask = (ref_mask > 0).float()
        tgt_mask = (tgt_mask > 0).float()

        ref_img = ref_img.unsqueeze(0)
        tgt_img = tgt_img.unsqueeze(0)

        return (
            tgt_img,
            tgt_mask,
            ref_img,
            ref_mask
        )
