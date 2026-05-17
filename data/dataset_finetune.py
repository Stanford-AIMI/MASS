"""Downstream segmentation finetuning dataset.

This lightweight dataset is used by the release finetuning example. It loads
processed image/label arrays, applies training-time augmentation, and returns
supervised segmentation batches for adapting a MASS checkpoint.
"""

import os
import json
import logging
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from utils.registry import register_dataset
from . import augmentation

@register_dataset("FineTuningDataset")
class FineTuningDataset(Dataset):
    """
    Simplified dataset for finetuning pretrained models on downstream tasks.
    """
    
    def __init__(
        self,
        data_root: str,
        datasets: List[str],
        mode: str = 'train',
        training_size: Union[List[int], Tuple[int, int, int]] = (128, 128, 128),
        augmentation_config: Optional[Dict] = None,
        aug_device: str = "cpu",
        foreground_classes: Optional[List[int]] = None,  # Explicit list of foreground class labels
        spacing: Optional[Union[List[float], Tuple[float, float, float]]] = None,
    ):
        """
        Args:
            data_root: Root directory for data
            datasets: List of dataset names
            mode: 'train' or 'test'
            training_size: Crop size for training
            augmentation_config: Augmentation settings
            aug_device: Device for augmentation
            foreground_classes: List of foreground class labels (e.g., [1,2,3,...,13] for BCV)
            spacing: Voxel spacing in array axis order (z, y, x), matching [D, H, W]
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.datasets = datasets
        self.mode = mode
        self.training_size = training_size
        self.augmentation_config = augmentation_config or {}
        self.aug_device = aug_device
        # Training/evaluation arrays are stored as [D, H, W], so spacing is (z, y, x).
        self.spacing = tuple(float(x) for x in (spacing or [1.5, 1.5, 1.5]))
        
        from .split import dataset_lab_map, ft_train_test_split
        self.train_test_split = ft_train_test_split
        
        if foreground_classes is not None:
            self.foreground_classes = sorted(foreground_classes)
        else:
            # Auto-detect local foreground label ids from the dataset metadata.
            all_classes = set()
            for dataset in self.datasets:
                if dataset in dataset_lab_map:
                    num_classes = len(dataset_lab_map[dataset])
                    for idx in range(0, num_classes):
                        all_classes.add(idx + 1)
            self.foreground_classes = sorted(list(all_classes))

        self.num_classes = len(self.foreground_classes)
        logging.info(f"Finetuning on {self.num_classes} foreground classes: {self.foreground_classes}")
        
        self._prepare_file_list()
        
        logging.info(f"[FineTuningDataset] mode={mode} - {len(self.file_list)} samples, {self.num_classes} classes")
    
    def _prepare_file_list(self):
        """Load file list based on mode and datasets."""
        self.file_list = []
        
        for dataset in self.datasets:
            dataset_dir = self.data_root / dataset
            
            logging.info(f"Looking for dataset in: {dataset_dir}")
            
            if not dataset_dir.exists():
                logging.error(f"Dataset directory not found: {dataset_dir}")
                continue
            
            split_key = f"{dataset}_{self.mode}"
            if split_key in self.train_test_split:
                split_names = self.train_test_split[split_key]
                logging.info(f"Found {len(split_names)} samples in split '{split_key}'")
            else:
                # If no split defined, skip this dataset
                logging.warning(f"No split found for {split_key}")
                continue
            
            files_found = 0
            for name in split_names:
                img_name = str(name)
                
                img_npy_path = dataset_dir / f"{img_name}_image.npy"
                gt_npy_path = dataset_dir / f"{img_name}_gt.npy"
                
                if img_npy_path.exists() and gt_npy_path.exists():
                    files_found += 1
                    self.file_list.append({
                        'img_npy_path': img_npy_path,
                        'gt_npy_path': gt_npy_path,
                        'img_name': img_name,
                        'dataset': dataset,
                        'sample_id': img_name,
                        'spacing': self.spacing,
                    })
                else:
                    if not img_npy_path.exists():
                        logging.warning(f"Image .npy file not found: {img_npy_path}")
                    if not gt_npy_path.exists():
                        logging.warning(f"GT .npy file not found: {gt_npy_path}")
            
            logging.info(f"Found {files_found} valid file pairs")
    
    def __len__(self):
        # Keep dataset length factual. Few-shot finetuning can request more
        # optimization steps per epoch through trainer-level max_iter_per_epoch.
        return len(self.file_list)
        
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - query_img: [1, D, H, W] query image
                - query_lab: [num_classes, D, H, W] ground truth labels (all classes)
                - class_indices: [num_classes] indices for all classes (0-based)
                - dataset_name: str
        """
        idx = idx % len(self.file_list)
        file_info = self.file_list[idx]
        
        img = np.load(file_info['img_npy_path'], mmap_mode='r').astype(np.float32)  # [D, H, W]
        lab = np.load(file_info['gt_npy_path'], mmap_mode='r').astype(np.float32)   # [D, H, W]
        
        if self.mode == 'train':
            # Random crop with margin for augmentation
            d, h, w = self.training_size
            crop_size = [d + 20, h + 20, w + 20]  # Add margin for augmentation
            
            # Random crop using existing augmentation function
            img, lab = augmentation.np_crop_3d(img, lab, crop_size, mode="random")
            
            img = torch.from_numpy(img).unsqueeze(0).float()  # [1, D, H, W]
            lab = torch.from_numpy(lab)                       # [D, H, W]
            
            query_labs = []
            for class_idx, class_label in enumerate(self.foreground_classes):
                class_mask = (lab == class_label).float().unsqueeze(0)  # [1, D, H, W]
                query_labs.append(class_mask)
            
            query_lab = torch.cat(query_labs, dim=0)  # [num_classes, D, H, W]
            
            if self.augmentation_config:
                img, query_lab = self._apply_augmentation(img, query_lab)
                
                # Final center crop to exact training size
                img_batch = img.unsqueeze(0)         # [1, 1, D, H, W]
                mask_batch = query_lab.unsqueeze(0)  # [1, num_classes, D, H, W]
                img_batch, mask_batch = augmentation.crop_3d(img_batch, mask_batch, self.training_size, mode="center")
                img = img_batch.squeeze(0)           # [1, D, H, W]
                query_lab = mask_batch.squeeze(0)    # [num_classes, D, H, W]
            else:
                # If no augmentation config, just center crop to training size
                img_batch = img.unsqueeze(0)         # [1, 1, D, H, W]
                mask_batch = query_lab.unsqueeze(0)  # [1, num_classes, D, H, W]
                img_batch, mask_batch = augmentation.crop_3d(img_batch, mask_batch, self.training_size, mode="center")
                img = img_batch.squeeze(0)
                query_lab = mask_batch.squeeze(0)
        
        else:  # Testing mode
            img = torch.from_numpy(img).unsqueeze(0).float()  # [1, D, H, W]
            lab = torch.from_numpy(lab)                       # [D, H, W]
            
            query_labs = []
            for class_idx, class_label in enumerate(self.foreground_classes):
                class_mask = (lab == class_label).float().unsqueeze(0)  # [1, D, H, W]
                query_labs.append(class_mask)
            
            query_lab = torch.cat(query_labs, dim=0)  # [num_classes, D, H, W]
        
        # These are positions in foreground_classes; raw label ids are stored in
        # foreground_classes itself.
        class_indices = torch.arange(self.num_classes, dtype=torch.long)  # [num_classes]
        
        return {
            'query_img': img,
            'query_lab': query_lab,
            'class_indices': class_indices,
            'dataset_name': file_info['dataset'],
            'sample_id': file_info['sample_id'],
            'spacing': file_info['spacing'],
        }

    def _apply_augmentation(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentation to image and mask tensors.
        
        Args:
            img: Image tensor [1, D, H, W]
            mask: Mask tensor [num_classes, D, H, W]
            
        Returns:
            Tuple of (augmented image, augmented mask)
        """
        # Expand to 5D for augmentation functions [B, C, D, H, W]
        img = img.unsqueeze(0)   # [1, 1, D, H, W]
        mask = mask.unsqueeze(0) # [1, num_classes, D, H, W]
        
        if self.aug_device == "gpu" and torch.cuda.is_available():
            img = img.cuda()
            mask = mask.cuda()
        
        affine_prob = self.augmentation_config.get('affine_prob', 0.3)
        scale = self.augmentation_config.get('scale', [0.1, 0.1, 0.1])
        rotate = self.augmentation_config.get('rotate', [10, 10, 10])
        translate = self.augmentation_config.get('translate', [0, 0, 0])
        shear = self.augmentation_config.get('shear', [0.0, 0.0, 0.0])
        
        if random.random() < affine_prob:
            img, mask = augmentation.random_scale_rotate_translate_3d(
                img, mask,
                scale=scale,
                rotate=rotate,
                translate=translate,
                shear=shear,
            )
        
        brightness_range = self.augmentation_config.get('brightness_range', [0.95, 1.05])
        gamma_range = self.augmentation_config.get('gamma_range', [0.95, 1.05])
        contrast_range = self.augmentation_config.get('contrast_range', [0.95, 1.05])
        blur_range = self.augmentation_config.get('blur_range', [0.9, 1.1])
        gaussian_noise_std = self.augmentation_config.get('gaussian_noise_std', 0.01)
        
        if random.random() < 0.2:
            img = augmentation.brightness_multiply(img, brightness_range)
        if random.random() < 0.2:
            img = augmentation.gamma(img, gamma_range)
        if random.random() < 0.2:
            img = augmentation.contrast(img, contrast_range)
        if random.random() < 0.2:
            img = augmentation.gaussian_blur(img, blur_range)
        if random.random() < 0.2:
            img = augmentation.gaussian_noise(img, std=gaussian_noise_std)
        
        if img.is_cuda:
            img = img.cpu()
            mask = mask.cpu()
        
        return img.squeeze(0).float(), mask.squeeze(0).long()  # [1, D, H, W], [num_classes, D, H, W]

    def get_all_samples_for_class(self, class_idx: int):
        """
        Get all samples containing a specific class.
        
        Args:
            class_idx: Index into self.foreground_classes (0-based)
            
        Returns:
            List of file info dictionaries
        """
        if class_idx >= len(self.foreground_classes):
            return []
        
        class_label = self.foreground_classes[class_idx]
        samples = []
        
        for file_info in self.file_list:
            # Quick check: load label from .npy to see if this class is present
            lab = np.load(file_info['gt_npy_path'], mmap_mode='r')
            if np.any(lab == class_label):
                samples.append(file_info)
        
        return samples
