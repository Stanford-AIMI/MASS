"""Downstream 3D medical image classification dataset.

This release example loader reads preprocessed image arrays and metadata labels
for single-label or multi-label classification tasks using the MASS encoder.
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import random

from utils.registry import register_dataset
from data import augmentation

@register_dataset("classification")
class ClassificationDataset(Dataset):
    """
    Generic dataset for 3D medical image classification tasks.

    Supports:
    - Single-label and multi-label classification
    - Multiple datasets with different class distributions
    - Automatic class weight computation for imbalanced data
    - Flexible label format (numbered or named columns)
    - Robust loading with retry mechanism

    Directory structure:
        data_root/
            dataset1/
                sample_001.npy
                sample_002.npy
                ...
                labels.csv
            dataset2/
                sample_003.npy
                ...
                labels.csv

    CSV format examples:

    Single-label:
        filename,label,split
        sample_001,0,train
        sample_002,1,val

    Multi-label (numbered):
        filename,label_0,label_1,label_2,split
        sample_001,0,1,0,train
        sample_002,1,0,1,val

    Multi-label (named):
        filename,class_a,class_b,class_c,split
        sample_001,0,1,0,train
        sample_002,1,0,1,val

    Multi-label (both - numbered used for training):
        filename,label_0,label_1,label_2,class_a,class_b,class_c,split
        sample_001,0,1,0,0,1,0,train
    """

    def __init__(
        self,
        data_root: str,
        datasets: List[str],
        mode: str = 'train',
        training_size: Tuple[int, int, int] = (96, 96, 96),
        augmentation_config: Optional[Dict[str, Any]] = None,
        num_classes: int = 2,
        multi_label: bool = False,
        label_csv_name: str = 'labels.csv',
        use_named_labels: bool = False,
        class_names: Optional[List[str]] = None,
        image_extension: str = '.npy',
        verify_images: bool = True,
        class_weights: bool = False,
        aug_device: str = 'cpu',
        **kwargs
    ):
        """
        Initialize classification dataset.

        Args:
            data_root: Root directory containing datasets
            datasets: List of dataset names to use (can mix different tasks)
            mode: 'train', 'val', or 'test'
            training_size: Size to crop/resize images to [D, H, W]
            augmentation_config: Augmentation configuration dict
            num_classes: Number of classification classes
            multi_label: Whether this is multi-label classification
            label_csv_name: Name of the CSV file containing labels
            use_named_labels: Use named columns instead of label_0, label_1, ...
            class_names: List of class names (required if use_named_labels=True)
            image_extension: File extension for images (default: .npy)
            verify_images: Verify that image files exist during initialization
            class_weights: Compute class weights for imbalanced data
            aug_device: Device for augmentation ('cpu' or 'gpu')
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.datasets = datasets
        self.mode = mode
        self.training_size = training_size
        self.augmentation_config = augmentation_config or {}
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.label_csv_name = label_csv_name
        self.use_named_labels = use_named_labels
        self.class_names = class_names
        self.image_extension = image_extension
        self.verify_images = verify_images
        self.class_weights = class_weights
        self.aug_device = aug_device
        self.device = torch.device('cuda' if aug_device == 'gpu' and torch.cuda.is_available() else 'cpu')

        if self.use_named_labels:
            # Named labels make CSVs easier to read, but training still expects
            # a fixed class order.
            if self.class_names is None:
                raise ValueError("class_names must be provided when use_named_labels=True")
            if len(self.class_names) != self.num_classes:
                raise ValueError(
                    f"Length of class_names ({len(self.class_names)}) "
                    f"must match num_classes ({self.num_classes})"
                )

        self._load_file_list()

        if self.mode == 'train':
            self._setup_augmentation()

        if self.mode == 'train' and self.class_weights:
            # Class weights are computed only from the training split.
            self._compute_class_weights()
        else:
            self.class_weights = None

        logging.info(
            f"Initialized ClassificationDataset: "
            f"{len(self.file_list)} samples, "
            f"{self.num_classes} classes, "
            f"{'multi-label' if self.multi_label else 'single-label'}, "
            f"mode={self.mode}"
        )

    def _setup_augmentation(self):
        """Setup augmentation parameters."""
        self.aug_config = {
            'affine_prob': self.augmentation_config.get('affine_prob', 0.5),
            'scale': self.augmentation_config.get('scale', [0.3, 0.3, 0.3]),
            'rotate': self.augmentation_config.get('rotate', [30, 30, 30]),
            'translate': self.augmentation_config.get('translate', [0, 0, 0]),
            'shear': self.augmentation_config.get('shear', [0.0, 0.0, 0.0]),
            'brightness_range': self.augmentation_config.get('brightness_range', [0.9, 1.1]),
            'brightness_additive_std': self.augmentation_config.get('brightness_additive_std', 0.1),
            'gamma_range': self.augmentation_config.get('gamma_range', [0.8, 1.2]),
            'contrast_range': self.augmentation_config.get('contrast_range', [0.8, 1.2]),
            'blur_range': self.augmentation_config.get('blur_range', [0.8, 1.2]),
            'gaussian_noise_std': self.augmentation_config.get('gaussian_noise_std', 0.02),
            'aug_prob': self.augmentation_config.get('aug_prob', 0.2)
        }

    def _load_file_list(self):
        """Load file list with labels from CSV."""
        self.file_list = []

        for dataset in self.datasets:
            dataset_dir = self.data_root / dataset

            if not dataset_dir.exists():
                logging.warning(f"Dataset directory {dataset_dir} not found")
                continue

            csv_path = dataset_dir / self.label_csv_name
            if not csv_path.exists():
                logging.warning(f"Labels CSV not found: {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path)

                if 'split' in df.columns:
                    df = df[df['split'] == self.mode]

                for _, row in df.iterrows():
                    filename = row['filename']

                    if filename.endswith('.npy'):
                        img_path = dataset_dir / filename
                    else:
                        img_path = dataset_dir / f"{filename}.npy"

                    if not img_path.exists():
                        logging.warning(f"Image file not found: {img_path}")
                        continue

                    if self.multi_label:
                        # Multi-label rows use one binary column per class.
                        labels = []
                        for i in range(self.num_classes):
                            label_col = f'label_{i}'
                            if label_col in row:
                                labels.append(int(row[label_col]))
                            else:
                                labels.append(0)
                        label = np.array(labels, dtype=np.float32)
                    else:
                        if 'label' not in row:
                            logging.warning(f"No 'label' column found for {filename}")
                            continue
                        label = int(row['label'])

                    self.file_list.append({
                        'img_path': img_path,
                        'label': label,
                        'dataset': dataset,
                        'filename': filename
                    })

            except Exception as e:
                logging.error(f"Error loading CSV {csv_path}: {e}")
                continue

        if len(self.file_list) == 0:
            raise RuntimeError(f"No valid samples found for {self.mode} mode")

    def _compute_class_weights(self):
        """
        Compute class weights for imbalanced data.

        For single-label: Returns weights for each class [num_classes]
        For multi-label: Returns pos_weight for each class [num_classes]
        """
        if self.multi_label:
            class_counts = np.zeros(self.num_classes)
            for item in self.file_list:
                class_counts += item['label']

            # pos_weight balances each independent binary classifier.
            total_samples = len(self.file_list)
            pos_weights = np.zeros(self.num_classes)

            for i in range(self.num_classes):
                num_pos = class_counts[i]
                num_neg = total_samples - num_pos

                if num_pos > 0:
                    pos_weights[i] = num_neg / num_pos
                else:
                    pos_weights[i] = 1.0

            self.class_weights = pos_weights
            logging.info(f"Multi-label pos_weights: {self.class_weights}")
            self.class_weights = np.clip(self.class_weights, 0.1, 10.0)
            logging.info(f"Multi-label clipped pos_weights: {self.class_weights}")

        else:
            label_counts = np.zeros(self.num_classes)
            for item in self.file_list:
                label_counts[item['label']] += 1

            total_samples = len(self.file_list)
            weights = np.zeros(self.num_classes)

            for i in range(self.num_classes):
                if label_counts[i] > 0:
                    weights[i] = total_samples / (self.num_classes * label_counts[i])
                else:
                    weights[i] = 1.0

            self.class_weights = weights
            logging.info(f"Single-label class weights: {self.class_weights}")
            self.class_weights = np.clip(self.class_weights, 0.1, 10.0)
            logging.info(f"Single-label clipped class weights: {self.class_weights}")

            class_dist = {int(i): int(count) for i, count in enumerate(label_counts)}
            logging.info(f"Single-label class distribution: {class_dist}")


    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - image: Tensor of shape [1, D, H, W]
                - label: Tensor of shape [] (single-label) or [num_classes] (multi-label)
                - filename: str
                - dataset: str (optional)
        """
        info = self.file_list[idx]

        img = np.load(info['img_path'], mmap_mode='r')
        img = np.array(img, dtype=np.float32)

        if img.ndim == 4:
            img = img[0]
        elif img.ndim == 2:
            img = img[np.newaxis, :, :]
        elif img.ndim != 3:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        # Classification examples normalize per volume after loading npy arrays.
        img = self._normalize_intensity(img)

        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        # Reuse segmentation augmentation utilities, which expect an image/mask pair.
        dummy_label = torch.zeros_like(img_tensor)

        if self.mode == 'train':
            img_tensor = self._apply_augmentation(img_tensor, dummy_label)
        else:
            img_tensor = self._center_crop_or_pad_tensor(img_tensor)

        img_tensor = img_tensor.squeeze(0).contiguous()

        if self.multi_label:
            label_tensor = torch.from_numpy(info['label']).float()
        else:
            label_tensor = torch.tensor(info['label'], dtype=torch.long)

        return {
            'image': img_tensor,
            'label': label_tensor,
            'filename': info['filename'],
            'dataset': info['dataset']
        }


    def _normalize_intensity(self, img: np.ndarray) -> np.ndarray:
        """
        z-score normalization
        """
        mean = np.mean(img)
        std = np.std(img)

        if std == 0:
            return img - mean
        else:
            return (img - mean) / std

    def _pad_or_crop_to_size(
        self,
        image: torch.Tensor,
        dummy_label: torch.Tensor,
        target_size: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad or crop image to target size.

        Args:
            image: Input tensor [1, 1, D, H, W]
            dummy_label: Dummy label tensor [1, 1, D, H, W]
            target_size: Target size [D, H, W]

        Returns:
            Tuple of (image, dummy_label) tensors with target size
        """
        _, _, img_d, img_h, img_w = image.shape
        target_d, target_h, target_w = target_size

        pad_d = max(0, target_d - img_d)
        pad_h = max(0, target_h - img_h)
        pad_w = max(0, target_w - img_w)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            padding = (
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2,
                pad_d // 2, pad_d - pad_d // 2
            )
            image = torch.nn.functional.pad(image, padding, mode='constant', value=0)
            dummy_label = torch.nn.functional.pad(dummy_label, padding, mode='constant', value=0)
            _, _, img_d, img_h, img_w = image.shape

        if img_d > target_d or img_h > target_h or img_w > target_w:
            start_d = random.randint(0, max(0, img_d - target_d))
            start_h = random.randint(0, max(0, img_h - target_h))
            start_w = random.randint(0, max(0, img_w - target_w))

            image = image[:, :,
                         start_d:start_d + target_d,
                         start_h:start_h + target_h,
                         start_w:start_w + target_w]
            dummy_label = dummy_label[:, :,
                                     start_d:start_d + target_d,
                                     start_h:start_h + target_h,
                                     start_w:start_w + target_w]

        return image.contiguous(), dummy_label.contiguous()

    def _apply_augmentation(
        self,
        image: torch.Tensor,
        dummy_label: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply augmentation to image using augmentation module functions.

        Args:
            image: Input image tensor with shape [1, 1, D, H, W]
            dummy_label: Dummy label tensor (not used, just for compatibility)

        Returns:
            Augmented image tensor [1, 1, D, H, W]
        """
        config = self.aug_config

        if self.aug_device == 'gpu' and torch.cuda.is_available():
            image = image.to(self.device)
            dummy_label = dummy_label.to(self.device)

        d, h, w = self.training_size
        aug_size = [d + 40, h + 40, w + 40]

        image, dummy_label = self._pad_or_crop_to_size(image, dummy_label, aug_size)

        if random.random() < config['affine_prob']:
            image, dummy_label = augmentation.random_scale_rotate_translate_3d(
                image, dummy_label,
                scale=config['scale'],
                rotate=config['rotate'],
                translate=config['translate'],
                shear=config['shear'],
            )
            image, dummy_label = augmentation.crop_3d(image, dummy_label, self.training_size, mode="center")
        else:
            image, dummy_label = augmentation.crop_3d(image, dummy_label, self.training_size, mode="random")

        aug_prob = config['aug_prob']

        if random.random() < aug_prob:
            image = augmentation.brightness_multiply(image, config['brightness_range'])

        if random.random() < aug_prob:
            image = augmentation.brightness_additive(image, std=config['brightness_additive_std'])

        if random.random() < aug_prob:
            image = augmentation.gamma(image, config['gamma_range'])

        if random.random() < aug_prob:
            image = augmentation.contrast(image, config['contrast_range'])

        if random.random() < aug_prob:
            image = augmentation.gaussian_blur(image, config['blur_range'])

        if random.random() < aug_prob:
            image = augmentation.gaussian_noise(image, std=config['gaussian_noise_std'])

        if image.is_cuda:
            image = image.cpu()

        return image.contiguous()

    def _center_crop_or_pad_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Center crop or pad tensor image to training size.

        Args:
            img_tensor: Input tensor [1, 1, D, H, W]

        Returns:
            Cropped/padded tensor [1, 1, D, H, W]
        """
        d, h, w = self.training_size
        _, _, img_d, img_h, img_w = img_tensor.shape

        pad_d = max(0, d - img_d)
        pad_h = max(0, h - img_h)
        pad_w = max(0, w - img_w)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            padding = (
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2,
                pad_d // 2, pad_d - pad_d // 2
            )
            img_tensor = torch.nn.functional.pad(img_tensor, padding, mode='constant', value=0)
            _, _, img_d, img_h, img_w = img_tensor.shape

        start_d = (img_d - d) // 2
        start_h = (img_h - h) // 2
        start_w = (img_w - w) // 2

        img_tensor = img_tensor[:, :, start_d:start_d + d, start_h:start_h + h, start_w:start_w + w]

        return img_tensor.contiguous()
