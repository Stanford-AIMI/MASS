"""Ground-truth segmentation dataset for MASS evaluation.

This module provides ``MetaUniversalDataset``, the processed-dataset loader used
for in-context segmentation evaluation. It reads ``*_image.npy`` and
``*_gt.npy`` files, constructs class targets, and prepares fixed or random
reference image-mask pairs for each semantic class.
"""

import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional, Union

from utils.registry import register_dataset
from . import augmentation
from .split import (
    train_test_split,
    dataset_lab_map,
    reference_dict,
)


@register_dataset("MetaUniversalDataset")
class MetaUniversalDataset(Dataset):
    """
    Ground-truth dataset used for MASS segmentation evaluation.

    The dataset loads full processed volumes for evaluation. In
    ``test_incontext`` mode it also prepares cropped reference image-mask pairs
    for each requested global class id.

    Args:
        datasets: List of dataset names to include
        mode: Dataset mode ('test' or 'test_incontext')
        data_root: Root directory for data
        training_size: Crop size used for reference examples
        num_references_per_class: Number of references per class in test_incontext mode
        reference_mode: Mode for reference selection ('random' or 'fixed')
        spacing: Voxel spacing in array axis order (z, y, x), matching [D, H, W]
    """

    def __init__(
        self,
        datasets: List[str] = ["bcv"],
        mode: str = "test_incontext",
        data_root: str = "./data",
        training_size: List[int] = [128, 128, 128],
        num_references_per_class: int = 1,
        reference_mode: str = "random",
        spacing: Optional[Union[List[float], Tuple[float, float, float]]] = None,
    ):
        assert mode in ("test", "test_incontext"), f"Invalid mode: {mode}"
        assert reference_mode in ("random", "fixed"), f"Invalid reference_mode: {reference_mode}"

        self.mode = mode
        self.datasets = list(datasets)
        self.data_root = Path(data_root)
        self.training_size = training_size
        self.num_references_per_class = num_references_per_class
        self.reference_mode = reference_mode
        # Spacing follows numpy array axis order: (z, y, x) for [D, H, W].
        self.spacing = tuple(float(x) for x in (spacing or [1.5, 1.5, 1.5]))

        self.dataset_lab_map = dataset_lab_map

        # max_classes is based on local saved GT labels (background + 1..N),
        # not on the sparse MASS global class ids in dataset_lab_map.
        max_class_idx = -1
        for ds in self.datasets:
            if ds in self.dataset_lab_map:
                if len(self.dataset_lab_map[ds]) > 0:
                    max_class_idx = max(max_class_idx, len(self.dataset_lab_map[ds]))

        self.max_classes = max_class_idx + 1 if max_class_idx >= 0 else 0
        logging.info(f"Calculated max_classes = {self.max_classes}")

        self._prepare_file_list()

        if self.mode == "test_incontext":
            self.ref_img_dict = {}
            self.ref_mask_dict = {}

            all_classes = []
            for ds in self.datasets:
                all_classes.extend(self.dataset_lab_map[ds])
            all_classes = sorted(list(set(all_classes)))

            self._prepare_incontext_references(all_classes)

        logging.info(f"[MetaUniversalDataset] mode={self.mode} - {len(self.file_list)} samples loaded")
        if self.mode == "test_incontext":
            logging.info(f"Number of references per class: {self.num_references_per_class}")
            logging.info(f"Reference images prepared for {len(self.ref_img_dict)} classes")

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get dataset item at index idx.

        Args:
            idx: Index of item to retrieve

        Returns:
            Tuple of tensors required for the specific mode
        """
        idx_mod = idx % len(self.file_list)
        tgt_img, tgt_mask = self._load_volume(self.file_list[idx_mod])

        file_info = self.file_list[idx_mod]
        class_targets = torch.from_numpy(file_info["label_map"])
        return tgt_img, tgt_mask, class_targets, file_info["spacing"]

    def _prepare_file_list(self):
        """Builds self.file_list: a list of dicts with img/lab paths and metadata."""
        self.file_list = []

        for ds in self.datasets:
            # Define split based on mode
            split_key = f"{ds}_{self.mode.split('_')[0]}"

            names = train_test_split.get(split_key, [])
            if not names:
                logging.warning(f"No samples found for {split_key}")
                continue

            dataset_dir = self.data_root / ds

            if not dataset_dir.exists():
                logging.warning(f"Dataset directory {dataset_dir} not found")
                continue

            for name in names:
                img_name = str(name)

                img_npy_path = dataset_dir / f"{img_name}_image.npy"
                gt_npy_path = dataset_dir / f"{img_name}_gt.npy"

                if img_npy_path.exists() and gt_npy_path.exists():
                    self.file_list.append({
                        "img_npy_path": img_npy_path,
                        "gt_npy_path": gt_npy_path,
                        "img_name": img_name,
                        "label_map": np.array(self.dataset_lab_map[ds], dtype=int),
                        "spacing": self.spacing,
                        "ds_name": ds,
                    })
                else:
                    if not img_npy_path.exists():
                        logging.warning(f"Image .npy file not found: {img_npy_path}")
                    if not gt_npy_path.exists():
                        logging.warning(f"GT .npy file not found: {gt_npy_path}")

    def _load_volume(self, info: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a full processed image and convert GT labels to one-hot masks."""
        arr = np.load(info["img_npy_path"], mmap_mode='r')
        lab = np.load(info["gt_npy_path"], mmap_mode='r')

        img_t = torch.from_numpy(arr).unsqueeze(0).float()
        lab_t = torch.from_numpy(lab).unsqueeze(0).long()

        mask_bin = self._to_one_hot(lab_t)

        return img_t, mask_bin

    def _to_one_hot(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor to one-hot encoding.
        Handles both 3D [D,H,W] and 4D [C,D,H,W] input tensors.
        """
        if mask.ndim > 3:
            spatial_dims = mask.shape[-3:]
            mask = mask.reshape(-1, *spatial_dims).squeeze(0)

        # The saved GT is indexed from 1 for foreground classes; channel 0 is
        # background and is dropped for metric/evaluation tensors.
        oh = F.one_hot(mask, num_classes=self.max_classes)
        oh = oh.permute(3, 0, 1, 2).to(torch.int8)
        return oh[1:]

    def _prepare_fixed_references(self, classes: List[int]):
        """
        Prepare fixed reference image-mask pairs based on configuration.
        Assumes reference_dict contains FULL group names (no prefix probing).
        """
        for cls in classes:
            if cls not in reference_dict:
                logging.warning(f"No fixed reference configured for class {cls}")
                continue

            dataset_name = None
            for ds, labels in self.dataset_lab_map.items():
                if cls in labels:
                    dataset_name = ds
                    break

            if dataset_name is None:
                logging.warning(f"No dataset found for class {cls}")
                continue

            self.ref_img_dict[cls] = []
            self.ref_mask_dict[cls] = []

            sample_names = reference_dict[cls]
            if len(sample_names) != self.num_references_per_class:
                logging.warning(
                    f"Config has {len(sample_names)} references for class {cls}, expected {self.num_references_per_class}"
                )

            references_found = 0
            dataset_dir = self.data_root / dataset_name

            if not dataset_dir.exists():
                logging.warning(f"Dataset directory not found: {dataset_dir}")
                continue

            for sample_name in sample_names[:self.num_references_per_class]:
                img_name = str(sample_name)
                img_npy_path = dataset_dir / f"{img_name}_image.npy"
                gt_npy_path = dataset_dir / f"{img_name}_gt.npy"

                if not img_npy_path.exists() or not gt_npy_path.exists():
                    logging.warning(f"Could not find .npy files for sample {img_name} for class {cls} in dataset {dataset_name}")
                    continue

                try:
                    img = np.load(img_npy_path, mmap_mode='r')
                    lab = np.load(gt_npy_path, mmap_mode='r')

                    cls_idx = list(self.dataset_lab_map[dataset_name]).index(cls)

                    if np.any(lab == cls_idx + 1):
                        locations = np.argwhere(lab == cls_idx + 1)
                        if len(locations) == 0:
                            continue
                        center = locations[len(locations)//2]

                        crop_size = self.training_size
                        img_crop, lab_crop = augmentation.np_crop_around_coordinate_3d(
                            img, lab, crop_size, center, mode="center"
                        )

                        img_t = torch.from_numpy(img_crop).unsqueeze(0).float()
                        lab_t = torch.from_numpy(lab_crop).unsqueeze(0).long()

                        # References are stored as binary masks for one global
                        # class even when the source GT is multi-class.
                        binary_mask = (lab_t == cls_idx + 1).to(torch.int8)

                        self.ref_img_dict[cls].append(img_t)
                        self.ref_mask_dict[cls].append(binary_mask)

                        references_found += 1
                        logging.info(f"Loaded fixed reference for class {cls} from {img_name}")

                except Exception as e:
                    logging.warning(f"Error processing {img_name} for class {cls}: {e}")
                    continue

            if references_found != min(len(sample_names), self.num_references_per_class):
                logging.warning(f"Only found {references_found} references for class {cls}")
                if references_found < self.num_references_per_class:
                    raise RuntimeError(
                        f"Not enough references for class {cls}: found {references_found}, expected {self.num_references_per_class}"
                    )

    def _prepare_incontext_references(self, classes):
        """
        Prepare reference image-mask pairs for in-context testing.
        Stores up to num_references_per_class samples for each class.
        Uses either random or fixed selection based on reference_mode.
        """
        if self.reference_mode == "fixed":
            self._prepare_fixed_references(classes)
            return

        # Random selection from training split based on FULL names in the split
        for cls in classes:
            dataset_name = None
            for ds, labels in self.dataset_lab_map.items():
                if cls in labels:
                    dataset_name = ds
                    break

            if dataset_name is None:
                logging.warning(f"No dataset found for class {cls}")
                continue

            train_key = f"{dataset_name}_train"
            train_names = train_test_split.get(train_key, [])

            if not train_names:
                logging.warning(f"No training samples found for {dataset_name}")
                continue

            self.ref_img_dict[cls] = []
            self.ref_mask_dict[cls] = []

            shuffled_names = list(train_names)
            random.shuffle(shuffled_names)

            references_found = 0
            dataset_dir = self.data_root / dataset_name

            if not dataset_dir.exists():
                logging.warning(f"Dataset directory not found: {dataset_dir}")
                continue

            for name in shuffled_names:
                name = str(name)
                if references_found >= self.num_references_per_class:
                    break

                img_npy_path = dataset_dir / f"{name}_image.npy"
                gt_npy_path = dataset_dir / f"{name}_gt.npy"

                if not img_npy_path.exists() or not gt_npy_path.exists():
                    continue

                try:
                    img = np.load(img_npy_path, mmap_mode='r')
                    lab = np.load(gt_npy_path, mmap_mode='r')

                    cls_idx = list(self.dataset_lab_map[dataset_name]).index(cls)

                    if np.any(lab == cls_idx + 1):
                        locations = np.argwhere(lab == cls_idx + 1)
                        if len(locations) == 0:
                            continue

                        center = locations[len(locations)//2]

                        crop_size = self.training_size
                        img_crop, lab_crop = augmentation.np_crop_around_coordinate_3d(
                            img, lab, crop_size, center, mode="center"
                        )

                        img_t = torch.from_numpy(img_crop).unsqueeze(0).float()
                        lab_t = torch.from_numpy(lab_crop).unsqueeze(0).long()

                        # References are stored as binary masks for one global
                        # class even when the source GT is multi-class.
                        binary_mask = (lab_t == cls_idx + 1).to(torch.int8)

                        self.ref_img_dict[cls].append(img_t)
                        self.ref_mask_dict[cls].append(binary_mask)

                        references_found += 1
                        logging.info(
                            f"Stored reference {references_found}/{self.num_references_per_class} for class {cls} from {name}"
                        )

                except Exception as e:
                    logging.warning(f"Error processing {name} for class {cls}: {e}")
                    continue

            if references_found != self.num_references_per_class:
                logging.warning(
                    f"Only found {references_found} references for class {cls}, requested {self.num_references_per_class}"
                )
                raise RuntimeError(
                    f"Not correct number of references for class {cls}: found {references_found}, expected {self.num_references_per_class}"
                )
            else:
                logging.info(f"Found {references_found} references for class {cls}")



    def get_reference_for_class(self, class_id: int, index: Optional[int] = None) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        List[Tuple[torch.Tensor, torch.Tensor]]
    ]:
        """
        Get reference image and mask for a specific class.
        """
        if self.mode != "test_incontext":
            raise RuntimeError("This method is only available in test_incontext mode")

        if class_id not in self.ref_img_dict:
            raise KeyError(f"No reference available for class {class_id}")

        if index is not None:
            if index >= len(self.ref_img_dict[class_id]):
                raise IndexError(
                    f"Reference index {index} out of range for class {class_id} with {len(self.ref_img_dict[class_id])} references"
                )
            return self.ref_img_dict[class_id][index], self.ref_mask_dict[class_id][index]

        return [(self.ref_img_dict[class_id][i], self.ref_mask_dict[class_id][i])
                for i in range(len(self.ref_img_dict[class_id]))]

    def get_num_references_for_class(self, class_id: int) -> int:
        """
        Get the number of available references for a class.
        """
        if self.mode != "test_incontext":
            raise RuntimeError("This method is only available in test_incontext mode")

        if class_id not in self.ref_img_dict:
            return 0

        return len(self.ref_img_dict[class_id])
