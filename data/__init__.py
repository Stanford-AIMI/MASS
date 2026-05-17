"""Dataset package initialization for MASS.

Importing this package registers the dataset classes used by training,
in-context evaluation, finetuning, and classification linear probing examples.
"""

from . import augmentation
from . import dataset_gt
from . import dataset_ss
from . import dataset_finetune
from . import dataset_classification

from .dataset_gt import MetaUniversalDataset
from .dataset_ss import MaskGuidedSelfSupervisedDataset
from .dataset_finetune import FineTuningDataset
from .dataset_classification import ClassificationDataset

__all__ = [
    'augmentation',
    'MetaUniversalDataset',
    "MaskGuidedSelfSupervisedDataset",
    "FineTuningDataset",
    "ClassificationDataset",
]
