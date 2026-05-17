"""
3D medical image augmentation and cropping utilities.

The dataset loaders use these functions to sample spatial crops and apply weak
or strong intensity/geometric augmentations during MASS pretraining and
downstream examples.
"""
import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Union, Optional, Sequence

def gaussian_noise(tensor_img: torch.Tensor, std: float, mean: float = 0) -> torch.Tensor:
    """
    Add Gaussian noise to a tensor image.
    
    Args:
        tensor_img: Input tensor image [1, C, D, H, W]
        std: Standard deviation of the noise
        mean: Mean of the noise
        
    Returns:
        Tensor with added Gaussian noise
    """
    return tensor_img + torch.randn(tensor_img.shape, device=tensor_img.device) * std + mean


def generate_3d_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Generate a 3D Gaussian kernel.
    
    Args:
        kernel_size: Size of the kernel
        sigma: Standard deviation of the Gaussian
        
    Returns:
        3D Gaussian kernel tensor [1, 1, kernel_size, kernel_size, kernel_size]
    """
    # Generate a meshgrid for the kernel
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    y = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    z = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    x, y, z = torch.meshgrid(x, y, z, indexing='ij')

    kernel = torch.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
    kernel = kernel / (2 * math.pi * sigma ** 2) ** 1.5
    kernel = kernel / kernel.sum()

    return kernel.unsqueeze(0).unsqueeze(0)


def gaussian_blur(tensor_img: torch.Tensor, sigma_range: Sequence[float] = (0.5, 1.0)) -> torch.Tensor:
    """
    Apply Gaussian blur to a 3D tensor image.
    
    Args:
        tensor_img: Input tensor image [1, C, D, H, W]
        sigma_range: Range for random sigma value [min, max]
        
    Returns:
        Blurred tensor image
    """
    if len(tensor_img.shape) != 5:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image [1, C, D, H, W]')
        
    sigma = torch.rand(1, device=tensor_img.device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    kernel_size = 2 * math.ceil(3 * sigma.item()) + 1
    
    # Generate kernel with the same dtype as the input tensor
    kernel = generate_3d_gaussian_kernel(kernel_size, sigma.item()).to(tensor_img.device).to(tensor_img.dtype)
    padding = [kernel_size // 2 for _ in range(3)]

    return F.conv3d(tensor_img, kernel, padding=padding)


def brightness_additive(
    tensor_img: torch.Tensor, 
    std: float, 
    mean: float = 0, 
    per_channel: bool = False
) -> torch.Tensor:
    """
    Add random brightness to a 3D tensor image.
    
    Args:
        tensor_img: Input tensor image [1, C, D, H, W]
        std: Standard deviation of the brightness
        mean: Mean of the brightness
        per_channel: Whether to apply different brightness to each channel
        
    Returns:
        Tensor with adjusted brightness
    """
    if len(tensor_img.shape) != 5:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image [1, C, D, H, W]')
        
    C = tensor_img.shape[1] if per_channel else 1
    rand_brightness = torch.normal(mean, std, size=(1, C, 1, 1, 1), device=tensor_img.device)
    
    return tensor_img + rand_brightness


def brightness_multiply(
    tensor_img: torch.Tensor, 
    multiply_range: Sequence[float] = (0.7, 1.3), 
    per_channel: bool = False
) -> torch.Tensor:
    """
    Multiply brightness of a 3D tensor image by a random factor.
    
    Args:
        tensor_img: Input tensor image [1, C, D, H, W]
        multiply_range: Range for random multiplication factor [min, max]
        per_channel: Whether to apply different factors to each channel
        
    Returns:
        Tensor with adjusted brightness
    """
    if len(tensor_img.shape) != 5:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image [1, C, D, H, W]')
        
    assert multiply_range[1] > multiply_range[0], 'Invalid range'
    
    C = tensor_img.shape[1] if per_channel else 1
    span = multiply_range[1] - multiply_range[0]
    rand_brightness = torch.rand(size=(1, C, 1, 1, 1), device=tensor_img.device) * span + multiply_range[0]
    
    return tensor_img * rand_brightness


def gamma(
    tensor_img: torch.Tensor, 
    gamma_range: Sequence[float] = (0.5, 2), 
    per_channel: bool = False, 
    retain_stats: bool = True
) -> torch.Tensor:
    """
    Apply random gamma correction to a 3D tensor image.
    
    Args:
        tensor_img: Input tensor image [1, C, D, H, W]
        gamma_range: Range for random gamma value [min, max]
        per_channel: Whether to apply different gamma to each channel
        retain_stats: Whether to retain the mean and std of the original tensor
        
    Returns:
        Gamma-corrected tensor
    """
    if len(tensor_img.shape) != 5:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image [1, C, D, H, W]')
        
    _, C, D, H, W = tensor_img.shape
    
    tmp_C = C if per_channel else 1
    tensor_img = tensor_img.view(tmp_C, -1)
    minm, _ = tensor_img.min(dim=1)
    maxm, _ = tensor_img.max(dim=1)
    minm, maxm = minm.unsqueeze(1), maxm.unsqueeze(1)  # unsqueeze for broadcast mechanism

    rng = maxm - minm

    mean = tensor_img.mean(dim=1).unsqueeze(1)
    std = tensor_img.std(dim=1).unsqueeze(1)
    gamma = torch.rand(C, 1, device=tensor_img.device) * (gamma_range[1] - gamma_range[0]) + gamma_range[0]

    tensor_img = torch.pow((tensor_img - minm) / rng, gamma) * rng + minm

    if retain_stats:
        tensor_img -= tensor_img.mean(dim=1).unsqueeze(1)
        tensor_img = tensor_img / tensor_img.std(dim=1).unsqueeze(1) * std + mean

    return tensor_img.view(1, C, D, H, W)


def contrast(
    tensor_img: torch.Tensor, 
    contrast_range: Sequence[float] = (0.65, 1.5), 
    per_channel: bool = False, 
    preserve_range: bool = True
) -> torch.Tensor:
    """
    Apply random contrast adjustment to a 3D tensor image.
    
    Args:
        tensor_img: Input tensor image [1, C, D, H, W]
        contrast_range: Range for random contrast factor [min, max]
        per_channel: Whether to apply different contrast to each channel
        preserve_range: Whether to preserve the min-max range of the original tensor
        
    Returns:
        Contrast-adjusted tensor
    """
    if len(tensor_img.shape) != 5:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image [1, C, D, H, W]')
        
    _, C, D, H, W = tensor_img.shape

    tmp_C = C if per_channel else 1
    tensor_img = tensor_img.view(tmp_C, -1)
    minm, _ = tensor_img.min(dim=1)
    maxm, _ = tensor_img.max(dim=1)
    minm, maxm = minm.unsqueeze(1), maxm.unsqueeze(1)  # unsqueeze for broadcast mechanism

    mean = tensor_img.mean(dim=1).unsqueeze(1)
    factor = torch.rand(C, 1, device=tensor_img.device) * (contrast_range[1] - contrast_range[0]) + contrast_range[0]

    tensor_img = (tensor_img - mean) * factor + mean

    if preserve_range:
        tensor_img = torch.clamp(tensor_img, min=minm, max=maxm)

    return tensor_img.view(1, C, D, H, W)


def mirror(tensor_img: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """
    Mirror a 3D tensor image along a specified axis.
    
    Args:
        tensor_img: Input tensor image [1, C, D, H, W]
        axis: The axis for mirroring (0: depth, 1: height, 2: width)
        
    Returns:
        Mirrored tensor
    """
    if len(tensor_img.shape) != 5:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image [1, C, D, H, W]')
        
    assert axis in [0, 1, 2], "axis should be either 0, 1 or 2 for volume images"
    
    return torch.flip(tensor_img, dims=[2+axis])


def random_scale_rotate_translate_3d(
    tensor_img: torch.Tensor, 
    tensor_lab: torch.Tensor, 
    scale: Union[float, Sequence[float]] = 0.3, 
    rotate: Union[float, Sequence[float]] = 45, 
    translate: Union[float, Sequence[float]] = 0.1, 
    shear: Union[float, Sequence[float]] = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random scale, rotation, translation, and shear to a 3D tensor image and its label.
    
    Args:
        tensor_img: Input tensor image [1, C, D, H, W]
        tensor_lab: Input tensor label [1, 1, D, H, W]
        scale: Scale factor range (can be a single value or sequence of 3 values)
        rotate: Rotation angle range in degrees (can be a single value or sequence of 3 values)
        translate: Translation range as a fraction of image size (can be a single value or sequence of 3 values)
        shear: Shear range (can be a single value or sequence of 3 values)
        
    Returns:
        Tuple of transformed (image, label) tensors
    """
    device = tensor_img.device
    # Always use float32 for the transformation matrix and grid
    # This avoids precision issues with grid_sample
    compute_dtype = torch.float32

    if isinstance(scale, (float, int)):
        scale = [scale] * 3
    if isinstance(translate, (float, int)):
        translate = [translate] * 3
    if isinstance(rotate, (float, int)):
        rotate = [rotate] * 3
    if isinstance(shear, (float, int)):
        shear = [shear] * 3

    # Random scale factors
    scale_x = np.random.uniform(low=1-scale[0], high=1/(1-scale[0]))
    scale_y = np.random.uniform(low=1-scale[1], high=1/(1-scale[1]))
    scale_z = np.random.uniform(low=1-scale[2], high=1/(1-scale[2]))

    # Random shear factors
    shear_xy = np.random.uniform(-shear[0], shear[0])
    shear_xz = np.random.uniform(-shear[0], shear[0])
    shear_yx = np.random.uniform(-shear[1], shear[1])
    shear_yz = np.random.uniform(-shear[1], shear[1])
    shear_zx = np.random.uniform(-shear[2], shear[2])
    shear_zy = np.random.uniform(-shear[2], shear[2])

    # Random translation
    translate_x = np.random.uniform(-translate[0], translate[0])
    translate_y = np.random.uniform(-translate[1], translate[1])
    translate_z = np.random.uniform(-translate[2], translate[2])

    theta_scale = torch.tensor([
        [scale_x, shear_xy, shear_xz, translate_x],
        [shear_yx, scale_y, shear_yz, translate_y],
        [shear_zx, shear_zy, scale_z, translate_z], 
        [0, 0, 0, 1]
    ], dtype=compute_dtype, device=device)
                                
    # Random rotation angles
    angle_x = (float(np.random.randint(-rotate[0], max(rotate[0], 1))) / 180.) * math.pi 
    angle_y = (float(np.random.randint(-rotate[1], max(rotate[1], 1))) / 180.) * math.pi
    angle_z = (float(np.random.randint(-rotate[2], max(rotate[2], 1))) / 180.) * math.pi
    
    # Rotation matrices
    theta_rotate_x = torch.tensor([
        [1, 0, 0, 0],
        [0, math.cos(angle_x), -math.sin(angle_x), 0],
        [0, math.sin(angle_x), math.cos(angle_x), 0],
        [0, 0, 0, 1]
    ], dtype=compute_dtype, device=device)
                                    
    theta_rotate_y = torch.tensor([
        [math.cos(angle_y), 0, -math.sin(angle_y), 0],
        [0, 1, 0, 0],
        [math.sin(angle_y), 0, math.cos(angle_y), 0],
        [0, 0, 0, 1]
    ], dtype=compute_dtype, device=device)
                                    
    theta_rotate_z = torch.tensor([
        [math.cos(angle_z), -math.sin(angle_z), 0, 0],
        [math.sin(angle_z), math.cos(angle_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=compute_dtype, device=device)

    # Combine transformations
    theta = torch.mm(theta_rotate_x, theta_rotate_y)
    theta = torch.mm(theta, theta_rotate_z)
    theta = torch.mm(theta, theta_scale)[0:3, :].unsqueeze(0)
    
    # Generate grid in float32
    grid = F.affine_grid(theta, tensor_img.size(), align_corners=True)

    orig_img_dtype = tensor_img.dtype
    img_float = tensor_img.to(compute_dtype)
    lab_float = tensor_lab.to(compute_dtype)
    
    img_float = F.grid_sample(img_float, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    lab_float = F.grid_sample(lab_float, grid, mode='nearest', padding_mode='zeros', align_corners=True)
    
    tensor_img = img_float.to(orig_img_dtype)
    tensor_lab = lab_float.long()

    return tensor_img, tensor_lab


def crop_3d(
    tensor_img: torch.Tensor, 
    tensor_lab: torch.Tensor, 
    crop_size: Union[int, Sequence[int]], 
    mode: str = "random"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crop a 3D tensor image and its label.
    
    Args:
        tensor_img: Input tensor image [1, C, D, H, W]
        tensor_lab: Input tensor label [1, C_lab, D, H, W]
        crop_size: Size of the crop (single int or sequence of 3 ints)
        mode: Cropping mode, either 'random' or 'center'
        
    Returns:
        Tuple of cropped (image, label) tensors
    """
    assert mode in ['random', 'center'], "Invalid Mode, should be 'random' or 'center'"
    
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 3

    _, _, D, H, W = tensor_img.shape

    diff_D = D - crop_size[0]
    diff_H = H - crop_size[1]
    diff_W = W - crop_size[2]
    
    if mode == 'random':
        rand_z = np.random.randint(0, max(diff_D, 1))
        rand_y = np.random.randint(0, max(diff_H, 1))
        rand_x = np.random.randint(0, max(diff_W, 1))
    else:
        rand_z = diff_D // 2
        rand_y = diff_H // 2
        rand_x = diff_W // 2

    cropped_img = tensor_img[:, :, rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]
    cropped_lab = tensor_lab[:, :, rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]

    return cropped_img.contiguous(), cropped_lab.contiguous()


def np_crop_3d(
    np_img: np.ndarray, 
    np_lab: np.ndarray, 
    crop_size: Union[int, Sequence[int]], 
    mode: str = "random"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop a 3D numpy array image and its label.
    
    Args:
        np_img: Input numpy array image [D, H, W]
        np_lab: Input numpy array label [D, H, W]
        crop_size: Size of the crop (single int or sequence of 3 ints)
        mode: Cropping mode, either 'random' or 'center'
        
    Returns:
        Tuple of cropped (image, label) numpy arrays
    """
    assert mode in ['random', 'center'], "Invalid Mode, should be 'random' or 'center'"
    
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 3

    D, H, W = np_img.shape

    diff_D = D - crop_size[0]
    diff_H = H - crop_size[1]
    diff_W = W - crop_size[2]
    
    if mode == 'random':
        rand_z = np.random.randint(0, max(diff_D, 1))
        rand_y = np.random.randint(0, max(diff_H, 1))
        rand_x = np.random.randint(0, max(diff_W, 1))
    else:
        rand_z = diff_D // 2
        rand_y = diff_H // 2
        rand_x = diff_W // 2

    cropped_img = np_img[rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]
    cropped_lab = np_lab[rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]

    return np.ascontiguousarray(cropped_img), np.ascontiguousarray(cropped_lab)


def np_crop_around_coordinate_3d(
    np_img: np.ndarray, 
    np_lab: np.ndarray, 
    crop_size: Union[int, Sequence[int]], 
    coordinate: Sequence[int],
    mode: str = "center"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop a window around a specified coordinate in 3D numpy arrays.
    
    Args:
        np_img: Input 3D image array [D, H, W]
        np_lab: Input 3D label array [D, H, W]
        crop_size: Desired crop size (single int or list of 3 ints)
        coordinate: (z, y, x) coordinate to crop around
        mode: 'random' or 'center' - determines crop position relative to coordinate
        
    Returns:
        Tuple of cropped (image, label) numpy arrays
    """
    assert np_img.shape == np_lab.shape, "Image and label arrays must have the same shape"
    assert mode in ['random', 'center'], "Invalid Mode, should be 'random' or 'center'"
    
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 3
    
    z, y, x = coordinate
    D, H, W = np_img.shape
    
    assert 0 <= z < D and 0 <= y < H and 0 <= x < W, "Coordinate out of bounds"
    
    # Adjust crop size if it's larger than the image dimension
    crop_size = [min(crop_size[0], D), min(crop_size[1], H), min(crop_size[2], W)]
    
    half_crop = [size // 2 for size in crop_size]
    
    if mode == 'center':
        starts = [
            max(0, coord - half_crop[i]) for i, coord in enumerate([z, y, x])
        ]
        ends = [
            min(dim, coord + crop_size[i] - half_crop[i]) 
            for i, (coord, dim) in enumerate(zip([z, y, x], [D, H, W]))
        ]
        
        # Adjust starts if ends would exceed image dimensions
        starts = [
            max(0, min(starts[i], dim - crop_size[i]))
            for i, dim in enumerate([D, H, W])
        ]
        # Recalculate ends based on adjusted starts
        ends = [
            starts[i] + crop_size[i] for i in range(3)
        ]
    
    else:  # random mode
        min_starts = [
            max(0, coord - crop_size[i] + 1) for i, coord in enumerate([z, y, x])
        ]
        max_starts = [
            min(coord, dim - crop_size[i]) for i, (coord, dim) in enumerate(zip([z, y, x], [D, H, W]))
        ]
        
        # Generate random starts within valid ranges
        starts = [
            np.random.randint(min_start, max_start + 1) if max_start >= min_start else min_start
            for min_start, max_start in zip(min_starts, max_starts)
        ]
        ends = [
            starts[i] + crop_size[i] for i in range(3)
        ]
    
    # Perform the cropping
    cropped_img = np_img[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
    cropped_lab = np_lab[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
    
    return np.ascontiguousarray(cropped_img), np.ascontiguousarray(cropped_lab)


def np_crop_around_coordinate_two_views_3d(
    np_img: np.ndarray, 
    np_lab: np.ndarray, 
    crop_size: Union[int, Sequence[int]], 
    coordinate: Sequence[int], 
    overlap_ratio: float, 
    mode: str = 'random'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate two overlapping crops around a specified coordinate in 3D numpy arrays.
    
    Args:
        np_img: Input 3D image array [D, H, W]
        np_lab: Input 3D or 4D label array [D, H, W] or [N, D, H, W]
        crop_size: Desired crop size (single int or list of 3 ints)
        coordinate: (z, y, x) coordinate to crop around
        overlap_ratio: Minimum overlap ratio between crops (0 to 1)
        mode: 'random' or 'center' - determines crop position relative to coordinate
        
    Returns:
        Tuple of cropped images and labels (img_1, lab_1, img_2, lab_2)
    """
    if np_lab.ndim == 4:
        has_channels = True
        lab_shape = np_lab.shape[1:]  # Get spatial dimensions [D, H, W]
    else:
        has_channels = False
        lab_shape = np_lab.shape
    
    assert np_img.shape == lab_shape, f"Image shape {np_img.shape} and label spatial shape {lab_shape} must match"
    assert mode in ['random', 'center'], "Invalid Mode, should be 'random' or 'center'"
    assert 0 <= overlap_ratio <= 1, "Overlap ratio must be between 0 and 1"
    
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 3
    
    z, y, x = coordinate
    D, H, W = np_img.shape
    
    assert 0 <= z < D and 0 <= y < H and 0 <= x < W, "Coordinate out of bounds"
    
    # Adjust crop size if it's larger than the image dimension
    crop_size = [min(crop_size[0], D), min(crop_size[1], H), min(crop_size[2], W)]
    
    half_crop = [size // 2 for size in crop_size]
    
    if mode == 'center':
        # In center mode, both crops are identical
        starts = [
            max(0, coord - half_crop[i]) for i, coord in enumerate([z, y, x])
        ]
        starts = [
            max(0, min(starts[i], dim - crop_size[i]))
            for i, dim in enumerate([D, H, W])
        ]
        ends = [
            starts[i] + crop_size[i] for i in range(3)
        ]
        
        starts_1 = starts
        starts_2 = starts
        ends_1 = ends
        ends_2 = ends
        
    else:  # random mode
        min_starts = [
            max(0, coord - crop_size[i] + 1) for i, coord in enumerate([z, y, x])
        ]
        max_starts = [
            min(coord, dim - crop_size[i]) for i, (coord, dim) in enumerate(zip([z, y, x], [D, H, W]))
        ]
        
        # Generate random starts for first crop
        starts_1 = [
            np.random.randint(min_start, max_start + 1) if max_start >= min_start else min_start
            for min_start, max_start in zip(min_starts, max_starts)
        ]
        ends_1 = [
            starts_1[i] + crop_size[i] for i in range(3)
        ]
        
        min_overlap = [int(size * overlap_ratio) for size in crop_size]
        
        starts_2 = []
        ends_2 = []
        
        for i in range(3):
            max_distance = crop_size[i] - min_overlap[i]
            min_start_2 = max(min_starts[i], starts_1[i] - max_distance)
            max_start_2 = min(max_starts[i], starts_1[i] + max_distance)
            
            # Generate random start for second crop within valid range
            start_2 = np.random.randint(min_start_2, max_start_2 + 1) if max_start_2 >= min_start_2 else min_start_2
            starts_2.append(start_2)
            ends_2.append(start_2 + crop_size[i])
    
    # Perform the cropping for both views
    cropped_img_1 = np_img[starts_1[0]:ends_1[0], starts_1[1]:ends_1[1], starts_1[2]:ends_1[2]]
    cropped_img_2 = np_img[starts_2[0]:ends_2[0], starts_2[1]:ends_2[1], starts_2[2]:ends_2[2]]
    
    if has_channels:
        cropped_lab_1 = np_lab[:, starts_1[0]:ends_1[0], starts_1[1]:ends_1[1], starts_1[2]:ends_1[2]]
        cropped_lab_2 = np_lab[:, starts_2[0]:ends_2[0], starts_2[1]:ends_2[1], starts_2[2]:ends_2[2]]
    else:
        cropped_lab_1 = np_lab[starts_1[0]:ends_1[0], starts_1[1]:ends_1[1], starts_1[2]:ends_1[2]]
        cropped_lab_2 = np_lab[starts_2[0]:ends_2[0], starts_2[1]:ends_2[1], starts_2[2]:ends_2[2]]
    
    return (
        np.ascontiguousarray(cropped_img_1), np.ascontiguousarray(cropped_lab_1),
        np.ascontiguousarray(cropped_img_2), np.ascontiguousarray(cropped_lab_2)
    )



def np_crop_around_coordinate_two_views_with_mask_slicing_3d(
    np_img: np.ndarray, 
    np_lab: np.ndarray,
    selected_indices: np.ndarray,
    crop_size: Union[int, Sequence[int]], 
    coordinate: Sequence[int], 
    overlap_ratio: float, 
    mode: str = 'random'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate two overlapping crops around a specified coordinate in 3D numpy arrays.
    
    Args:
        np_img: Input 3D image array [D, H, W]
        np_lab: Input 3D or 4D label array [D, H, W] or [N, D, H, W]
        crop_size: Desired crop size (single int or list of 3 ints)
        coordinate: (z, y, x) coordinate to crop around
        overlap_ratio: Minimum overlap ratio between crops (0 to 1)
        mode: 'random' or 'center' - determines crop position relative to coordinate
        
    Returns:
        Tuple of cropped images and labels (img_1, lab_1, img_2, lab_2)
    """

    if np_lab.ndim == 4:
        has_channels = True
        lab_shape = np_lab.shape[1:]  # Get spatial dimensions [D, H, W]
    else:
        has_channels = False
        lab_shape = np_lab.shape
    
    assert np_img.shape == lab_shape, f"Image shape {np_img.shape} and label spatial shape {lab_shape} must match"
    assert mode in ['random', 'center'], "Invalid Mode, should be 'random' or 'center'"
    assert 0 <= overlap_ratio <= 1, "Overlap ratio must be between 0 and 1"
    
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 3
    
    z, y, x = coordinate
    D, H, W = np_img.shape
    
    assert 0 <= z < D and 0 <= y < H and 0 <= x < W, "Coordinate out of bounds"
    
    # Adjust crop size if it's larger than the image dimension
    crop_size = [min(crop_size[0], D), min(crop_size[1], H), min(crop_size[2], W)]
    
    half_crop = [size // 2 for size in crop_size]
    
    if mode == 'center':
        # In center mode, both crops are identical
        starts = [
            max(0, coord - half_crop[i]) for i, coord in enumerate([z, y, x])
        ]
        starts = [
            max(0, min(starts[i], dim - crop_size[i]))
            for i, dim in enumerate([D, H, W])
        ]
        ends = [
            starts[i] + crop_size[i] for i in range(3)
        ]
        
        starts_1 = starts
        starts_2 = starts
        ends_1 = ends
        ends_2 = ends
        
    else:  # random mode
        min_starts = [
            max(0, coord - crop_size[i] + 1) for i, coord in enumerate([z, y, x])
        ]
        max_starts = [
            min(coord, dim - crop_size[i]) for i, (coord, dim) in enumerate(zip([z, y, x], [D, H, W]))
        ]
        
        # Generate random starts for first crop
        starts_1 = [
            np.random.randint(min_start, max_start + 1) if max_start >= min_start else min_start
            for min_start, max_start in zip(min_starts, max_starts)
        ]
        ends_1 = [
            starts_1[i] + crop_size[i] for i in range(3)
        ]
        
        min_overlap = [int(size * overlap_ratio) for size in crop_size]
        
        starts_2 = []
        ends_2 = []
        
        for i in range(3):
            max_distance = crop_size[i] - min_overlap[i]
            min_start_2 = max(min_starts[i], starts_1[i] - max_distance)
            max_start_2 = min(max_starts[i], starts_1[i] + max_distance)
            
            # Generate random start for second crop within valid range
            start_2 = np.random.randint(min_start_2, max_start_2 + 1) if max_start_2 >= min_start_2 else min_start_2
            starts_2.append(start_2)
            ends_2.append(start_2 + crop_size[i])
    
    # Perform the cropping for both views
    cropped_img_1 = np_img[starts_1[0]:ends_1[0], starts_1[1]:ends_1[1], starts_1[2]:ends_1[2]]
    cropped_img_2 = np_img[starts_2[0]:ends_2[0], starts_2[1]:ends_2[1], starts_2[2]:ends_2[2]]
    
    if has_channels:
        cropped_lab_1 = np_lab[selected_indices, starts_1[0]:ends_1[0], starts_1[1]:ends_1[1], starts_1[2]:ends_1[2]]
        cropped_lab_2 = np_lab[selected_indices, starts_2[0]:ends_2[0], starts_2[1]:ends_2[1], starts_2[2]:ends_2[2]]
    else:
        cropped_lab_1 = np_lab[starts_1[0]:ends_1[0], starts_1[1]:ends_1[1], starts_1[2]:ends_1[2]]
        cropped_lab_2 = np_lab[starts_2[0]:ends_2[0], starts_2[1]:ends_2[1], starts_2[2]:ends_2[2]]
    
    return (
        np.ascontiguousarray(cropped_img_1), np.ascontiguousarray(cropped_lab_1),
        np.ascontiguousarray(cropped_img_2), np.ascontiguousarray(cropped_lab_2)
    )
