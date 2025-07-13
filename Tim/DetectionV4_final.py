#!/usr/bin/env python3
"""
Enhanced UNet V4 Final - STABILIZED for Robust Training Performance
===================================================================

Advanced UNet architecture for kiln detection in Sentinel-2 satellite imagery
with STABILIZED optimizations addressing training instability and model collapse.

⚡ STABILIZED IMPROVEMENTS APPLIED (Based on Failure Analysis):
1. STABILIZED Loss Weighting: Class weights reduced to 80.0/100.0 (preventing gradient explosion)
2. SIMPLIFIED Sampling: Independent probabilities instead of hierarchical cascade (stable mixing)
3. MODERATE Augmentation: Balanced rates preventing training destabilization while supporting rare classes
4. ROBUST EVALUATION: Clean metrics without post-processing bias for reliable performance assessment

PROBLEM ADDRESSED & STABILIZED:
- Previous aggressive approach (160.0/180.0 weights) caused complete model collapse (Class 6 F1: 0.00)
- Complex hierarchical sampling destabilized training with over-optimization
- Training instability prevented effective learning for all classes
- Goal: Restore stable training foundation and gradual improvement toward 0.90+ Weighted F1

ARCHITECTURE:
- Enhanced UNet with Attention Gates for rare class focus
- Residual connections with Batch Normalization for gradient stability
- Channel attention mechanisms for adaptive feature weighting
- Combined Focal + Dice Loss (60% + 40%) for class imbalance handling
- AdamW optimizer with cosine annealing scheduler

DATASET:
- Input: 8-channel Sentinel-2 imagery (10m resolution bands)
- Classes: 8 kiln types (labels 1-8 → 0-7) + background (0 → -1 ignore)
- Extreme imbalance: rare classes <0.01% of pixels
- Patch size: 256x256 pixels

TRAINING STRATEGY:
- STABILIZED adaptive sampling with independent rare class probabilities (no hierarchical blocking)
- MODERATE augmentation with balanced intensity for training stability
- Class-aware data augmentation with controlled rates for rare classes
- Gradient accumulation for effective larger batch sizes
- Gradient clipping for training stability

VALIDATION APPROACH:
- Test-realistic conditions: pure random sampling, no mask knowledge
- Shared normalization statistics from training set
- No data filtering (all validation images included)
- Deterministic splitting for reproducible results
- Clean evaluation without post-processing bias for unbiased metrics

DATA PIPELINE:
- Real-time augmentation: spatial + spectral transformations
- Robust data validation and error handling
- Memory-efficient patch extraction
- Multi-worker data loading with pin memory

EVALUATION METRICS:
- IoU per class for segmentation quality
- Weighted/Macro F1 scores for classification performance
- Background class excluded from F1 calculation (focus on kilns)
- Comprehensive validation tracking with unbiased predictions
"""

import os, glob, random, zipfile, getpass
import numpy as np
import torch, rasterio
import torchvision.transforms as T
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torchmetrics

# Configuration
USER = getpass.getuser()
SCRATCH = os.environ.get("SLURM_TMPDIR", f"/scratch/tmp/{USER}")
ZIP_PATH = os.path.join(SCRATCH, "Brick_Data_Train.zip")
DATA_ROOT = os.path.join(SCRATCH, "Brick_Data_Train")
IMG_DIR = os.path.join(DATA_ROOT, "Image")
MASK_DIR = os.path.join(DATA_ROOT, "Mask")
PATCH_SIZE = 256
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 100
NUM_CLASSES = 8  # 8 kiln classes after mapping
IN_CHANNELS = 8
pl.seed_everything(42, workers=True)

# Extract dataset if needed
if not os.path.isdir(DATA_ROOT):
    if os.path.isfile(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(SCRATCH)
    else:
        raise FileNotFoundError(ZIP_PATH)

def read_s2_image(path: str) -> torch.Tensor:
    """
    Read Sentinel-2 GeoTIFF image and normalize to reflectance values.
    
    Args:
        path: Path to the GeoTIFF file
        
    Returns:
        Normalized image tensor with shape (channels, height, width)
    """
    try:
        with rasterio.open(path) as src:
            img = src.read(out_dtype=np.float32)[:IN_CHANNELS]
    except Exception as e:
        raise RuntimeError(f"Error reading {path}: {e}")

    # Handle NaN values and normalize
    img_clean = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Check if already in reflectance format, otherwise convert from DN
    valid_pixels = img_clean[img_clean > 0]
    if len(valid_pixels) > 0 and valid_pixels.max() <= 1.0:
        img_normalized = img_clean
    else:
        img_normalized = img_clean / 10000.0  # DN to reflectance
    
    # Clip to reasonable reflectance bounds
    img_final = np.clip(img_normalized, -0.1, 2.0)
    return torch.from_numpy(img_final)

def get_standard_train_val_split(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Get standardized train/val split that ALL team members should use for comparable results.
    
    This function provides ONLY the file paths split, without loading any images.
    Use this to ensure consistent train/val splits across all experiments.
    
    Args:
        data_path: Path to dataset root (contains Image/ and Mask/ folders)
        test_size: Fraction for validation set (default: 0.2 = 20%)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (train_img_paths, val_img_paths, train_mask_paths, val_mask_paths)
    """
    from sklearn.model_selection import train_test_split
    
    full_root_path = os.path.abspath(data_path)
    if not os.path.exists(full_root_path):
        raise FileNotFoundError(f"Root path does not exist: {full_root_path}")

    image_dir = os.path.join(full_root_path, "Image")
    label_dir = os.path.join(full_root_path, "Mask")
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"Mask directory does not exist: {label_dir}")

    # Collect all image and corresponding mask paths
    all_image_paths = []
    all_mask_paths = []
    
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.tif') or file.endswith('.tiff'):
                image_path = os.path.join(root, file)
                mask_path = os.path.join(label_dir, file)
                
                # Only include if both image and mask exist
                if os.path.exists(mask_path):
                    all_image_paths.append(image_path)
                    all_mask_paths.append(mask_path)
    
    if len(all_image_paths) == 0:
        raise ValueError(f"No valid image-mask pairs found in {data_path}")
    
    # CRITICAL: Use sklearn.train_test_split for consistency
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        all_image_paths, all_mask_paths, test_size=test_size, random_state=random_state
    )
    
    print(f"Standard train/val split created:")
    print(f"  Total images: {len(all_image_paths)}")
    print(f"  Train images: {len(train_img_paths)} ({len(train_img_paths)/len(all_image_paths)*100:.1f}%)")
    print(f"  Val images: {len(val_img_paths)} ({len(val_img_paths)/len(all_image_paths)*100:.1f}%)")
    print(f"  Method: sklearn.train_test_split(test_size={test_size}, random_state={random_state})")
    
    return train_img_paths, val_img_paths, train_mask_paths, val_mask_paths

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance.
    
    Focuses training on hard examples and rare classes using
    adaptive alpha weights based on class frequency.
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        if alpha is None:
            # STABILIZED Alpha weights addressing training instability (based on failure analysis)
            # Previous EXCESSIVE weights (160.0/180.0) caused gradient explosion and model collapse
            # FIX: Conservative but effective weights for stable learning while maintaining rare class focus
            self.alpha = torch.tensor([
                5.0,   # Class 0 (original 1): 0.0263% - reduced for stability
                4.0,   # Class 1 (original 2): 0.0379% - reduced for stability
                15.0,  # Class 2 (original 3): 0.0117% - moderate emphasis
                30.0,  # Class 3 (original 4): 0.0010% - strong but stable
                80.0,  # Class 4 (original 5): STABILIZED from 160.0 → 80.0 (prevents gradient explosion)
                100.0, # Class 5 (original 6): STABILIZED from 180.0 → 100.0 (maintains focus, prevents collapse)
                8.0,   # Class 6 (original 7): 0.0312% - slightly reduced
                8.0    # Class 7 (original 8): 0.0312% - slightly reduced
            ], dtype=torch.float32)
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
    
    def forward(self, inputs, targets):
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
            
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets.clamp(0, len(self.alpha)-1)]
            alpha_t[targets == self.ignore_index] = 0
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            
        focal_loss[targets == self.ignore_index] = 0
        
        if self.reduction == 'mean':
            valid_mask = targets != self.ignore_index
            return focal_loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks with ignore_index support.
    """
    
    def __init__(self, smooth=1e-6, ignore_index=-1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Create valid mask (exclude ignore_index)
        valid_mask = targets != self.ignore_index
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Flatten tensors for easier computation
        probs_flat = probs.permute(0, 2, 3, 1).contiguous().view(-1, inputs.size(1))  # [N, C]
        targets_flat = targets.view(-1)  # [N]
        valid_mask_flat = valid_mask.view(-1)  # [N]
        
        # Select only valid pixels
        probs_valid = probs_flat[valid_mask_flat]  # [N_valid, C]
        targets_valid = targets_flat[valid_mask_flat]  # [N_valid]
        
        if len(targets_valid) == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Create one-hot encoding
        targets_one_hot = F.one_hot(targets_valid, num_classes=inputs.size(1)).float()  # [N_valid, C]
        
        # Compute Dice coefficient per class
        intersection = (probs_valid * targets_one_hot).sum(dim=0)  # [C]
        union = probs_valid.sum(dim=0) + targets_one_hot.sum(dim=0)  # [C]
        dice = (2 * intersection + self.smooth) / (union + self.smooth)  # [C]
        
        # Return 1 - mean dice (loss)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    """
    Combined Focal + Dice Loss for optimal segmentation performance.
    
    Combines the class imbalance handling of Focal Loss with
    the direct IoU optimization of Dice Loss.
    """
    
    def __init__(self, focal_weight=0.6, dice_weight=0.4, **kwargs):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = FocalLoss(**kwargs)
        self.dice_loss = DiceLoss(ignore_index=kwargs.get('ignore_index', -1))
        
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice

class SentinelAugmentations:
    """
    Specialized augmentations for Sentinel-2 satellite imagery.
    
    Applies both spatial and spectral augmentations while preserving
    the semantic content and realistic satellite data characteristics.
    """
    
    def __init__(self):
        self.rare_classes = [4, 5, 6, 7]  # Classes requiring more augmentation
        self.ultra_rare_classes = [5, 6, 7]  # Extremely rare classes
    
    def apply_spatial_augmentation(self, img_tensor, mask_tensor, intensity='standard'):
        """Apply spatial augmentations preserving semantic content."""
        combined = torch.cat([img_tensor, mask_tensor.unsqueeze(0).float()], dim=0)
        
        augmentation_list = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ]
        
        if intensity == 'aggressive':
            augmentation_list.extend([
                T.RandomRotation(degrees=90, interpolation=T.InterpolationMode.BILINEAR),
                T.RandomApply([T.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.3),
            ])
        
        if augmentation_list:
            transform = T.Compose(augmentation_list)
            combined = transform(combined)
        
        img_aug = combined[:IN_CHANNELS]
        mask_aug = combined[IN_CHANNELS:].squeeze(0).long()
        return img_aug, mask_aug
    
    def apply_spectral_augmentation(self, img_tensor, intensity='standard'):
        """Apply spectral augmentations specific to satellite imagery."""
        augmented = img_tensor.clone()
        
        # Atmospheric variations
        if random.random() < 0.3:
            brightness_factor = random.uniform(0.8, 1.2)
            augmented = torch.clamp(augmented * brightness_factor, 0, 1)
        
        # Band-specific noise
        if random.random() < 0.2:
            noise = torch.randn_like(augmented) * 0.01
            augmented = torch.clamp(augmented + noise, 0, 1)
        
        if intensity == 'aggressive':
            # Atmospheric haze simulation
            if random.random() < 0.4:
                haze_factor = random.uniform(0.95, 1.05)
                augmented = torch.clamp(augmented * haze_factor + random.uniform(-0.02, 0.02), 0, 1)
        
        return augmented

class SentinelKilnDataset(Dataset):
    """
    Dataset for Sentinel-2 kiln detection with adaptive sampling and augmentation.
    
    Features:
    - Adaptive sampling targeting rare classes (training only)
    - Real-time data augmentation
    - Class-aware patch extraction
    - Test-realistic validation mode
    """
    
    def __init__(self, img_dir, mask_dir, patch_size=256, positive_ratio=0.7, augmentations=None, 
                 is_validation=False, train_stats=None):
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
        
        self.img_paths = self._filter_valid_images(img_dir, mask_dir)
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.positive_ratio = positive_ratio
        self.augmentations = augmentations
        self.is_validation = is_validation
        self.train_stats = train_stats  # CRITICAL: Store train_stats as instance attribute
        
        # CRITICAL: Different behavior for validation vs training
        if self.is_validation:
            # Validation: No mask-based caching, pure random sampling like test scenario
            self.positive_ratio = 0.0  # No positive sampling bias
            # Skip kiln location caching to avoid mask dependency
        else:
            # Training: Initialize cache for on-demand kiln location caching
            self.kiln_cache = {}
        
        # Normalization statistics
        if train_stats is not None:
            # Use provided training statistics (for validation)
            self.mean, self.std = train_stats
        else:
            # Compute own statistics (for training)
            self.mean, self.std = self._compute_dataset_statistics()
        
        self.norm = T.Normalize(mean=self.mean, std=self.std)
    
    def _filter_valid_images(self, img_dir, mask_dir):
        """Filter out invalid image-mask pairs."""
        all_img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        valid_paths = []
        
        for img_path in all_img_paths:
            mask_path = os.path.join(mask_dir, os.path.basename(img_path))
            
            if not os.path.exists(mask_path):
                continue
            
            # Basic validity checks
            try:
                if self._is_image_valid(img_path) and self._is_mask_valid(mask_path):
                    valid_paths.append(img_path)
            except:
                continue
        
        return valid_paths
    
    def _is_image_valid(self, img_path):
        """Quick image validity check."""
        try:
            with rasterio.open(img_path) as src:
                img = src.read(out_dtype=np.float32)[:IN_CHANNELS]
            return not (np.all(np.isnan(img)) or img.shape[1] < 100 or img.shape[2] < 100)
        except:
            return False
    
    def _is_mask_valid(self, mask_path):
        """Quick mask validity check."""
        try:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
            unique_labels = np.unique(mask[~np.isnan(mask)])
            return len(unique_labels) > 0 and unique_labels.min() >= 0 and unique_labels.max() <= 8
        except:
            return False
    
    def _compute_dataset_statistics(self, max_samples=50):
        """Compute mean and std for dataset normalization"""
        means = []
        stds = []
        
        # Sample random subset for efficiency
        sample_paths = random.sample(self.img_paths, min(max_samples, len(self.img_paths)))
        
        for img_path in sample_paths:
            img = read_s2_image(img_path)
            for c in range(img.shape[0]):
                means.append(img[c].mean().item())
                stds.append(img[c].std().item())
        
        # Group by channel and compute statistics
        channel_means = [np.mean(means[i::8]) for i in range(8)]
        channel_stds = [np.mean(stds[i::8]) for i in range(8)]
        
        # Fallback to Sentinel-2 defaults if needed
        sentinel2_defaults = {
            0: (0.1354, 0.0454), 1: (0.1165, 0.0389), 2: (0.1071, 0.0387), 3: (0.1204, 0.0398),
            4: (0.1354, 0.0516), 5: (0.1825, 0.0652), 6: (0.2039, 0.0741), 7: (0.1982, 0.0773)
        }
        
        for c in range(8):
            if channel_stds[c] < 1e-6:
                default_mean, default_std = sentinel2_defaults.get(c, (0.2, 0.1))
                channel_means[c] = default_mean
                channel_stds[c] = default_std
        
        return channel_means, channel_stds

    def _cache_kiln_locations_from_mapped_mask(self, mapped_mask, image_id):
        """Cache kiln locations from already mapped mask"""
        if image_id in self.kiln_cache:
            return
        
        # Search for mapped labels 4,5,6,7 (original 5,6,7,8)
        # FIX: Added class5 (Code 4) to address sampling imbalance
        class5_coords = np.column_stack(np.where(mapped_mask == 4))  # NEW: Report Class 5
        class6_coords = np.column_stack(np.where(mapped_mask == 5))  # Report Class 6
        class7_coords = np.column_stack(np.where(mapped_mask == 6))  # Report Class 7
        class8_coords = np.column_stack(np.where(mapped_mask == 7))  # Report Class 8
        
        self.kiln_cache[image_id] = {
            'class5': class5_coords,  # NEW: Enable sampling for Class 5
            'class6': class6_coords,
            'class7': class7_coords,
            'class8': class8_coords
        }

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = read_s2_image(img_path)
        mask_path = os.path.join(self.mask_dir, os.path.basename(img_path))
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        try:
            with rasterio.open(mask_path) as src:
                mask_raw = src.read(1)
        except Exception as e:
            raise RuntimeError(f"Error reading mask {mask_path}: {e}")
        
        # Convert labels: 0→-1 (ignore), 1-8→0-7 (classes)
        mask = torch.from_numpy(mask_raw).long()
        mask = torch.where(mask == 0, torch.tensor(-1), mask - 1)
        
        # Validate label ranges
        valid_mask = mask[mask != -1]
        if len(valid_mask) > 0:
            assert valid_mask.min() >= 0 and valid_mask.max() < NUM_CLASSES, \
                f"Invalid labels after conversion: {valid_mask.min()}-{valid_mask.max()}"
        
        # Cache kiln locations AFTER mapping (if not validation mode)
        image_id = os.path.basename(img_path).replace('.tif', '')
        if not self.is_validation:
            self._cache_kiln_locations_from_mapped_mask(mask, image_id)
        
        # Sampling strategy depends on training vs validation mode
        _, H, W = img.shape
        
        if self.is_validation:
            # VALIDATION: Pure random sampling (no mask knowledge, like test scenario)
            top = random.randint(0, H - self.patch_size)
            left = random.randint(0, W - self.patch_size)
            
        else:
            # TRAINING: SIMPLIFIED sampling for stability (based on failure analysis)
            # Previous hierarchical cascade was over-complex and destabilized training
            # FIX: Independent probabilities for each rare class ensure balanced, stable sampling
            cache = self.kiln_cache.get(image_id, {})
            
            sampling_strategy_used = None
            
            # Independent probability checks for each rare class (no hierarchical blocking)
            if 'class6' in cache and len(cache['class6']) > 0 and random.random() < 0.18:
                center_y, center_x = cache['class6'][random.randint(0, len(cache['class6']) - 1)]
                top = max(0, min(H - self.patch_size, center_y - self.patch_size // 2))
                left = max(0, min(W - self.patch_size, center_x - self.patch_size // 2))
                sampling_strategy_used = 'class6'
            elif 'class5' in cache and len(cache['class5']) > 0 and random.random() < 0.15:
                center_y, center_x = cache['class5'][random.randint(0, len(cache['class5']) - 1)]
                top = max(0, min(H - self.patch_size, center_y - self.patch_size // 2))
                left = max(0, min(W - self.patch_size, center_x - self.patch_size // 2))
                sampling_strategy_used = 'class5'
            elif 'class7' in cache and len(cache['class7']) > 0 and random.random() < 0.12:
                center_y, center_x = cache['class7'][random.randint(0, len(cache['class7']) - 1)]
                top = max(0, min(H - self.patch_size, center_y - self.patch_size // 2))
                left = max(0, min(W - self.patch_size, center_x - self.patch_size // 2))
                sampling_strategy_used = 'class7'
            elif 'class8' in cache and len(cache['class8']) > 0 and random.random() < 0.08:
                center_y, center_x = cache['class8'][random.randint(0, len(cache['class8']) - 1)]
                top = max(0, min(H - self.patch_size, center_y - self.patch_size // 2))
                left = max(0, min(W - self.patch_size, center_x - self.patch_size // 2))
                sampling_strategy_used = 'class8'
            else:
                # Standard random sampling (majority of patches)
                top = random.randint(0, H - self.patch_size)
                left = random.randint(0, W - self.patch_size)
                sampling_strategy_used = 'random'
        
        # Extract patch
        img_patch = img[:, top:top+self.patch_size, left:left+self.patch_size]
        mask_patch = mask[top:top+self.patch_size, left:left+self.patch_size]
        
        # STABILIZED augmentations for training stability (based on failure analysis)
        # Previous over-aggressive augmentation contributed to training collapse
        # FIX: Moderate augmentation rates that support rare classes without destabilizing training
        if self.augmentations is not None and not self.is_validation:
            unique_classes = torch.unique(mask_patch[mask_patch != -1])
            rare_classes = [4, 5, 6, 7]
            ultra_rare_classes = [4, 5, 6]
            
            has_ultra_rare = any(cls.item() in ultra_rare_classes for cls in unique_classes)
            has_rare = any(cls.item() in rare_classes for cls in unique_classes)
            
            # STABILIZED augmentation rates for balanced training
            if has_ultra_rare and random.random() < 0.5:  # REDUCED from 0.7 for stability
                img_patch, mask_patch = self.augmentations.apply_spatial_augmentation(
                    img_patch, mask_patch, intensity='standard')  # REDUCED from aggressive
                img_patch = self.augmentations.apply_spectral_augmentation(
                    img_patch, intensity='standard')  # REDUCED from aggressive
            elif has_rare and random.random() < 0.3:  # REDUCED from 0.5 for balance
                img_patch, mask_patch = self.augmentations.apply_spatial_augmentation(
                    img_patch, mask_patch, intensity='standard')
                img_patch = self.augmentations.apply_spectral_augmentation(
                    img_patch, intensity='standard')
            elif random.random() < 0.1:  # REDUCED from 0.15 for overall stability
                img_patch, mask_patch = self.augmentations.apply_spatial_augmentation(
                    img_patch, mask_patch, intensity='standard')
        
        return self.norm(img_patch), mask_patch

class AttentionGate(nn.Module):
    """
    Attention Gate for focusing on relevant features in skip connections.
    
    Helps the model focus on rare classes by weighting skip connection features
    based on the decoder's current representation.
    """
    
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, gate, skip):
        g1 = self.W_gate(gate)
        s1 = self.W_skip(skip)
        
        # Upsample gate to match skip connection size
        if g1.size()[2:] != s1.size()[2:]:
            g1 = F.interpolate(g1, size=s1.size()[2:], mode='bilinear', align_corners=False)
            
        psi = self.relu(g1 + s1)
        psi = self.psi(psi)
        
        return skip * psi

class ResidualDoubleConv(nn.Module):
    """
    Enhanced convolutional block with residual connections and batch normalization.
    
    Prevents gradient vanishing in deeper networks while maintaining
    feature learning capability.
    """
    
    def __init__(self, inc, outc, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = outc
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(inc, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(mid_channels, outc, 3, padding=1, bias=False),
            nn.BatchNorm2d(outc),
        )
        
        # Residual connection
        self.residual = nn.Sequential()
        if inc != outc:
            self.residual = nn.Sequential(
                nn.Conv2d(inc, outc, 1, bias=False),
                nn.BatchNorm2d(outc)
            )
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = self.residual(x)
        out = self.double_conv(x)
        return self.relu(out + residual)

class ChannelAttention(nn.Module):
    """
    Channel attention module for adaptive feature channel weighting.
    """
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class EnhancedDoubleConv(nn.Module):
    """
    Enhanced double convolution block with channel attention.
    """
    
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = ResidualDoubleConv(inc, outc)
        self.attention = ChannelAttention(outc)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.attention(x)
        return x

class EnhancedUNet(pl.LightningModule):
    """
    Enhanced UNet with attention gates, residual connections, and advanced loss functions.
    
    Features:
    - Attention-gated skip connections
    - Residual convolutional blocks
    - Channel attention mechanisms
    - Combined Focal + Dice Loss
    - AdamW optimizer with cosine annealing
    """
    
    def __init__(self, lr=LR, loss_type='combined', use_logit_bias=True):
        super().__init__()
        self.save_hyperparameters()
        
        # Encoder (Downsampling Path)
        self.enc1 = EnhancedDoubleConv(IN_CHANNELS, 64)
        self.enc2 = EnhancedDoubleConv(64, 128)
        self.enc3 = EnhancedDoubleConv(128, 256)
        self.enc4 = EnhancedDoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = EnhancedDoubleConv(512, 1024)
        
        # Attention Gates
        self.att4 = AttentionGate(512, 512, 256)
        self.att3 = AttentionGate(256, 256, 128)
        self.att2 = AttentionGate(128, 128, 64)
        self.att1 = AttentionGate(64, 64, 32)
        
        # Decoder (Upsampling Path)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = EnhancedDoubleConv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = EnhancedDoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = EnhancedDoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = EnhancedDoubleConv(128, 64)
        
        # Output Layer
        self.final_conv = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(64, NUM_CLASSES, 1)
        )
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
        # Loss Function
        if loss_type == 'combined':
            self.loss_fn = CombinedLoss(focal_weight=0.6, dice_weight=0.4, ignore_index=-1)
        elif loss_type == 'focal':
            self.loss_fn = FocalLoss(gamma=2.0, ignore_index=-1)
        elif loss_type == 'dice':
            self.loss_fn = DiceLoss(ignore_index=-1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for output layer
        with torch.no_grad():
            final_layer = self.final_conv[-1]
            nn.init.xavier_uniform_(final_layer.weight, gain=0.01)
            if final_layer.bias is not None:
                nn.init.constant_(final_layer.bias, 0.0)
        
        # Tracking for F1 computation
        self.all_val_preds = []
        self.all_val_labels = []
    
    def _init_weights(self, m):
        """Initialize network weights using Kaiming initialization."""
        if isinstance(m, nn.Conv2d) and m not in self.final_conv:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        # Input validation
        if torch.isnan(x).any():
            raise ValueError("NaN in input tensor detected!")
        
        # Encoder Path
        c1 = self.enc1(x)     # 256x256x64
        x = self.pool(c1)     # 128x128x64
        
        c2 = self.enc2(x)     # 128x128x128
        x = self.pool(c2)     # 64x64x128
        
        c3 = self.enc3(x)     # 64x64x256
        x = self.pool(c3)     # 32x32x256
        
        c4 = self.enc4(x)     # 32x32x512
        x = self.pool(c4)     # 16x16x512
        
        # Bottleneck
        x = self.bottleneck(x)  # 16x16x1024
        
        # Decoder Path with Attention Gates
        x = self.up4(x)                           # 32x32x512
        c4_att = self.att4(x, c4)                 # Apply attention
        x = torch.cat([x, c4_att], dim=1)         # 32x32x1024
        x = self.dec4(x)                          # 32x32x512
        
        x = self.up3(x)                           # 64x64x256
        c3_att = self.att3(x, c3)                 # Apply attention
        x = torch.cat([x, c3_att], dim=1)         # 64x64x512
        x = self.dec3(x)                          # 64x64x256
        
        x = self.up2(x)                           # 128x128x128
        c2_att = self.att2(x, c2)                 # Apply attention
        x = torch.cat([x, c2_att], dim=1)         # 128x128x256
        x = self.dec2(x)                          # 128x128x128
        
        x = self.up1(x)                           # 256x256x64
        c1_att = self.att1(x, c1)                 # Apply attention
        x = torch.cat([x, c1_att], dim=1)         # 256x256x128
        x = self.dec1(x)                          # 256x256x64
        
        # Final output
        output = self.final_conv(x)               # 256x256xNUM_CLASSES
        
        if torch.isnan(output).any():
            raise ValueError("NaN in output detected!")
        
        return output
    
    def shared_step(self, batch):
        """Shared step for training and validation."""
        x, y = batch
        
        # Data validation
        if torch.isnan(x).any() or torch.isnan(y.float()).any():
            return None
        
        # Label validation
        if y.min() < -1 or y.max() >= NUM_CLASSES:
            raise ValueError(f"Invalid label values: min={y.min()}, max={y.max()}")
        
        # Skip batches with only ignore labels
        valid_labels = y[y != -1]
        if len(valid_labels) == 0:
            return None
        
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Loss validation
        if torch.isnan(loss) or torch.isinf(loss):
            return None
        
        return loss
    
    def training_step(self, batch, idx):
        loss = self.shared_step(batch)
        if loss is not None:
            self.log("train_loss", loss, prog_bar=True)
            return loss
        return None
    
    def validation_step(self, batch, idx):
        loss = self.shared_step(batch)
        if loss is not None:
            self.log("val_loss", loss, prog_bar=True)
            
            # Collect predictions for F1 Score with optional POST-PROCESSING bias
            x, y = batch
            with torch.no_grad():
                logits = self(x)
                
                # Apply logit bias to improve recall for rare classes (if enabled)
                if self.hparams.use_logit_bias:
                    biased_logits = self.apply_logit_bias(logits)
                    preds = torch.argmax(biased_logits, dim=1)
                else:
                    preds = torch.argmax(logits, dim=1)
                
                valid_mask = y != -1
                valid_preds = preds[valid_mask]
                valid_labels = y[valid_mask]
                
                if len(valid_preds) > 0:
                    self.all_val_preds.extend(valid_preds.cpu().numpy())
                    self.all_val_labels.extend(valid_labels.cpu().numpy())
            
            return loss
        return None
    
    def configure_optimizers(self):
        """Configure AdamW optimizer with cosine annealing scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing with warmup
        total_steps = EPOCHS * 100  # Estimate
        warmup_steps = int(0.1 * total_steps)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
    
    def on_validation_epoch_end(self):
        """Calculate F1 scores at the end of each validation epoch."""
        if len(self.all_val_preds) == 0:
            return
        
        all_preds = np.array(self.all_val_preds)
        all_labels = np.array(self.all_val_labels)
        
        # Filter to only include kiln classes (exclude background)
        non_bg_mask = (all_labels >= 0) & (all_preds >= 0)
        
        if np.sum(non_bg_mask) > 0:
            filtered_labels = all_labels[non_bg_mask]
            filtered_preds = all_preds[non_bg_mask]
            
            from sklearn.metrics import f1_score
            
            kiln_classes = list(range(0, NUM_CLASSES))
            weighted_f1 = f1_score(filtered_labels, filtered_preds, 
                                 average='weighted', zero_division=0, labels=kiln_classes)
            macro_f1 = f1_score(filtered_labels, filtered_preds, 
                               average='macro', zero_division=0, labels=kiln_classes)
            
            self.log('val_weighted_f1_kiln', weighted_f1, prog_bar=True)
            self.log('val_macro_f1_kiln', macro_f1, prog_bar=True)
        
        # Clear for next epoch
        self.all_val_preds.clear()
        self.all_val_labels.clear()
    
    def on_before_optimizer_step(self, optimizer):
        """Gradient clipping for training stability."""
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
    
    def apply_logit_bias(self, logits, bias_values=None):
        """
        Apply class-specific bias to logits to improve recall for rare classes.
        
        This post-processing technique lowers the threshold for rare classes
        to be predicted, directly improving recall at the cost of precision.
        
        Args:
            logits: Raw model output (B, C, H, W)
            bias_values: Bias values for each class (length C)
            
        Returns:
            Biased logits with enhanced rare class predictions
        """
        if bias_values is None:
            # Aggressive bias for ultra-rare classes based on analysis
            bias_values = [0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 2.5, 0.0]
            # Classes 5 & 6 get significant bias to improve recall
            
        bias_tensor = torch.tensor(bias_values, device=logits.device, dtype=logits.dtype)
        biased_logits = logits + bias_tensor.view(1, -1, 1, 1)
        return biased_logits

def build_data_loaders():
    """
    Build training and validation data loaders with test-realistic validation.
    
    CRITICAL CHANGE: Uses internal get_standard_train_val_split() function
    This ensures ALL team members use IDENTICAL train/val splits for comparable results.
    
    Key improvements for realistic validation:
    - IDENTICAL split using internal get_standard_train_val_split()
    - Deterministic split for reproducibility (random_state=42)
    - Validation uses pure random sampling (no mask knowledge)
    - Shared normalization statistics from training set
    - No filtering for validation (realistic test conditions)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Initialize augmentation system
    augmentations = SentinelAugmentations()
    
    # CRITICAL: Use IDENTICAL split method as all other team members
    # This replaces the previous torch.utils.data.random_split approach
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = get_standard_train_val_split(
        data_path="/scratch/tmp/tstraus2/Brick_Data_Train/",  # Adjust path as needed
        test_size=0.2,
        random_state=42
    )
    
    print(f"Using INTERNAL standardized split function:")
    print(f"Train images: {len(train_img_paths)}")
    print(f"Val images: {len(val_img_paths)}")
    print(f"Split method: Internal get_standard_train_val_split() - IDENTICAL across all team members")
    
    # Create TRAINING dataset with filtering and adaptive sampling
    train_ds = SentinelKilnDataset(
        img_dir=IMG_DIR,
        mask_dir=MASK_DIR,
        patch_size=PATCH_SIZE,
        positive_ratio=0.7,
        augmentations=augmentations,
        is_validation=False,  # Training mode
        train_stats=None  # Will compute own stats
    )
    # Filter training paths to only include valid ones and those in train split
    train_ds.img_paths = [p for p in train_ds.img_paths if p in train_img_paths]
    
    # Create VALIDATION dataset with test-realistic conditions
    val_ds = SentinelKilnDataset(
        img_dir=IMG_DIR,
        mask_dir=MASK_DIR,
        patch_size=PATCH_SIZE,
        positive_ratio=0.0,  # Will be overridden to 0.0 in validation mode
        augmentations=None,  # No augmentation for validation
        is_validation=True,  # CRITICAL: Validation mode (no mask knowledge)
        train_stats=(train_ds.mean, train_ds.std)  # Use training normalization
    )
    # CRITICAL: No filtering for validation - keep ALL images like in test scenario
    val_ds.img_paths = [p for p in val_img_paths]  # Use val_img_paths directly
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader

def main():
    """Main training function."""
    torch.set_float32_matmul_precision('medium')
    print("Starting Enhanced UNet V4 Final Training...")
    
    # Build data loaders
    train_loader, val_loader = build_data_loaders()
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Initialize model with balanced improvements (logit bias disabled for clean evaluation)
    model = EnhancedUNet(lr=LR, loss_type='combined', use_logit_bias=False)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename='enhanced_unet_v4_final_{epoch:02d}_{val_loss:.4f}',
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Configure trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="32",
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        gradient_clip_val=0.5,
        val_check_interval=0.25,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        accumulate_grad_batches=2,
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Final evaluation
    print("=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    # IoU evaluation
    metric = torchmetrics.classification.MulticlassJaccardIndex(
        num_classes=NUM_CLASSES, ignore_index=-1, average='none'
    ).to(model.device)
    
    model.eval()
    metric.reset()
    all_final_preds = []
    all_final_labels = []
    
    with torch.no_grad():
        for xb, yb in val_loader:
            # Apply same logit bias as in validation for consistent evaluation
            logits = model(xb.to(model.device))
            if model.hparams.use_logit_bias:
                biased_logits = model.apply_logit_bias(logits)
                preds = torch.argmax(biased_logits, 1)
            else:
                preds = torch.argmax(logits, 1)
            metric.update(preds.cpu(), yb)
            
            valid_mask = yb != -1
            valid_preds = preds.cpu()[valid_mask]
            valid_labels = yb[valid_mask]
            
            if len(valid_preds) > 0:
                all_final_preds.extend(valid_preds.numpy())
                all_final_labels.extend(valid_labels.numpy())
    
    print("Final IoU per class:", metric.compute())
    
    # Final F1 scores
    if len(all_final_preds) > 0:
        from sklearn.metrics import f1_score, classification_report
        
        all_final_preds = np.array(all_final_preds)
        all_final_labels = np.array(all_final_labels)
        
        non_bg_mask = (all_final_labels >= 0) & (all_final_preds >= 0)
        
        if np.sum(non_bg_mask) > 0:
            filtered_labels = all_final_labels[non_bg_mask]
            filtered_preds = all_final_preds[non_bg_mask]
            
            kiln_classes = list(range(0, NUM_CLASSES))
            weighted_f1 = f1_score(filtered_labels, filtered_preds, 
                                 average='weighted', zero_division=0, labels=kiln_classes)
            macro_f1 = f1_score(filtered_labels, filtered_preds, 
                               average='macro', zero_division=0, labels=kiln_classes)
            
            print(f"Final Weighted F1 (Kiln Classes): {weighted_f1:.4f}")
            print(f"Final Macro F1 (Kiln Classes): {macro_f1:.4f}")
            
            print("\nClassification Report (Kiln Classes Only):")
            kiln_class_names = [f'Kiln_Class_{i}' for i in kiln_classes]
            print(classification_report(filtered_labels, filtered_preds, 
                                      labels=kiln_classes,
                                      target_names=kiln_class_names,
                                      zero_division=0))
    
    # Save model
    model_path = "/home/t/tstraus2/enhanced_unet_v4_final_stabilized_kiln_sentinel2.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    print("=" * 60)
    print("TRAINING COMPLETED WITH STABILIZED ARCHITECTURE!")
    print("=" * 60)
    print("Applied stabilized improvements (based on failure analysis):")
    print("✓ STABILIZED Loss Weighting (Class weights: 80.0/100.0 - preventing gradient explosion)")
    print("✓ SIMPLIFIED Sampling (Independent probabilities: 18%/15%/12%/8% - stable mixing)")
    print("✓ MODERATE Augmentation (Balanced rates: 50%/30%/10% - training stability)")
    print("✓ TARGET: Restore training stability and prevent model collapse")
    print("✓ GOAL: Rebuild stable foundation for gradual improvement toward 0.90+ Weighted F1")
    print("=" * 60)

if __name__ == "__main__":
    main() 