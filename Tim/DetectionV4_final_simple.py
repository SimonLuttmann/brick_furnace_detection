#!/usr/bin/env python3
"""
Simple UNet V4 - BALANCED-AGGRESSIVE for Rare Class Recall
==========================================================

Vereinfachte UNet-Architektur für Kiln-Detektion in Sentinel-2 Satellitendaten
mit Fokus auf Verständlichkeit und BALANCED-AGGRESSIVEN Optimierungen für seltene Klassen.

⚡ BALANCED-AGGRESSIVE IMPROVEMENTS APPLIED (Proactive Anti-Regression Strategy):
1. BALANCED Loss Weighting: Class 4 increased to 300.0 (from 200.0), Class 5 maintained at 400.0
2. HIERARCHICAL Sampling: Class5 30%, Class4 25%, Other rare 15% (prevent Class 5 regression)
3. MAXIMUM Augmentation: 90% rate for ultra-rare classes (4&5) vs 50% for others (maintained)

PROBLEM ADDRESSED (Proactive Prevention of "Robbing-Peter-to-pay-Paul"):
- Anticipated: Class 6 improvement may come at cost of Class 5 regression
- Strategy: Strengthen BOTH critical classes (4&5) simultaneously through balanced optimization
- Goal: Achieve high F1 scores for Classes 4 AND 5 without sacrificing one for the other
- Target: Push Weighted F1 toward 0.9 through balanced rare class optimization

VEREINFACHUNGEN gegenüber DetectionV4_final.py:
- Standard UNet ohne Attention Gates und Residual Connections
- Nur Focal Loss (keine Combined Loss) - BUT with aggressive alpha weights
- ReduceLROnPlateau Scheduler (kein komplexer Cosine Annealing)
- Simplified but TARGETED augmentation for rare classes

BEIBEHALTENE KERNFUNKTIONEN:
- Focal Loss für extreme class imbalance (with aggressive weights)
- AdamW Optimizer für stabile Konvergenz
- AGGRESSIVE Sampling für seltene Klassen
- Test-realistische Validation
- Robuste Datenaufbereitung

ARCHITEKTUR:
- Standard UNet mit Batch Normalization
- Focal Loss mit aggressive alpha weights für class imbalance handling
- AdamW optimizer mit ReduceLROnPlateau scheduler

DATASET:
- Input: 8-channel Sentinel-2 imagery (10m resolution bands)
- Classes: 8 kiln types (labels 1-8 → 0-7) + background (0 → -1 ignore)
- Extreme imbalance: rare classes <0.01% of pixels
- Patch size: 256x256 pixels

TRAINING STRATEGY:
- AGGRESSIVE sampling targeting ultra-rare classes (50% probability)
- TARGETED data augmentation (80% for ultra-rare classes)
- Gradient clipping for training stability

VALIDATION APPROACH:
- Test-realistic conditions: pure random sampling, no mask knowledge
- Shared normalization statistics from training set
- No data filtering (all validation images included)
- Deterministic splitting for reproducible results
"""

import os, glob, random, zipfile, getpass
import numpy as np
import torch, rasterio
import torchvision.transforms as T
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
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
    
    # Use sklearn.train_test_split for consistency
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
            # BALANCED-AGGRESSIVE Alpha weights (proactive anti-regression strategy)
            # Strategy: Prevent "Robbing-Peter-to-pay-Paul" by strengthening BOTH critical classes
            # Goal: Achieve high performance for Classes 4 AND 5 simultaneously
            self.alpha = torch.tensor([
                10.0,  # Class 0 (original 1): 0.0263%
                8.0,   # Class 1 (original 2): 0.0379%
                25.0,  # Class 2 (original 3): 0.0117%
                60.0,  # Class 3 (original 4): 0.0010%
                300.0, # Class 4 (original 5): INCREASED from 200.0 → 300.0 (Report Class 5 - prevent regression!)
                400.0, # Class 5 (original 6): MAINTAINED at 400.0 (Report Class 6 - preserve gains!)
                1.0,   # Class 6 (original 7): moderate
                10.0   # Class 7 (original 8): 0.0312%
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

class SimpleAugmentations:
    """
    Vereinfachte Augmentierungen für Sentinel-2 Satellitendaten.
    
    Wendet einheitliche Augmentierungen an (keine klassenspezifische Logik).
    """
    
    def __init__(self):
        pass
    
    def apply_spatial_augmentation(self, img_tensor, mask_tensor):
        """Apply spatial augmentations preserving semantic content."""
        combined = torch.cat([img_tensor, mask_tensor.unsqueeze(0).float()], dim=0)
        
        # Standard spatial augmentations
        augmentation_list = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=90, interpolation=T.InterpolationMode.BILINEAR),
        ]
        
        # Apply with 50% probability
        if random.random() < 0.5:
            transform = T.Compose(augmentation_list)
            combined = transform(combined)
        
        img_aug = combined[:IN_CHANNELS]
        mask_aug = combined[IN_CHANNELS:].squeeze(0).long()
        return img_aug, mask_aug
    
    def apply_spectral_augmentation(self, img_tensor):
        """Apply spectral augmentations specific to satellite imagery."""
        augmented = img_tensor.clone()
        
        # Brightness variations
        if random.random() < 0.3:
            brightness_factor = random.uniform(0.8, 1.2)
            augmented = torch.clamp(augmented * brightness_factor, 0, 1)
        
        # Gaussian noise
        if random.random() < 0.2:
            noise = torch.randn_like(augmented) * 0.01
            augmented = torch.clamp(augmented + noise, 0, 1)
        
        return augmented

class SentinelKilnDataset(Dataset):
    """
    Dataset für Sentinel-2 kiln detection mit vereinfachter Augmentierung.
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
        self.train_stats = train_stats
        
        if self.is_validation:
            # Validation: No mask-based caching, pure random sampling
            self.positive_ratio = 0.0
        else:
            # Training: Initialize cache for kiln locations
            self.kiln_cache = {}
        
        # Normalization statistics
        if train_stats is not None:
            self.mean, self.std = train_stats
        else:
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
        """Cache kiln locations from already mapped mask - BALANCED for critical classes"""
        if image_id in self.kiln_cache:
            return
        
        # FIX: Separate caching for critical classes to enable balanced sampling
        # Classes 4 and 5 need special treatment to prevent regression
        class4_coords = np.column_stack(np.where(mapped_mask == 4))  # Report Class 5
        class5_coords = np.column_stack(np.where(mapped_mask == 5))  # Report Class 6
        
        # Other rare classes for fallback
        other_rare_coords = []
        for class_id in [6, 7]:  # Less critical rare classes
            coords = np.column_stack(np.where(mapped_mask == class_id))
            if len(coords) > 0:
                other_rare_coords.extend(coords)
        
        self.kiln_cache[image_id] = {
            'class4': class4_coords,  # Critical: Report Class 5
            'class5': class5_coords,  # Critical: Report Class 6
            'other_rare': other_rare_coords  # Fallback
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
        
        # Cache kiln locations for adaptive sampling (training only)
        image_id = os.path.basename(img_path).replace('.tif', '')
        if not self.is_validation:
            self._cache_kiln_locations_from_mapped_mask(mask, image_id)
        
        # Sampling strategy
        _, H, W = img.shape
        
        if self.is_validation:
            # VALIDATION: Pure random sampling
            top = random.randint(0, H - self.patch_size)
            left = random.randint(0, W - self.patch_size)
        else:
            # TRAINING: BALANCED-AGGRESSIVE sampling preventing Class 5 regression
            # FIX: Separate sampling logic for critical classes 4 and 5
            cache = self.kiln_cache.get(image_id, {})
            
            # Priority 1: Class 5 (Report Class 6) - Most critical, maintain high priority
            use_class5_sampling = (random.random() < 0.30 and  # 30% chance
                                  'class5' in cache and len(cache['class5']) > 0)
            
            # Priority 2: Class 4 (Report Class 5) - Prevent regression, NEW priority
            use_class4_sampling = (not use_class5_sampling and
                                  random.random() < 0.25 and  # 25% chance if Class 5 not chosen
                                  'class4' in cache and len(cache['class4']) > 0)
            
            # Priority 3: Other rare classes - Fallback
            use_other_rare_sampling = (not use_class5_sampling and not use_class4_sampling and
                                      random.random() < 0.15 and  # 15% chance
                                      'other_rare' in cache and len(cache['other_rare']) > 0)
            
            # Apply sampling strategy
            if use_class5_sampling:
                center_y, center_x = cache['class5'][random.randint(0, len(cache['class5']) - 1)]
                top = max(0, min(H - self.patch_size, center_y - self.patch_size // 2))
                left = max(0, min(W - self.patch_size, center_x - self.patch_size // 2))
            elif use_class4_sampling:
                center_y, center_x = cache['class4'][random.randint(0, len(cache['class4']) - 1)]
                top = max(0, min(H - self.patch_size, center_y - self.patch_size // 2))
                left = max(0, min(W - self.patch_size, center_x - self.patch_size // 2))
            elif use_other_rare_sampling:
                center_y, center_x = cache['other_rare'][random.randint(0, len(cache['other_rare']) - 1)]
                top = max(0, min(H - self.patch_size, center_y - self.patch_size // 2))
                left = max(0, min(W - self.patch_size, center_x - self.patch_size // 2))
            else:
                # Random sampling (~30% of the time)
                top = random.randint(0, H - self.patch_size)
                left = random.randint(0, W - self.patch_size)
        
        # Extract patch
        img_patch = img[:, top:top+self.patch_size, left:left+self.patch_size]
        mask_patch = mask[top:top+self.patch_size, left:left+self.patch_size]
        
        # ULTRA-AGGRESSIVE AUGMENTIERUNG: Maximale Intensität für ultra-seltene Klassen
        if self.augmentations is not None and not self.is_validation:
            unique_classes = torch.unique(mask_patch[mask_patch != -1])
            
            # Classes 4 and 5 in code correspond to problematic classes 5 and 6 in report
            is_ultra_rare_patch = any(cls.item() in [4, 5] for cls in unique_classes)
            
            # MAXIMUM augmentation frequency for ultra-rare patches: 80% → 90%
            apply_aug = (is_ultra_rare_patch and random.random() < 0.9) or \
                       (not is_ultra_rare_patch and random.random() < 0.5)
            
            if apply_aug:
                img_patch, mask_patch = self.augmentations.apply_spatial_augmentation(
                    img_patch, mask_patch)
                img_patch = self.augmentations.apply_spectral_augmentation(img_patch)
        
        return self.norm(img_patch), mask_patch

class DoubleConv(nn.Module):
    """
    Standard Double Convolution Block.
    
    Einfacher Standard-Block ohne Residual Connections oder Attention.
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SimpleUNet(pl.LightningModule):
    """
    Vereinfachtes UNet für Kiln Detection.
    
    Standard UNet-Architektur ohne komplexe Features:
    - Keine Attention Gates
    - Keine Residual Connections
    - Nur Focal Loss
    - ReduceLROnPlateau Scheduler
    """
    
    def __init__(self, lr=LR):
        super().__init__()
        self.save_hyperparameters()
        
        # Encoder (Downsampling Path)
        self.enc1 = DoubleConv(IN_CHANNELS, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder (Upsampling Path)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = DoubleConv(128, 64)
        
        # Output Layer
        self.final_conv = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(64, NUM_CLASSES, 1)
        )
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
        # VEREINFACHTE LOSS: Nur Focal Loss
        self.loss_fn = FocalLoss(gamma=2.0, ignore_index=-1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tracking for F1 computation
        self.all_val_preds = []
        self.all_val_labels = []
    
    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
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
        
        # Decoder Path (Standard U-Net ohne Attention Gates)
        x = self.up4(x)                           # 32x32x512
        x = torch.cat([x, c4], dim=1)             # 32x32x1024 (direkte Verkettung)
        x = self.dec4(x)                          # 32x32x512
        
        x = self.up3(x)                           # 64x64x256
        x = torch.cat([x, c3], dim=1)             # 64x64x512
        x = self.dec3(x)                          # 64x64x256
        
        x = self.up2(x)                           # 128x128x128
        x = torch.cat([x, c2], dim=1)             # 128x128x256
        x = self.dec2(x)                          # 128x128x128
        
        x = self.up1(x)                           # 256x256x64
        x = torch.cat([x, c1], dim=1)             # 256x256x128
        x = self.dec1(x)                          # 256x256x64
        
        # Final output
        output = self.final_conv(x)               # 256x256xNUM_CLASSES
        
        return output
    
    def shared_step(self, batch):
        """Shared step for training and validation."""
        x, y = batch
        
        # Data validation
        if torch.isnan(x).any() or torch.isnan(y.float()).any():
            return None
        
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
            
            # Collect predictions for F1 Score
            x, y = batch
            with torch.no_grad():
                logits = self(x)
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
        """VEREINFACHTER SCHEDULER: ReduceLROnPlateau statt komplexem Cosine Annealing."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Einfacher, verständlicher Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,    # Reduziert LR um 80%
            patience=5,    # Wartet 5 Epochen
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Überwacht die Validierungs-Loss
                "interval": "epoch",
            },
        }
    
    def on_validation_epoch_end(self):
        """Calculate F1 scores at the end of each validation epoch."""
        if len(self.all_val_preds) == 0:
            return
        
        all_preds = np.array(self.all_val_preds)
        all_labels = np.array(self.all_val_labels)
        
        # Filter to only include kiln classes
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

def build_data_loaders():
    """
    Build training and validation data loaders.
    
    Uses same standardized split as DetectionV4_final.py but with simplified dataset.
    """
    # Vereinfachte Augmentierung
    augmentations = SimpleAugmentations()
    
    # Standardisierter Split
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = get_standard_train_val_split(
        data_path=DATA_ROOT,
        test_size=0.2,
        random_state=42
    )
    
    print(f"Using standardized split:")
    print(f"Train images: {len(train_img_paths)}")
    print(f"Val images: {len(val_img_paths)}")
    
    # Create datasets
    train_ds = SentinelKilnDataset(
        img_dir=IMG_DIR,
        mask_dir=MASK_DIR,
        patch_size=PATCH_SIZE,
        positive_ratio=0.7,
        augmentations=augmentations,
        is_validation=False,
        train_stats=None
    )
    train_ds.img_paths = [p for p in train_ds.img_paths if p in train_img_paths]
    
    val_ds = SentinelKilnDataset(
        img_dir=IMG_DIR,
        mask_dir=MASK_DIR,
        patch_size=PATCH_SIZE,
        positive_ratio=0.0,
        augmentations=None,
        is_validation=True,
        train_stats=(train_ds.mean, train_ds.std)
    )
    val_ds.img_paths = [p for p in val_img_paths]
    
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
    print("Starting Simple UNet V4 Training...")
    
    # Build data loaders
    train_loader, val_loader = build_data_loaders()
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Initialize model
    model = SimpleUNet(lr=LR)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename='simple_unet_v4_{epoch:02d}_{val_loss:.4f}',
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
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
            preds = torch.argmax(model(xb.to(model.device)), 1)
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
    model_path = os.path.join(os.path.expanduser("~"), "simple_unet_v4_balanced_aggressive_kiln_sentinel2.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    print("=" * 60)
    print("SIMPLE UNet V4 TRAINING COMPLETED WITH BALANCED-AGGRESSIVE RARE CLASS OPTIMIZATIONS!")
    print("=" * 60)
    print("Applied balanced-aggressive improvements (proactive anti-regression strategy):")
    print("✓ BALANCED Loss Weighting (Class 4: 200.0→300.0, Class 5: 400.0 maintained)")
    print("✓ HIERARCHICAL Sampling (Class5: 30%, Class4: 25%, Other rare: 15%)")
    print("✓ MAXIMUM Augmentation (90% for ultra-rare classes - maintained)")
    print("✓ STRATEGY: Strengthen BOTH critical classes to prevent Class 5 regression")
    print("✓ TARGET: Achieve high F1 for Classes 4 AND 5 without sacrificing one for the other")
    print("✓ GOAL: Push Weighted F1 toward 0.9 through balanced rare class optimization")
    print("=" * 60)

if __name__ == "__main__":
    main() 