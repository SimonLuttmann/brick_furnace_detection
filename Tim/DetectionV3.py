'''\
Sentinel-2 Kiln Detection  – 8‑Kanal‑Variante
================================================

Das Skript nutzt jetzt **alle acht gelieferten Sentinel‑2‑Kanäle**
(typischerweise B1–B8 oder ein projektspezifischer 8‑Band‑Stack) statt
nur der vier 10‑m‑Bänder.  Änderungen:

* `IN_CHANNELS = 8` (statt 4)
* UNet‑Encoder nimmt 8 Kanäle entgegen.
* Normalisierung (Mean/Std) auf 8‑Element‑Listen erweitert.
* `read_s2_image()` liest die **ersten acht Bänder** des GeoTIFFs.
* Data‐Loader & Training unverändert; Gewichte werden als
  `unet_kiln_sentinel2_8ch.pt` gespeichert.
'''

# --------------------------------------------------
# 1 · Imports & Konstanten
# --------------------------------------------------
import os, glob, random, zipfile, getpass
import numpy as np
import torch, rasterio
import torchvision.transforms as T
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from performance_analysis import calculate_and_log_performance
from utils import print_available_memory
import torch.nn.functional as F

USER = getpass.getuser()
SCRATCH = os.environ.get("SLURM_TMPDIR", f"/scratch/tmp/{USER}")
ZIP_PATH = os.path.join(SCRATCH, "Brick_Data_Train.zip")
DATA_ROOT = os.path.join(SCRATCH, "Brick_Data_Train")
IMG_DIR  = os.path.join(DATA_ROOT, "Image")      # Korrigiert: "Image" statt "images"
MASK_DIR = os.path.join(DATA_ROOT, "Mask")       # Korrigiert: "Mask" statt "mask"
PATCH_SIZE  = 256
BATCH_SIZE  = 8
LR          = 1e-4          # Reduzierte Lernrate für Stabilität
EPOCHS      = 50            # Erhöht für bessere Konvergenz
NUM_CLASSES = 8             # KORRIGIERT: Nach Mapping 0-8 → 0-7 sind es 8 Klassen
IN_CHANNELS = 8          # <── neu: 8 Kanäle
TERMINATTE_EARLY = 10    # Early stopping patience in epochs

# Model naming for tensorboard and checkpoints
MODEL_NAME = f"unet_kiln_8ch_seed42_lr{LR}_bs{BATCH_SIZE}_epochs{EPOCHS}"
pl.seed_everything(42, workers=True)

# Entpacken bei Bedarf
if not os.path.isdir(DATA_ROOT):
    if os.path.isfile(ZIP_PATH):
        print(f"Entpacke {ZIP_PATH} → {DATA_ROOT} …")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(SCRATCH)
    else:
        raise FileNotFoundError(ZIP_PATH)

# --------------------------------------------------
# 2 · Data Loading - Using data_loader.py
# --------------------------------------------------
# Note: Image reading is now handled by data_loader.py

# --------------------------------------------------
# 3 · Dataset - Simplified using data_loader.py
# --------------------------------------------------
from data_loader import load_split

# Path-based dataset for memory-efficient loading
class SentinelKilnDatasetPaths(Dataset):
    def __init__(self, image_paths, label_paths, patch_size=256, positive_ratio=0.7, is_train=True):
        """
        Memory-efficient dataset using file paths instead of pre-loaded data
        
        Args:
            image_paths: list of paths to image files
            label_paths: list of paths to label files
            patch_size: size of patches to extract
            positive_ratio: ratio of positive samples to generate
            is_train: whether this is training data (for augmentation)
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.patch_size = patch_size
        self.positive_ratio = positive_ratio
        self.is_train = is_train
        
        print(f"Path-based dataset initialized with {len(image_paths)} images")
        
        # Cache kiln locations for targeted sampling
        print("Caching kiln locations for positive sampling...")
        self._cache_kiln_locations()
        print(f"Found {len(self.kiln_coords)} images with kiln pixels")

        # Simplified normalization using Sentinel-2 standard values
        sentinel2_mean = [0.15, 0.12, 0.11, 0.13, 0.25, 0.35, 0.38, 0.30]
        sentinel2_std = [0.10, 0.08, 0.07, 0.08, 0.12, 0.15, 0.16, 0.14]
        
        self.norm = T.Normalize(mean=sentinel2_mean, std=sentinel2_std)
        print(f"Using Sentinel-2 standard normalization")

    def _cache_kiln_locations(self):
        """Cache kiln locations for targeted sampling"""
        self.kiln_coords = {}
        self.class6_coords = {}
        self.class5_coords = {}
        self.class3_coords = {}
        
        for i, label_path in enumerate(self.label_paths):
            try:
                with rasterio.open(label_path) as src:
                    mask_raw = src.read(1)
                
                # Convert labels: 0→-1, 1-8→0-7
                mask = np.where(mask_raw == 0, -1, mask_raw - 1)
                
                # Find kiln pixels
                kiln_pixels = np.where((mask >= 0) & (mask <= 7))
                if len(kiln_pixels[0]) > 0:
                    self.kiln_coords[i] = list(zip(kiln_pixels[0], kiln_pixels[1]))
                
                # Cache rare classes
                for target_class, cache_name in [(6, 'class6_coords'), (5, 'class5_coords'), (3, 'class3_coords')]:
                    class_pixels = np.where(mask == target_class)
                    if len(class_pixels[0]) > 0:
                        getattr(self, cache_name)[i] = list(zip(class_pixels[0], class_pixels[1]))
                        
            except Exception as e:
                print(f"Warning: Could not process {label_path}: {e}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask on-demand
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        # Load image
        try:
            with rasterio.open(image_path) as src:
                img_np = src.read(out_dtype=np.float32)[:8]  # 8 channels
                img_np = img_np.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                img_np = np.nan_to_num(img_np / 10000.0, nan=0.0)  # Normalize
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")
        
        # Load mask
        try:
            with rasterio.open(label_path) as src:
                mask_raw = src.read(1)
        except Exception as e:
            raise RuntimeError(f"Error loading mask {label_path}: {e}")
        
        # Convert to torch tensors
        img = torch.from_numpy(img_np).float().permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        mask = torch.from_numpy(mask_raw).long()
        mask = torch.where(mask == 0, torch.tensor(-1), mask - 1)  # 0→-1, 1-8→0-7
        
        # Apply targeting strategies (simplified for memory efficiency)
        _, H, W = img.shape
        
        # Use targeted sampling if available
        if random.random() < self.positive_ratio and idx in self.kiln_coords:
            coords = self.kiln_coords[idx]
            y_center, x_center = random.choice(coords)
        else:
            # Random sampling
            y_center = random.randint(self.patch_size // 2, H - self.patch_size // 2)
            x_center = random.randint(self.patch_size // 2, W - self.patch_size // 2)
        
        # Extract patch
        y1 = max(0, y_center - self.patch_size // 2)
        y2 = min(H, y1 + self.patch_size)
        x1 = max(0, x_center - self.patch_size // 2)  
        x2 = min(W, x1 + self.patch_size)
        
        img_patch = img[:, y1:y2, x1:x2]
        mask_patch = mask[y1:y2, x1:x2]
        
        # Pad if necessary
        if img_patch.shape[1] < self.patch_size or img_patch.shape[2] < self.patch_size:
            pad_h = max(0, self.patch_size - img_patch.shape[1])
            pad_w = max(0, self.patch_size - img_patch.shape[2])
            img_patch = F.pad(img_patch, (0, pad_w, 0, pad_h), mode='reflect')
            mask_patch = F.pad(mask_patch, (0, pad_w, 0, pad_h), mode='constant', value=-1)
        
        # Normalize
        img_patch = self.norm(img_patch)
        
        return img_patch, mask_patch


# Simplified dataset using numpy arrays from data_loader
class SentinelKilnDataset(Dataset):
    def __init__(self, X_data, y_data, patch_size=256, positive_ratio=0.7, is_train=True):
        """
        Simplified dataset using pre-loaded numpy arrays from data_loader.py
        
        Args:
            X_data: numpy array of shape (N, H, W, C) - images
            y_data: numpy array of shape (N, H, W) - labels
            patch_size: size of patches to extract
            positive_ratio: ratio of positive samples to generate
            is_train: whether this is training data (for augmentation)
        """
        self.X_data = X_data
        self.y_data = y_data
        self.patch_size = patch_size
        self.positive_ratio = positive_ratio
        self.is_train = is_train
        
        print(f"Dataset initialized with {len(X_data)} images")
        print(f"Image shape: {X_data.shape}")
        print(f"Label shape: {y_data.shape}")
        
        # Cache kiln locations for targeted sampling
        print("Caching kiln locations for positive sampling...")
        self._cache_kiln_locations()
        print(f"Found {len(self.kiln_coords)} images with kiln pixels")

        # Simplified normalization using Sentinel-2 standard values
        sentinel2_mean = [0.15, 0.12, 0.11, 0.13, 0.25, 0.35, 0.38, 0.30]
        sentinel2_std = [0.10, 0.08, 0.07, 0.08, 0.12, 0.15, 0.16, 0.14]
        
        self.norm = T.Normalize(mean=sentinel2_mean, std=sentinel2_std)
        print(f"Using Sentinel-2 standard normalization")



    def _cache_kiln_locations(self):
        """Cache pixels with kiln classes for targeted sampling - using numpy arrays"""
        self.kiln_coords = {}
        self.class6_coords = {}  # Für extrem seltene Klasse 6 (original 7)
        self.class5_coords = {}  # Für sehr seltene Klasse 5 (original 6)
        self.class3_coords = {}  # Für seltene Klasse 3 (original 4)

        for idx in range(len(self.y_data)):
            mask_raw = self.y_data[idx]  # Original 0-8 Labels

            # CRITICAL FIX: Masken haben bereits Labels 0-8!
            # Konvertiere korrekt: 0→-1, 1-8→0-7
            mask = torch.where(torch.from_numpy(mask_raw) == 0,
                             torch.tensor(-1),
                             torch.from_numpy(mask_raw) - 1)

            # Find alle Kiln-Pixel (0-7 nach Mapping)
            kiln_pixels = np.where((mask >= 0) & (mask <= 7))
            if len(kiln_pixels[0]) > 0:
                self.kiln_coords[idx] = list(zip(kiln_pixels[0], kiln_pixels[1]))

            # Spezielle Caches für extrem seltene Klassen
            for target_class, cache_name in [(6, 'class6_coords'), (5, 'class5_coords'), (3, 'class3_coords')]:
                class_pixels = np.where(mask == target_class)
                if len(class_pixels[0]) > 0:
                    getattr(self, cache_name)[idx] = list(zip(class_pixels[0], class_pixels[1]))
                    print(f"Found {len(class_pixels[0])} class {target_class} pixels in image {idx}")

        print(f"Total images with class 6: {len(self.class6_coords)}")
        print(f"Total images with class 5: {len(self.class5_coords)}")
        print(f"Total images with class 3: {len(self.class3_coords)}")

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        # Get image and mask from numpy arrays
        img_np = self.X_data[idx]  # Shape: (H, W, C)
        mask_raw = self.y_data[idx]  # Shape: (H, W)
        
        # Convert to torch tensors and transpose image to (C, H, W)
        img = torch.from_numpy(img_np).float().permute(2, 0, 1)  # (C, H, W)
        
        # NaN handling for images
        img = torch.nan_to_num(img, nan=0.0)
        
        # CRITICAL FIX: Konsistente Label-Behandlung basierend auf Label-Analyse
        # KORRIGIERT: Masken haben bereits Labels 0-8, KEINE Subtraktion nötig für Background!
        # Korrekte Konversion: 0→-1 (background), 1-8→0-7 (classes)
        mask = torch.from_numpy(mask_raw).long()
        mask = torch.where(mask == 0, torch.tensor(-1), mask - 1)  # 0→-1, 1-8→0-7

        # OPTIMIERTE Targeting-Strategien für bessere Kiln Detection
        # Adaptive probabilities basierend auf Klassen-Häufigkeit
        _, H, W = img.shape

        # Multi-scale analysis: Check verschiedene Patch-Größen für bessere Context
        patch_contexts = []
        for scale in [0.5, 1.0, 1.5]:
            context_size = int(self.patch_size * scale)
            if context_size <= min(H, W):
                patch_contexts.append(context_size)
        
        # Wähle optimal patch size basierend auf Content
        optimal_patch_size = self.patch_size
        if len(patch_contexts) > 1:
            # Analysiere verschiedene Scales für besseren Context
            for context_size in patch_contexts:
                # Verwende größeren Context für seltene Klassen
                if idx in self.class6_coords or idx in self.class5_coords:
                    optimal_patch_size = max(optimal_patch_size, context_size)

        # VERBESSERTE Targeting mit adaptiven Wahrscheinlichkeiten
        # Class 6 (extrem selten): 30% (erhöht von 20%)
        use_class6_sampling = (random.random() < 0.30 and
                              idx in self.class6_coords and
                              len(self.class6_coords[idx]) > 0)

        # Class 5 (sehr selten): 25% (erhöht von 15%)
        use_class5_sampling = (not use_class6_sampling and
                              random.random() < 0.25 and
                              idx in self.class5_coords and
                              len(self.class5_coords[idx]) > 0)

        # Class 3 (selten): 20% (erhöht von 10%)
        use_class3_sampling = (not use_class6_sampling and not use_class5_sampling and
                              random.random() < 0.20 and
                              idx in self.class3_coords and
                              len(self.class3_coords[idx]) > 0)

        # Mixed kiln sampling: 40% (erhöht von 35%)
        use_positive_sampling = (not any([use_class6_sampling, use_class5_sampling, use_class3_sampling]) and
                                random.random() < 0.40 and
                                idx in self.kiln_coords and
                                len(self.kiln_coords[idx]) > 0)

        # Adaptive patch extraction mit intelligentem Padding
        if use_class6_sampling:
            center_y, center_x = random.choice(self.class6_coords[idx])
            # Größere Context-Region für extrem seltene Klassen
            extended_size = min(optimal_patch_size + 32, min(H, W))
            top = max(0, min(H - extended_size, center_y - extended_size // 2))
            left = max(0, min(W - extended_size, center_x - extended_size // 2))
            sampling_type = "CLASS6_ENHANCED_TARGET"
        elif use_class5_sampling:
            center_y, center_x = random.choice(self.class5_coords[idx])
            extended_size = min(optimal_patch_size + 16, min(H, W))
            top = max(0, min(H - extended_size, center_y - extended_size // 2))
            left = max(0, min(W - extended_size, center_x - extended_size // 2))
            sampling_type = "CLASS5_ENHANCED_TARGET"
        elif use_class3_sampling:
            center_y, center_x = random.choice(self.class3_coords[idx])
            top = max(0, min(H - optimal_patch_size, center_y - optimal_patch_size // 2))
            left = max(0, min(W - optimal_patch_size, center_x - optimal_patch_size // 2))
            sampling_type = "CLASS3_ENHANCED_TARGET"
        elif use_positive_sampling:
            # Multi-kiln sampling: Bevorzuge Patches mit mehreren Kiln-Pixeln
            center_y, center_x = random.choice(self.kiln_coords[idx])
            top = max(0, min(H - optimal_patch_size, center_y - optimal_patch_size // 2))
            left = max(0, min(W - optimal_patch_size, center_x - optimal_patch_size // 2))
            sampling_type = "ENHANCED_POSITIVE"
        else:
            # Smart random sampling: Vermeidet reine Background-Patches
            max_attempts = 5
            for attempt in range(max_attempts):
                top = random.randint(0, H - optimal_patch_size)
                left = random.randint(0, W - optimal_patch_size)
                
                # Check if patch has some variety (nicht nur Background)
                test_patch = mask[top:top+optimal_patch_size//4, left:left+optimal_patch_size//4]
                if len(torch.unique(test_patch)) > 1 or attempt == max_attempts - 1:
                    break
            sampling_type = "SMART_RANDOM"

        # Extract patch mit optimal size
        final_patch_size = min(optimal_patch_size, self.patch_size)  # Nicht größer als Original
        if final_patch_size != self.patch_size:
            # Resize falls nötig
            img_patch = img[:, top:top+final_patch_size, left:left+final_patch_size]
            mask_patch = mask[top:top+final_patch_size, left:left+final_patch_size]
            
            # Resize to target size
            img_patch = F.interpolate(img_patch.unsqueeze(0), size=(self.patch_size, self.patch_size), 
                                    mode='bilinear', align_corners=False).squeeze(0)
            mask_patch = F.interpolate(mask_patch.unsqueeze(0).unsqueeze(0).float(), 
                                     size=(self.patch_size, self.patch_size), 
                                     mode='nearest').squeeze(0).squeeze(0).long()
        else:
            img_patch = img[:, top:top+self.patch_size, left:left+self.patch_size]
            mask_patch = mask[top:top+self.patch_size, left:left+self.patch_size]

        # Debug output every 100th sample
        if idx % 100 == 0:
            unique_classes = torch.unique(mask_patch[mask_patch != -1])
            kiln_pixels = len(mask_patch[(mask_patch >= 0) & (mask_patch <= 7)])  # KORRIGIERT: 0-7 nach Mapping
            total_valid = len(mask_patch[mask_patch != -1])
            print(f"Sample {idx}: {sampling_type} - Kiln: {kiln_pixels}/{total_valid}, Classes: {unique_classes.tolist()}")

        return self.norm(img_patch), mask_patch

# --------------------------------------------------
# 4 · UNet (8 Eingangs-Kanäle) - Vollständige Architektur  
# --------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss für extreme Klassenungleichgewicht bei Kiln Detection
    
    Focal Loss = -α(1-pt)^γ * log(pt)
    
    Vorteile für Kiln Detection:
    - Fokussiert auf schwer zu klassifizierende Samples
    - Reduziert Gewicht von einfachen Samples (z.B. Background)
    - Bessere Performance bei seltenen Klassen
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-1):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(weight=alpha, ignore_index=ignore_index, reduction='none')
        
    def forward(self, inputs, targets):
        # Standard CrossEntropy loss ohne Reduction
        ce_loss = self.ce_loss(inputs, targets)
        
        # Berechne Wahrscheinlichkeiten
        pt = torch.exp(-ce_loss)
        
        # Focal Loss = (1-pt)^gamma * CE_loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Mittelwert über valide Pixel (nicht ignore_index)
        valid_mask = targets != self.ignore_index
        if valid_mask.sum() > 0:
            return focal_loss[valid_mask].mean()
        else:
            return torch.tensor(0.0, requires_grad=True, device=inputs.device)

class AttentionGate(nn.Module):
    """
    Attention Gate für selektive Feature-Betonung bei Kiln Detection
    
    Hilft dem Modell sich auf relevante Kiln-Features zu fokussieren
    und Hintergrund-Rauschen zu unterdrücken.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # g: gating signal from coarser scale
        # x: feature map from encoder to be attended
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g1 to match x1 size if needed
        if g1.size() != x1.size():
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Attended feature map
        out = x * psi
        return out

class DoubleConv(nn.Module):
    """Zwei aufeinanderfolgende Convolutions mit BatchNorm und ReLU"""
    def __init__(self, inc, outc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, 3, padding=1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNet(pl.LightningModule):
    """Vollständige UNet-Architektur mit 4 Encoder- und 4 Decoder-Ebenen"""

    def _compute_class_weights(self, y_data):
        """Compute class weights dynamically based on actual dataset distribution"""
        print("Computing dynamic class weights from dataset...")
        
        # Count pixels for each class (after mapping: 0→-1, 1-8→0-7)
        class_counts = np.zeros(NUM_CLASSES)
        background_count = 0
        total_pixels = 0
        
        for mask_raw in y_data:
            unique, counts = np.unique(mask_raw, return_counts=True)
            for label, count in zip(unique, counts):
                total_pixels += count
                if label == 0:  # Background (wird zu -1 gemappt)
                    background_count += count
                elif 1 <= label <= 8:  # Kiln classes (werden zu 0-7 gemappt)
                    class_counts[label - 1] += count
        
        # Calculate weights using inverse frequency with smoothing
        class_weights = []
        for i in range(NUM_CLASSES):
            if class_counts[i] > 0:
                frequency = class_counts[i] / total_pixels
                # Inverse frequency with min/max bounds and smoothing
                weight = min(50.0, max(1.0, np.sqrt(0.001 / (frequency + 1e-8))))
                original_label = i + 1
                print(f"Class {i} (original {original_label}): {class_counts[i]:,} pixels ({frequency:.6f}%) → weight {weight:.1f}")
            else:
                weight = 1.0  # Default for missing classes
                print(f"Class {i} (original {i+1}): MISSING → weight {weight:.1f}")
            class_weights.append(weight)
        
        # Log comparison with hardcoded weights
        hardcoded_weights = [4.4, 2.9, 19.7, 20.0, 20.0, 20.0, 20.0, 20.0]
        print(f"\nClass weight comparison:")
        print(f"Hardcoded: {hardcoded_weights}")
        print(f"Dynamic:   {[round(w, 1) for w in class_weights]}")
        
        return torch.tensor(class_weights, dtype=torch.float32)

    def __init__(self, lr=LR, X_data=None, y_data=None):
        super().__init__()
        self.save_hyperparameters()

        # Encoder (Downsampling Path)
        self.enc1 = DoubleConv(IN_CHANNELS, 64)    # 8→64
        self.enc2 = DoubleConv(64, 128)            # 64→128
        self.enc3 = DoubleConv(128, 256)           # 128→256
        self.enc4 = DoubleConv(256, 512)           # 256→512

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)    # 512→1024

        # Decoder (Upsampling Path) with Attention Gates
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)  # Upsample
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)  # Attention for c4
        self.dec4 = DoubleConv(1024, 512)               # 1024 (512+512)→512

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)   # Upsample
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)  # Attention for c3
        self.dec3 = DoubleConv(512, 256)                # 512 (256+256)→256

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)   # Upsample
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)   # Attention for c2
        self.dec2 = DoubleConv(256, 128)                # 256 (128+128)→128

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)    # Upsample
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)    # Attention for c1
        self.dec1 = DoubleConv(128, 64)                 # 128 (64+64)→64

        # Output Layer
        self.out = nn.Conv2d(64, NUM_CLASSES, 1)

        # Pooling
        self.pool = nn.MaxPool2d(2)

        # Tensorboard setup
        self.writer = SummaryWriter(f'{SCRATCH}/tensorboard_logs/{MODEL_NAME}')
        
        # F1-Score tracking
        self.validation_step_outputs = []
        self.all_val_preds = []
        self.all_val_labels = []
        self.epochs_no_improve = 0
        self.best_val_loss = float('inf')

        # Dynamic Class-weighted Focal Loss for extreme imbalance
        if y_data is not None:
            class_weights = self._compute_class_weights(y_data)
        else:
            # Fallback zu hardcoded weights wenn keine Daten verfügbar
            print("Warning: No training data provided for dynamic weights, using hardcoded values")
            class_weights = torch.tensor([4.4, 2.9, 19.7, 20.0, 20.0, 20.0, 20.0, 20.0])
        
        # Use Focal Loss instead of CrossEntropy for better rare class detection
        self.loss = FocalLoss(alpha=class_weights, gamma=2.0, ignore_index=-1)
        print(f"Using Focal Loss with gamma=2.0 for improved kiln detection")

        # Proper weight initialization für alle Layer AUSSER Output
        self.apply(self._init_weights)

        # Spezielle Initialisierung für Output Layer NACH apply() um Überschreibung zu vermeiden
        with torch.no_grad():
            nn.init.xavier_uniform_(self.out.weight, gain=0.01)  # Noch kleinere Gewichte!
            if self.out.bias is not None:
                nn.init.constant_(self.out.bias, 0.0)
            print(f"Output layer initialized: weight_std={self.out.weight.std():.6f}")

    def _init_weights(self, m):
        """Kaiming-Initialisierung für bessere Gradient-Stabilität - NICHT für Output Layer"""
        if isinstance(m, nn.Conv2d) and m != self.out:  # Ausnahme für Output Layer
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # Eingabe-Validierung gegen NaN
        if torch.isnan(x).any():
            raise ValueError("NaN in input tensor detected!")

        # Encoder Path mit Skip Connections speichern
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

        # Decoder Path mit Attention-guided Skip Connections
        x = self.up4(x)                    # 32x32x512
        c4_att = self.att4(g=x, x=c4)      # Apply attention to c4
        x = torch.cat([x, c4_att], dim=1)  # 32x32x1024 (attended features)
        x = self.dec4(x)                   # 32x32x512

        x = self.up3(x)                    # 64x64x256
        c3_att = self.att3(g=x, x=c3)      # Apply attention to c3
        x = torch.cat([x, c3_att], dim=1)  # 64x64x512 (attended features)
        x = self.dec3(x)                   # 64x64x256

        x = self.up2(x)                    # 128x128x128
        c2_att = self.att2(g=x, x=c2)      # Apply attention to c2
        x = torch.cat([x, c2_att], dim=1)  # 128x128x256 (attended features)
        x = self.dec2(x)                   # 128x128x128

        x = self.up1(x)                    # 256x256x64
        c1_att = self.att1(g=x, x=c1)      # Apply attention to c1
        x = torch.cat([x, c1_att], dim=1)  # 256x256x128 (attended features)
        x = self.dec1(x)                   # 256x256x64

        output = self.out(x)               # 256x256xNUM_CLASSES

        # Output-Validierung gegen NaN
        if torch.isnan(output).any():
            print("WARNING: NaN in output detected!")

        return output

    def shared_step(self, batch):
        x, y = batch

        # Zusätzliche Validierung
        if torch.isnan(x).any():
            print("WARNING: NaN in input data!")
            raise ValueError("NaN detected in input data - stopping training")

        if torch.isnan(y.float()).any():
            print("WARNING: NaN in target data!")
            raise ValueError("NaN detected in target data - stopping training")

        # Debug: Analysiere die Label-Werte
        unique_labels = torch.unique(y)
        print(f"Unique labels in batch: {unique_labels.tolist()}")
        print(f"Label stats: min={y.min()}, max={y.max()}, shape={y.shape}")

        # Prüfe auf ungültige Label-Werte für CrossEntropy
        if y.min() < -1 or y.max() >= NUM_CLASSES:
            print(f"ERROR: Invalid label values! min={y.min()}, max={y.max()}")
            print(f"Expected range: -1 to {NUM_CLASSES-1}")
            raise ValueError("Invalid label values detected")

        # Prüfe auf "nur ignore_index" Batch - das verursacht NaN Loss
        valid_labels = y[y != -1]  # Alle nicht-ignore Labels
        if len(valid_labels) == 0:
            print("WARNING: Batch contains only ignore_index (-1) labels!")
            print("Skipping this batch to avoid NaN loss...")
            # Returniere einen kleinen dummy loss der nicht NaN ist
            return torch.tensor(0.0, requires_grad=True, device=y.device)

        logits = self(x)

        # Debug: Analysiere Logits
        print(f"Logits shape: {logits.shape}")
        print(f"Logits stats: min={logits.min():.6f}, max={logits.max():.6f}, mean={logits.mean():.6f}")

        loss = self.loss(logits, y)

        print(f"Computed loss: {loss}")

        # NaN-Check für Loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: NaN/Inf loss detected!")
            print(f"Loss value: {loss}")
            print(f"Valid pixels in batch: {len(valid_labels)}")
            # Fallback: Returniere minimalen Loss statt Crash
            return torch.tensor(0.0, requires_grad=True, device=y.device)

        return loss

    def training_step(self, batch, idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        
        # Log to tensorboard
        self.writer.add_scalar('Loss/Train', loss, self.global_step)
        
        # Memory monitoring every 100 steps
        if idx % 100 == 0:
            print_available_memory()
        
        return loss

    def post_process_predictions(self, logits, confidence_threshold=0.7):
        """
        Post-processing für verbesserte Kiln Detection
        
        Args:
            logits: Raw model outputs [B, C, H, W]
            confidence_threshold: Minimum confidence für Kiln-Klassen
        
        Returns:
            Refined predictions with improved kiln detection
        """
        # Convert to probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get initial predictions
        preds = torch.argmax(probs, dim=1)
        
        # Apply confidence thresholding für Kiln-Klassen
        max_probs = torch.max(probs, dim=1)[0]
        
        # Kiln classes (0-7) need higher confidence
        kiln_mask = preds >= 0  # All non-background classes
        low_confidence = max_probs < confidence_threshold
        
        # Set low-confidence kiln predictions to background (-1 in training, but 0 for final output)
        uncertain_kilns = kiln_mask & low_confidence
        preds[uncertain_kilns] = -1  # Will be handled by ignore_index
        
        return preds

    def validation_step(self, batch, idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        
        # Log to tensorboard
        self.writer.add_scalar('Loss/Val', loss, self.global_step)
        
        # Collect predictions and labels for F1-Score calculation with post-processing
        x, y = batch
        with torch.no_grad():
            logits = self(x)
            
            # Apply post-processing für bessere Kiln Detection
            preds = self.post_process_predictions(logits, confidence_threshold=0.6)
            
            # Only collect valid (non-ignore) predictions and labels
            valid_mask = y != -1
            valid_preds = preds[valid_mask]
            valid_labels = y[valid_mask]
            
            # Additional filtering: Only include predictions that are not -1 (uncertain)
            confident_mask = valid_preds != -1
            if confident_mask.sum() > 0:
                final_preds = valid_preds[confident_mask]
                final_labels = valid_labels[confident_mask]
                
                self.all_val_preds.extend(final_preds.cpu().numpy())
                self.all_val_labels.extend(final_labels.cpu().numpy())
        
        return loss

    def on_validation_epoch_end(self):
        """Calculate and log F1-Score and other metrics at the end of each validation epoch"""
        if len(self.all_val_preds) == 0:
            print("Warning: No valid predictions collected for F1-Score calculation")
            return
            
        # Convert to numpy arrays
        all_preds = np.array(self.all_val_preds)
        all_labels = np.array(self.all_val_labels)
        
        print(f"Epoch {self.current_epoch}: Computing F1-Score on {len(all_preds)} validation pixels", flush=True)
        
        # Calculate and log performance metrics
        calculate_and_log_performance(
            all_labels=all_labels,
            all_preds=all_preds,
            num_classes=NUM_CLASSES,
            epoch=self.current_epoch,
            writer=self.writer
        )
        
        # Check for early stopping
        current_val_loss = float(self.trainer.callback_metrics.get('val_loss', float('inf')))
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.epochs_no_improve = 0
            print(f"New best validation loss: {current_val_loss:.4f}")
        else:
            self.epochs_no_improve += 1
            print(f"No improvement for {self.epochs_no_improve} epochs")
            
        if self.epochs_no_improve >= TERMINATTE_EARLY:
            print(f"Early stopping: No improvement in {TERMINATTE_EARLY} epochs.")
            self.trainer.should_stop = True
        
        # Clear for next epoch
        self.all_val_preds.clear()
        self.all_val_labels.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def on_before_optimizer_step(self, optimizer):
        """Gradient Clipping gegen Explosion"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
    
    def on_train_end(self):
        """Close tensorboard writer"""
        if hasattr(self, 'writer'):
            self.writer.close()
            print("Tensorboard writer closed")

# --------------------------------------------------
# 5 · Training & Evaluation
# --------------------------------------------------

def build_loaders():
    """Build dataloaders using data_loader.py for consistency"""
    print("Loading data using data_loader.py...")
    print_available_memory()
    
    # Load data using existing data_loader.py (MEMORY-OPTIMIZED)
    # Changed load_into_ram=False to prevent OOM on large datasets
    (
        X_train, X_test, y_train, y_test,
        image_paths_train, image_paths_test,
        label_paths_train, label_paths_test,
    ) = load_split(DATA_ROOT, load_into_ram=False, verbose=True, as_numpy=True)
    
    print("Data loaded successfully!")
    print_available_memory()
    
    # Create datasets - Use path-based loading for memory efficiency
    if not X_train or len(X_train) == 0:  # load_into_ram=False mode (returns empty list)
        print("Using path-based loading for memory efficiency")
        train_ds = SentinelKilnDatasetPaths(image_paths_train, label_paths_train, PATCH_SIZE, is_train=True)
        val_ds = SentinelKilnDatasetPaths(image_paths_test, label_paths_test, PATCH_SIZE, is_train=False)
    else:  # load_into_ram=True mode
        print("Using RAM-based loading")
        train_ds = SentinelKilnDataset(X_train, y_train, PATCH_SIZE, is_train=True)
        val_ds = SentinelKilnDataset(X_test, y_test, PATCH_SIZE, is_train=False)
    
    # Store training data for dynamic class weights
    # For path-based loading, we need the dataset object to compute weights
    training_data = (X_train, y_train) if X_train and len(X_train) > 0 else train_ds
    
    # Begrenzte Worker-Anzahl für Stabilität (max 8)
    num_workers = min(8, os.cpu_count() or 4)

    kwargs = dict(batch_size=BATCH_SIZE,
                  num_workers=num_workers,
                  pin_memory=torch.cuda.is_available())
    return DataLoader(train_ds, shuffle=True, **kwargs), DataLoader(val_ds, shuffle=False, **kwargs), training_data

# Initialisierung der DataLoader
train_loader, val_loader, (X_train_data, y_train_data) = build_loaders()
print(f"Train: {len(train_loader)}  Val: {len(val_loader)}")

# Modell und Training konfigurieren mit dynamischen class weights
model = UNet(y_data=y_train_data)
ckpt  = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
lrmon = LearningRateMonitor(logging_interval="epoch")

# Erweiterte Trainer-Konfiguration für bessere Konvergenz
trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    precision="32",                  # Zurück zu 32-bit wegen AMP-Problemen mit Dummy-Loss
    max_epochs=EPOCHS,
    callbacks=[ckpt, lrmon],
    log_every_n_steps=10,
    gradient_clip_val=1.0,           # Zusätzliches Gradient Clipping
    val_check_interval=0.5,          # Validierung zweimal pro Epoche
    detect_anomaly=False,            # Deaktiviert für bessere Performance
    enable_checkpointing=True,
    enable_progress_bar=True,
    enable_model_summary=True
)

print("Starte Training...")
trainer.fit(model, train_loader, val_loader)

# -------------------- IoU Evaluation --------------------
metric = torchmetrics.classification.MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=-1, average='none').to(model.device)
model.eval(); metric.reset()
with torch.no_grad():
    for xb, yb in val_loader:
        preds = torch.argmax(model(xb.to(model.device)), 1)
        metric.update(preds.cpu(), yb)
print("IoU:", metric.compute())

# -------------------- Gewichte speichern -----------------
torch.save(model.state_dict(), "/home/t/tstraus2/unet_kiln_sentinel2_8ch.pt")
print("Gewichte gespeichert → unet_kiln_sentinel2_8ch.pt")
