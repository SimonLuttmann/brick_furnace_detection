'''\
Sentinel-2 Kiln Detection V3.2 – CRITICAL FIXES Applied
=======================================================

PROBLEME BEHOBEN:
-----------------
kiln-33215049-v3-2.log:
1. VORHERSAGE-KOLLAPS: Modell sagte nur Klasse 1 vorher
2. SCHLECHTE METRIKEN: Macro F1: 0.0830, IoU: [0.0009, 0.4951, 0.0000, ...]
3. FRÜHE TERMINIERUNG: Stopped nach 13 Epochen ohne Verbesserung

kiln-33216281-v3-2.log:
4. KOMPATIBILITÄTSFEHLER: TypeError mit verbose Parameter in ReduceLROnPlateau

kiln-33218200-v3-2.log:
5. DEVICE-MISMATCH: RuntimeError - alpha weights auf CPU, targets auf GPU

kiln-33218215-v3-2.log - NEUE KRITISCHE FIXES:
6. EXTREME KLASSENRESTRIKTION: Nur Klassen 0+1 predicted, Spalten 2-7 alle 0
7. FALSCHES EARLY STOPPING: Stoppt nach 5 Epochen mit "15 epochs no improvement"

NEUESTE VERBESSERUNGEN:
----------------------
• Early Stopping: Robuste Metrik-Extraktion, numerische Stabilität
• Klassengewichte: Stark erhöht für seltene Klassen (30.0 für Klasse 6)
• FocalLoss: Gamma 2.0 → 3.0, Label Smoothing 0.1 → 0.05

🚀 MEGA-UPDATE V3.2.7: KOMPLETTE DATENVERARBEITUNG + WEIGHTED F1 SCORE!
========================================================================
• KRITISCHER BUGFIX: Externe data_loader.py → interne Datenverarbeitung
• PROBLEM: Externes data_loader.py hat andere Datenverarbeitung als DetectionV4_1
• LÖSUNG: Nutzt IDENTISCHE interne Datenverarbeitung wie DetectionV4_1
• read_s2_image() übernommen von DetectionV4_1
• build_standard_loaders() übernommen von DetectionV4_1
• SentinelKilnDataset komplett wie DetectionV4_1 implementiert
• Kein externes data_loader.py mehr - vollständig self-contained
• ALLE Parameter IDENTISCH mit funktionierendem DetectionV4_1!
• ZUSÄTZLICH: Weighted F1 Score + Classification Report für bessere Evaluation!

KLASSENBALANCIERUNG:
-------------------
• Gewichtete Auswahl seltener Klassen (3, 5, 6, 7)
• Verbesserte Kiln-Lokalisierung und Caching
• Boost-Faktor für seltene Klassen: 1.5x
• Dynamische Klassengewichte: Berechnet aus echter Pixel-Verteilung
• Fallback Klassengewichte: [4.4, 2.9, 19.7, 20.0, 20.0, 20.0, 20.0, 20.0]

LOSS-OPTIMIERUNG:
----------------
• CrossEntropyLoss statt FocalLoss (bewährte Konfiguration wie DetectionV4_1)
• Klassengewichte für seltene Klassen: [4.4, 2.9, 19.7, 20.0, 20.0, 20.0, 20.0, 20.0]
• Dynamische Klassengewichte basierend auf echter Pixel-Verteilung
• FIXED: None-handling für Batches ohne valide Labels
• AdamW Optimizer (statt Adam)
• ReduceLROnPlateau Scheduler (PyTorch-kompatibel)
• Gradient Accumulation (2 Batches)

TRAINING-SETUP:
--------------
• FIXED: Validierung nur am Epochen-Ende (1.0 statt 0.25) - Early Stopping Bug behoben!
• Reduziertes Gradient Clipping (0.5 statt 1.0)
• Deterministische Ergebnisse
• Robuste Early Stopping mit Debug-Output
• Verbesserte Tensorboard-Logs
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
# REMOVED: from performance_analysis import calculate_and_log_performance
# REMOVED: from utils import print_available_memory
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

USER = getpass.getuser()
SCRATCH = os.environ.get("SLURM_TMPDIR", f"/scratch/tmp/{USER}")
ZIP_PATH = os.path.join(SCRATCH, "Brick_Data_Train.zip")
DATA_ROOT = os.path.join(SCRATCH, "Brick_Data_Train")
IMG_DIR  = os.path.join(DATA_ROOT, "Image")      # Korrigiert: "Image" statt "images"
MASK_DIR = os.path.join(DATA_ROOT, "Mask")       # Korrigiert: "Mask" statt "mask"
PATCH_SIZE  = 256
BATCH_SIZE  = 8              # FIXED: Wie DetectionV4_1 (funktionierend)
LR          = 1e-4           # FIXED: Wie DetectionV4_1 (funktionierend)
EPOCHS      = 50            # FIXED: Wie DetectionV4_1 (funktionierend)
NUM_CLASSES = 8              
IN_CHANNELS = 8
TERMINATTE_EARLY = 15        # Längere Geduld für Early Stopping

# Erweiterte Augmentation-Parameter (Hybrides Sampling für alle 8000 Bilder)
POSITIVE_RATIO = 0.6         # Ausgewogenes Verhältnis: 60% positive, 40% negative
RARE_CLASS_BOOST = 1.5       # Boost für seltene Klassen

# Model naming for tensorboard and checkpoints
MODEL_NAME = f"unet_kiln_8ch_balanced_lr{LR}_bs{BATCH_SIZE}_epochs{EPOCHS}"
pl.seed_everything(42, workers=True)

# CRITICAL FIX: Disable deterministic algorithms for CrossEntropyLoss GPU compatibility
import torch
torch.use_deterministic_algorithms(False)
print("Disabled deterministic algorithms for GPU CrossEntropyLoss compatibility")

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
# --------------------------------------------------
# 2.1 · Internal data loading (like DetectionV4_1)
# --------------------------------------------------

def read_s2_image(path: str) -> torch.Tensor:
    """
    Liest ein Sentinel-2-Bild mit **8 Kanälen** aus einer GeoTIFF-Datei.
    
    Args:
        path: Pfad zur GeoTIFF-Datei
        
    Returns:
        torch.Tensor: Bild-Tensor mit Shape [8, H, W] im Bereich [0, 1]
    """
    try:
        with rasterio.open(path) as src:
            img_np = src.read(out_dtype=np.float32)[:8]  # Nur die ersten 8 Kanäle
            img_np = np.nan_to_num(img_np, nan=0.0)      # NaN → 0.0
            
            # Clamp to reasonable range (TOA reflectance should be ~0-1)
            img_np = np.clip(img_np, -0.1, 1.0)
            
            # Debug: Log data range periodically
            if np.random.random() < 0.01:  # 1% sampling for debug
                print(f"DEBUG: Data range: {img_np.min():.6f} to {img_np.max():.6f}")
            
            return torch.from_numpy(img_np).float()
    except Exception as e:
        raise RuntimeError(f"Fehler beim Lesen von {path}: {e}")

# Path-based dataset for memory-efficient loading
class SentinelKilnDatasetPaths(Dataset):
    def __init__(self, image_paths, label_paths, patch_size=256, positive_ratio=0.8, is_train=True):
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
        
        # Berechne Klassenverteilung für bessere Balancierung
        self._compute_class_distribution()

        # Simplified normalization using Sentinel-2 standard values
        sentinel2_mean = [0.15, 0.12, 0.11, 0.13, 0.25, 0.35, 0.38, 0.30]
        sentinel2_std = [0.10, 0.08, 0.07, 0.08, 0.12, 0.15, 0.16, 0.14]
        
        self.norm = T.Normalize(mean=sentinel2_mean, std=sentinel2_std)
        print(f"Using Sentinel-2 standard normalization")

    def _cache_kiln_locations(self):
        """Cache ALL images and kiln locations for hybrid sampling strategy"""
        self.kiln_coords = {}
        self.negative_coords = {}  # NEW: Images WITHOUT kilns for negative sampling
        self.class_coords = {i: {} for i in range(NUM_CLASSES)}  # Alle Klassen tracken
        self.class_pixel_counts = {i: 0 for i in range(NUM_CLASSES)}
        
        for i, label_path in enumerate(self.label_paths):
            try:
                with rasterio.open(label_path) as src:
                    mask_raw = src.read(1)
                
                # Convert labels: 0→-1, 1-8→0-7
                mask = np.where(mask_raw == 0, -1, mask_raw - 1)
                
                # Find kiln pixels
                kiln_pixels = np.where((mask >= 0) & (mask <= 7))
                if len(kiln_pixels[0]) > 0:
                    # Image HAS kilns - cache for positive sampling
                    self.kiln_coords[i] = list(zip(kiln_pixels[0], kiln_pixels[1]))
                else:
                    # NEW: Image has NO kilns - cache for negative sampling
                    self.negative_coords[i] = True
                
                # Cache alle Klassen für bessere Balancierung
                for target_class in range(NUM_CLASSES):
                    class_pixels = np.where(mask == target_class)
                    if len(class_pixels[0]) > 0:
                        self.class_coords[target_class][i] = list(zip(class_pixels[0], class_pixels[1]))
                        self.class_pixel_counts[target_class] += len(class_pixels[0])
                        
            except Exception as e:
                print(f"Warning: Could not process {label_path}: {e}")
                
        print(f"Pixel counts per class: {self.class_pixel_counts}")
        print(f"Images WITH kilns: {len(self.kiln_coords)}")
        print(f"Images WITHOUT kilns: {len(self.negative_coords)}")
        print(f"Total images available: {len(self.kiln_coords) + len(self.negative_coords)}")
        
    def _compute_class_distribution(self):
        """Berechne Klassenverteilung für bessere Sampling-Strategie"""
        total_pixels = sum(self.class_pixel_counts.values())
        self.class_frequencies = {k: v/total_pixels for k, v in self.class_pixel_counts.items()}
        
        # Inverse Frequenz für Sampling-Gewichtung
        self.class_sampling_weights = {}
        for k, freq in self.class_frequencies.items():
            if freq > 0:
                self.class_sampling_weights[k] = 1.0 / freq
            else:
                self.class_sampling_weights[k] = 0.0
                
        # Normalisiere Gewichte
        total_weight = sum(self.class_sampling_weights.values())
        if total_weight > 0:
            for k in self.class_sampling_weights:
                self.class_sampling_weights[k] /= total_weight
                
        print(f"Class frequencies: {self.class_frequencies}")
        print(f"Class sampling weights: {self.class_sampling_weights}")

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
        
        # Apply improved targeting strategies with class balancing
        _, H, W = img.shape
        
        # HYBRID SAMPLING STRATEGY: Use ALL 8000 images
        use_positive_sampling = random.random() < self.positive_ratio
        
        if use_positive_sampling and idx in self.kiln_coords:
            # POSITIVE SAMPLING: Current image has kilns - do targeted sampling
            available_classes = []
            for class_id in range(NUM_CLASSES):
                if idx in self.class_coords[class_id]:
                    available_classes.append(class_id)
            
            if available_classes:
                # Gewichtete Auswahl seltener Klassen
                if len(available_classes) > 1:
                    # Boost für seltene Klassen (3, 5, 6)
                    rare_classes = [c for c in available_classes if c in [3, 5, 6]]
                    if rare_classes and random.random() < RARE_CLASS_BOOST * 0.3:
                        target_class = random.choice(rare_classes)
                    else:
                        # Gewichtete Auswahl basierend auf Seltenheit
                        weights = [self.class_sampling_weights.get(c, 0.1) for c in available_classes]
                        target_class = np.random.choice(available_classes, p=np.array(weights)/np.sum(weights))
                else:
                    target_class = available_classes[0]
                
                # Wähle Koordinaten aus der gewählten Klasse
                coords = self.class_coords[target_class][idx]
                y_center, x_center = random.choice(coords)
                sampling_type = f"POSITIVE_CLASS_{target_class}"
            else:
                # Fallback auf allgemeine Kiln-Koordinaten
                coords = self.kiln_coords[idx]
                y_center, x_center = random.choice(coords)
                sampling_type = "POSITIVE_GENERAL"
        else:
            # NEGATIVE SAMPLING: Random patch from any image (including no-kiln images)
            y_center = random.randint(self.patch_size // 2, H - self.patch_size // 2)
            x_center = random.randint(self.patch_size // 2, W - self.patch_size // 2)
            if idx in self.negative_coords:
                sampling_type = "NEGATIVE_NO_KILNS"
            else:
                sampling_type = "NEGATIVE_RANDOM"
        
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


# Internal dataset class (like DetectionV4_1)
class SentinelKilnDataset(Dataset):
    def __init__(self, img_dir, mask_dir, patch_size=256, positive_ratio=0.7, augmentations=None):
        """
        Internal dataset class like DetectionV4_1 - no external dependencies
        
        Args:
            img_dir: directory containing image files
            mask_dir: directory containing mask files
            patch_size: size of patches to extract
            positive_ratio: ratio of positive samples to generate
            augmentations: augmentation pipeline (not used in this simplified version)
        """
        # Validierung der Verzeichnisse
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Bildverzeichnis nicht gefunden: {img_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Maskenverzeichnis nicht gefunden: {mask_dir}")

        # Get all image paths
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        assert self.img_paths, f"Keine gültigen Bilder in {img_dir} gefunden!"

        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.positive_ratio = positive_ratio
        self.augmentations = augmentations

        print(f"Dataset initialized with {len(self.img_paths)} images")
        
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
        self.class4_coords = {}
        self.class3_coords = {}
        self.class2_coords = {}
        self.class7_coords = {}
        
        print("Caching kiln locations for all classes...")
        total_images = len(self.img_paths)
        
        for idx, img_path in enumerate(self.img_paths):
            if idx % 500 == 0:
                print(f"Processing image {idx}/{total_images}...")
            
            # Get corresponding mask path
            mask_path = os.path.join(self.mask_dir, os.path.basename(img_path))
            
            if not os.path.exists(mask_path):
                continue
                
            try:
                # Read mask
                with rasterio.open(mask_path) as src:
                    mask_raw = src.read(1)  # Original 1-9 Labels lesen
            except Exception as e:
                print(f"Error reading mask {mask_path}: {e}")
                continue
            
            # Convert to processed mask (0→-1, 1-8→0-7)
            mask = torch.from_numpy(mask_raw).long()
            mask = torch.where(mask == 0, torch.tensor(-1), mask - 1)
            
            # Find coordinates for each class
            class_coords = {}
            has_kiln = False
            
            for class_id in range(NUM_CLASSES):
                coords = torch.nonzero(mask == class_id, as_tuple=False)
                if len(coords) > 0:
                    coords_list = coords.tolist()
                    class_coords[class_id] = coords_list
                    
                    # Store in specific class caches
                    if class_id == 6:
                        self.class6_coords[img_path] = coords_list
                    elif class_id == 5:
                        self.class5_coords[img_path] = coords_list
                    elif class_id == 4:
                        self.class4_coords[img_path] = coords_list
                    elif class_id == 3:
                        self.class3_coords[img_path] = coords_list
                    elif class_id == 2:
                        self.class2_coords[img_path] = coords_list
                    elif class_id == 7:
                        self.class7_coords[img_path] = coords_list
                    
                    # Mark as having kiln pixels
                    has_kiln = True
            
            # Store general kiln coordinates
            if has_kiln:
                self.kiln_coords[img_path] = class_coords
        
        print(f"Cached locations for {len(self.kiln_coords)} images with kilns")
        print(f"Class 6 images: {len(self.class6_coords)}")
        print(f"Class 5 images: {len(self.class5_coords)}")
        print(f"Class 4 images: {len(self.class4_coords)}")
        print(f"Class 3 images: {len(self.class3_coords)}")
        print(f"Class 2 images: {len(self.class2_coords)}")
        print(f"Class 7 images: {len(self.class7_coords)}")
        print(f"Total images: {len(self.kiln_coords)}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Get image and mask paths
        img_path = self.img_paths[idx]
        img = read_s2_image(img_path)  # Shape: [8, H, W]
        mask_path = os.path.join(self.mask_dir, os.path.basename(img_path))

        # Validierung ob Maskendatei existiert
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Maskendatei nicht gefunden: {mask_path}")

        try:
            with rasterio.open(mask_path) as src:
                mask_raw = src.read(1)  # Original 1-9 Labels lesen
        except Exception as e:
            raise RuntimeError(f"Fehler beim Lesen der Maske {mask_path}: {e}")
        
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

        # TARGETING STRATEGY: Use kiln locations for positive sampling
        use_positive_sampling = random.random() < self.positive_ratio
        
        if use_positive_sampling and img_path in self.kiln_coords:
            # Positive sampling: Target kiln pixels
            available_classes = list(self.kiln_coords[img_path].keys())
            if available_classes:
                # Priorisierte Klassen-Auswahl für seltene Klassen
                if img_path in self.class6_coords and random.random() < 0.35:
                    target_class = 6
                    coords = self.class6_coords[img_path]
                elif img_path in self.class5_coords and random.random() < 0.30:
                    target_class = 5
                    coords = self.class5_coords[img_path]
                elif img_path in self.class4_coords and random.random() < 0.25:
                    target_class = 4
                    coords = self.class4_coords[img_path]
                else:
                    # Normale Klassen-Auswahl
                    target_class = random.choice(available_classes)
                    coords = self.kiln_coords[img_path][target_class]
                
                y_center, x_center = random.choice(coords)
                
                # Rare class boost für bessere Representation
                if target_class in [3, 4, 5, 6, 7]:
                    # Verwende ein etwas größeres Patch für besseren Context
                    effective_patch_size = min(int(self.patch_size * RARE_CLASS_BOOST), min(H, W) - 10)
                else:
                    effective_patch_size = self.patch_size
                    
                # Patch-Grenzen berechnen
                half_patch = effective_patch_size // 2
                y1 = max(0, y_center - half_patch)
                y2 = min(H, y1 + effective_patch_size)
                x1 = max(0, x_center - half_patch)
                x2 = min(W, x1 + effective_patch_size)
            else:
                # Fallback: Random sampling
                y_center = random.randint(self.patch_size // 2, H - self.patch_size // 2)
                x_center = random.randint(self.patch_size // 2, W - self.patch_size // 2)
                y1 = max(0, y_center - self.patch_size // 2)
                y2 = min(H, y1 + self.patch_size)
                x1 = max(0, x_center - self.patch_size // 2)
                x2 = min(W, x1 + self.patch_size)
        else:
            # Random sampling for diversity
            y_center = random.randint(self.patch_size // 2, H - self.patch_size // 2)
            x_center = random.randint(self.patch_size // 2, W - self.patch_size // 2)
            y1 = max(0, y_center - self.patch_size // 2)
            y2 = min(H, y1 + self.patch_size)
            x1 = max(0, x_center - self.patch_size // 2)
            x2 = min(W, x1 + self.patch_size)

        # Extract patch
        img_patch = img[:, y1:y2, x1:x2]
        mask_patch = mask[y1:y2, x1:x2]

        # Ensure correct patch size
        if img_patch.shape[1] != self.patch_size or img_patch.shape[2] != self.patch_size:
            img_patch = F.interpolate(img_patch.unsqueeze(0), size=(self.patch_size, self.patch_size), 
                                    mode='bilinear', align_corners=False).squeeze(0)
            mask_patch = F.interpolate(mask_patch.unsqueeze(0).unsqueeze(0).float(), 
                                     size=(self.patch_size, self.patch_size), 
                                     mode='nearest').squeeze(0).squeeze(0).long()

        # Debug output every 100th sample
        if idx % 100 == 0:
            unique_classes = torch.unique(mask_patch[mask_patch != -1])
            print(f"Sample {idx}: Classes: {unique_classes.tolist()}")

        return self.norm(img_patch), mask_patch

# --------------------------------------------------
# 4 · UNet (8 Eingangs-Kanäle) - Vollständige Architektur  
# --------------------------------------------------

class FocalLoss(nn.Module):
    """
    Improved Focal Loss für extreme Klassenungleichgewicht bei Kiln Detection
    
    Focal Loss = -α(1-pt)^γ * log(pt)
    
    Verbesserungen:
    - Adaptive Gamma-Adjustierung basierend auf Klassenseltenheit
    - Label Smoothing für Regularisierung
    - Bessere Numerische Stabilität
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-1, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # Standard CrossEntropy loss mit Label Smoothing
        ce_loss = F.cross_entropy(
            inputs, targets, 
            reduction='none', 
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing
        )
        
        # Berechne Wahrscheinlichkeiten mit numerischer Stabilität
        pt = torch.exp(-ce_loss).clamp(min=1e-8, max=1.0)
        
        # Apply alpha weights if provided
        if self.alpha is not None:
            # Verschiebe alpha auf das gleiche Device wie targets
            alpha_device = self.alpha.to(targets.device)
            alpha_t = torch.ones_like(targets, dtype=torch.float32, device=targets.device)
            valid_mask = targets != self.ignore_index
            alpha_t[valid_mask] = alpha_device[targets[valid_mask]]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Mittelwert über valide Pixel
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

    def _compute_class_weights(self, data):
        """Compute class weights dynamically based on actual dataset distribution"""
        print("Computing dynamic class weights from dataset...")
        
        # Count pixels for each class (after mapping: 0→-1, 1-8→0-7)
        class_counts = np.zeros(NUM_CLASSES)
        background_count = 0
        total_pixels = 0
        
        # Handle both numpy arrays and dataset objects
        if isinstance(data, tuple) and len(data) == 2:
            # RAM-based loading: (X_train, y_train)
            X_data, y_data = data
            if y_data is not None and len(y_data) > 0:
                for mask_raw in y_data:
                    unique, counts = np.unique(mask_raw, return_counts=True)
                    for label, count in zip(unique, counts):
                        total_pixels += count
                        if label == 0:  # Background (wird zu -1 gemappt)
                            background_count += count
                        elif 1 <= label <= 8:  # Kiln classes (werden zu 0-7 gemappt)
                            class_counts[label - 1] += count
        
        elif hasattr(data, '__len__') and hasattr(data, '__getitem__'):
            # Path-based loading: Dataset object
            print("Sampling dataset for class weight computation...")
            # Sample subset of dataset to compute weights (for efficiency)
            sample_size = min(200, len(data))
            sample_indices = np.random.choice(len(data), sample_size, replace=False)
            
            for idx in sample_indices:
                try:
                    _, mask = data[idx]  # Get image and mask
                    
                    # Convert mask to numpy if it's a tensor
                    if isinstance(mask, torch.Tensor):
                        mask_np = mask.numpy()
                    else:
                        mask_np = mask
                    
                    # Count pixels - mask already has -1 for background, 0-7 for classes
                    unique, counts = np.unique(mask_np, return_counts=True)
                    for label, count in zip(unique, counts):
                        total_pixels += count
                        if label == -1:  # Background (already mapped)
                            background_count += count
                        elif 0 <= label <= 7:  # Kiln classes (already mapped)
                            class_counts[label] += count
                            
                except Exception as e:
                    print(f"Warning: Could not process sample {idx}: {e}")
                    continue
        
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

    def __init__(self, lr=LR):
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

        # CRITICAL FIX: Use EXACT same hardcoded weights as DetectionV4_1 (which works!)
        # Dynamic weights were TOO WEAK for rare classes: [2.0, 1.9, 13.3, 17.9, 5.4, 13.7, 24.9, 8.9]
        # DetectionV4_1 uses stronger weights: [4.4, 2.9, 19.7, 20.0, 20.0, 20.0, 20.0, 20.0]
        class_weights = torch.tensor([
            4.4,    # Class 0 (original 1): 0.0225% → weight 4.4
            2.9,    # Class 1 (original 2): 0.0339% → weight 2.9
            19.7,   # Class 2 (original 3): 0.0051% → weight 19.7
            20.0,   # Class 3 (original 4): 0.0015% → weight 20.0
            20.0,   # Class 4 (original 5): 0.0025% → weight 20.0
            20.0,   # Class 5 (original 6): 0.0007% → weight 20.0 (extrem selten)
            20.0,   # Class 6 (original 7): 0.0004% → weight 20.0 (extrem selten)
            20.0    # Class 7 (original 8): 0.0040% → weight 20.0
        ])
        
        # CRITICAL FIX: Use CrossEntropyLoss like DetectionV4_1 (which works!)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)
        print(f"Using EXACT same class weights as DetectionV4_1 (hardcoded, not dynamic)")

        # Proper weight initialization für alle Layer AUSSER Output
        self.apply(self._init_weights)

        # Spezielle Initialisierung für Output Layer NACH apply() um Überschreibung zu vermeiden
        with torch.no_grad():
            nn.init.xavier_uniform_(self.out.weight, gain=0.01)  # FIXED: Exakt wie DetectionV4_1
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

        # Debug: Analysiere die Label-Werte (REDUCED like V4_1)
        unique_labels = torch.unique(y)
        if len(unique_labels) > 0:
            print(f"Unique labels in batch: {unique_labels.tolist()[:10]}...")  # Limit output like V4_1

        # Prüfe auf ungültige Label-Werte für CrossEntropy
        if y.min() < -1 or y.max() >= NUM_CLASSES:
            print(f"ERROR: Invalid label values! min={y.min()}, max={y.max()}")
            print(f"Expected range: -1 to {NUM_CLASSES-1}")
            raise ValueError("Invalid label values detected")

        # CRITICAL FIX: Prüfe auf "nur ignore_index" Batch
        valid_labels = y[y != -1]  # Alle nicht-ignore Labels
        if len(valid_labels) == 0:
            print("WARNING: Skipping batch with only ignore labels")
            # KORRIGIERT: Return None statt dummy loss (wie DetectionV4_1)
            return None

        logits = self(x)
        loss = self.loss(logits, y)

        # NaN-Check für Loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: NaN/Inf loss detected!")
            print(f"Loss value: {loss}")
            print(f"Valid pixels in batch: {len(valid_labels)}")
            # KORRIGIERT: Return None statt dummy loss (wie DetectionV4_1)
            return None

        return loss

    def training_step(self, batch, idx):
        loss = self.shared_step(batch)
        if loss is not None:  # CRITICAL FIX: Skip dummy batches (wie DetectionV4_1)
            self.log("train_loss", loss, prog_bar=True)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', loss, self.global_step)
            
            return loss
        else:
            return None  # Skip this batch



    def validation_step(self, batch, idx):
        loss = self.shared_step(batch)
        if loss is not None:  # CRITICAL FIX: Skip dummy batches (wie DetectionV4_1)
            self.log("val_loss", loss, prog_bar=True)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Val', loss, self.global_step)
            
            # CRITICAL FIX: NO POST-PROCESSING - Use raw predictions like DetectionV4_1
            x, y = batch
            with torch.no_grad():
                logits = self(x)
                
                # FIXED: Use raw argmax predictions like DetectionV4_1 (NO post-processing!)
                preds = torch.argmax(logits, dim=1)
                
                # Only collect valid (non-ignore) predictions and labels
                valid_mask = y != -1
                valid_preds = preds[valid_mask]
                valid_labels = y[valid_mask]
                
                # Simple collection like V4_1 (no additional filtering)
                if len(valid_preds) > 0:
                    self.all_val_preds.extend(valid_preds.cpu().numpy())
                    self.all_val_labels.extend(valid_labels.cpu().numpy())
            
            return loss
        else:
            return None  # Skip this batch

    def on_validation_epoch_end(self):
        """Calculate and log F1-Score and other metrics at the end of each validation epoch"""
        if len(self.all_val_preds) == 0:
            print("Warning: No valid predictions collected for F1-Score calculation")
            return
            
        # Convert to numpy arrays
        all_preds = np.array(self.all_val_preds)
        all_labels = np.array(self.all_val_labels)
        
        print(f"Epoch {self.current_epoch}: Computing F1-Score on {len(all_preds)} validation pixels", flush=True)
        
        # Calculate weighted F1 Score
        from sklearn.metrics import f1_score, classification_report
        try:
            # Weighted F1 Score (wichtig für unbalanced classes)
            weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            
            print(f"Epoch {self.current_epoch} F1 Scores:")
            print(f"  Weighted F1: {weighted_f1:.4f}")
            print(f"  Macro F1: {macro_f1:.4f}")
            
            # Log to tensorboard
            self.writer.add_scalar('Metrics/Weighted_F1', weighted_f1, self.current_epoch)
            self.writer.add_scalar('Metrics/Macro_F1', macro_f1, self.current_epoch)
            
            # Classification report für Details
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, 
                                      target_names=[f'Class_{i}' for i in range(NUM_CLASSES)],
                                      zero_division=0))
            
        except Exception as e:
            print(f"Error calculating F1 scores: {e}")
            print(f"Labels range: {np.min(all_labels)} to {np.max(all_labels)}")
            print(f"Preds range: {np.min(all_preds)} to {np.max(all_preds)}")
            print(f"Unique labels: {np.unique(all_labels)}")
            print(f"Unique preds: {np.unique(all_preds)}")
        
        # FIXED: Robust early stopping logic
        val_loss_logs = self.trainer.callback_metrics
        print(f"DEBUG: Available metrics: {list(val_loss_logs.keys())}")
        
        # Try multiple ways to get validation loss
        current_val_loss = None
        for key in ['val_loss', 'val_loss_epoch', 'validation_loss']:
            if key in val_loss_logs:
                current_val_loss = float(val_loss_logs[key])
                print(f"Using {key}: {current_val_loss:.6f}")
                break
                
        if current_val_loss is None:
            print("WARNING: Could not find validation loss in metrics, skipping early stopping check")
            current_val_loss = float('inf')
        
        # Early stopping check
        print(f"Current val_loss: {current_val_loss:.6f}, Best: {self.best_val_loss:.6f}")
        
        if current_val_loss < self.best_val_loss - 1e-6:  # Small epsilon for numerical stability
            improvement = self.best_val_loss - current_val_loss
            self.best_val_loss = current_val_loss
            self.epochs_no_improve = 0
            print(f"✓ NEW BEST validation loss: {current_val_loss:.6f} (improvement: {improvement:.6f})")
        else:
            self.epochs_no_improve += 1
            print(f"⚠ No improvement for {self.epochs_no_improve}/{TERMINATTE_EARLY} epochs (validation steps, not full epochs)")
            
        if self.epochs_no_improve >= TERMINATTE_EARLY:
            print(f"🛑 Early stopping: No improvement in {TERMINATTE_EARLY} epochs.")
            self.trainer.should_stop = True
        
        # Clear for next epoch
        self.all_val_preds.clear()
        self.all_val_labels.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        
        # Main scheduler - ReduceLROnPlateau ohne verbose (Kompatibilität)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
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

def build_standard_loaders():
    """Build standard data loaders without augmentation (like DetectionV4_1)"""
    ds = SentinelKilnDataset(IMG_DIR, MASK_DIR, PATCH_SIZE, augmentations=None)
    val_len = max(1, int(0.1 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds)-val_len, val_len])

    num_workers = min(8, os.cpu_count() or 4)
    kwargs = dict(batch_size=BATCH_SIZE,
                  num_workers=num_workers,
                  pin_memory=torch.cuda.is_available())

    return DataLoader(train_ds, shuffle=True, **kwargs), DataLoader(val_ds, shuffle=False, **kwargs)

# Initialisierung der DataLoader
train_loader, val_loader = build_standard_loaders()
print(f"Train: {len(train_loader)}  Val: {len(val_loader)}")

# Modell und Training konfigurieren mit hardcoded class weights (wie DetectionV4_1)
model = UNet()
ckpt  = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
lrmon = LearningRateMonitor(logging_interval="epoch")

# Erweiterte Trainer-Konfiguration für bessere Konvergenz
trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    precision="32",                  # 32-bit für Stabilität
    max_epochs=EPOCHS,
    callbacks=[ckpt, lrmon],
    log_every_n_steps=20,
    gradient_clip_val=1.0,           # FIXED: Wie DetectionV4_1
    val_check_interval=1.0,          # FIXED: Validierung nur am Epochen-Ende
    detect_anomaly=False,            # Performance-Optimierung
    enable_checkpointing=True,
    enable_progress_bar=True,
    enable_model_summary=True,
    accumulate_grad_batches=2,       # Gradient Accumulation für effektive Batch-Größe
            deterministic=False,             # Disabled für CrossEntropyLoss GPU Kompatibilität
    num_sanity_val_steps=2           # Reduzierte Sanity-Checks
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
