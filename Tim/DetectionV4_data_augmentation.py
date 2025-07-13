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

USER = getpass.getuser()
SCRATCH = os.environ.get("SLURM_TMPDIR", f"/scratch/tmp/{USER}")
ZIP_PATH = os.path.join(SCRATCH, "Brick_Data_Train.zip")
DATA_ROOT = os.path.join(SCRATCH, "Brick_Data_Train")
IMG_DIR  = os.path.join(DATA_ROOT, "Image")
MASK_DIR = os.path.join(DATA_ROOT, "Mask")
PATCH_SIZE  = 256
BATCH_SIZE  = 8
LR          = 1e-4          # Reduzierte Lernrate für Stabilität
EPOCHS      = 25            # ERHÖHT: Mindestens 25 Epochen für seltene Klassen
NUM_CLASSES = 8             # KORRIGIERT: Nach Mapping 0-8 → 0-7 sind es 8 Klassen
IN_CHANNELS = 8
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
# 2 · Hilfe: Bild einlesen (8 Bänder)
# --------------------------------------------------

def read_s2_image(path: str) -> torch.Tensor:
    """Liest GeoTIFF, skaliert DN auf [0, 1] und ersetzt NaNs durch 0.

    FIXED: Critical return statement bug and improved data validation
    """
    try:
        with rasterio.open(path) as src:
            img = src.read(out_dtype=np.float32)[:IN_CHANNELS]

            # DEBUG: Check for completely invalid data
            if np.all(np.isnan(img)) or np.all(img == 0):
                print(f"WARNING: Image {path} contains only NaN or zero values!")

            # DEBUG: Check original data range (only for non-NaN values)
            valid_data = img[~np.isnan(img)]
            if len(valid_data) > 0:
                print(f"DEBUG: Original data range: {valid_data.min():.6f} to {valid_data.max():.6f}")
            else:
                print(f"DEBUG: Original data range: ALL NaN values!")

    except Exception as e:
        raise RuntimeError(f"Fehler beim Lesen von {path}: {e}")

    # IMPROVED: Better handling of different data formats
    # Replace NaNs first to avoid issues in comparison
    img_clean = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

    # Check if data is already in reasonable reflectance range
    valid_pixels = img_clean[img_clean > 0]  # Exclude zeros from analysis

    if len(valid_pixels) > 0 and valid_pixels.max() <= 1.0 and valid_pixels.min() >= -0.1:
        print("DEBUG: Data appears to be already in TOA reflectance format (0-1 range)")
        img_normalized = img_clean
    else:
        print("DEBUG: Converting DN values to TOA reflectance (/10000)")
        # DN‑Werte nach ESA‑Empfehlung auf 0‑1 skalieren
        img_normalized = img_clean / 10000.0

    # Final cleanup and validation
    img_final = np.clip(img_normalized, -0.1, 2.0)  # Reasonable bounds for reflectance

    print(f"DEBUG: Final data range: {img_final.min():.6f} to {img_final.max():.6f}")

    # CRITICAL FIX: Actually return the processed data!
    return torch.from_numpy(img_final)

# --------------------------------------------------
# 2.5 · Advanced Data Augmentation for Rare Classes
# --------------------------------------------------

class SentinelAugmentations:
    """Spezialisierte Augmentations für Sentinel-2 Daten und seltene Klassen - REDUCED INTENSITY"""

    def __init__(self):
        # KORRIGIERTE Klassen-spezifische Augmentation-Wahrscheinlichkeiten (weniger aggressiv)
        self.class_augmentation_probs = {
            0: 0.05,  # Background - minimale Augmentation (reduziert von 0.1)
            1: 0.15,  # Häufige Kiln-Klassen (reduziert von 0.3)
            2: 0.15,  # (reduziert von 0.3)
            3: 0.20,  # (reduziert von 0.4)
            4: 0.30,  # Seltene Klassen (reduziert von 0.6)
            5: 0.25,  # (reduziert von 0.5)
            6: 0.35,  # Sehr seltene Klassen (reduziert von 0.7)
            7: 0.40,  # (reduziert von 0.8)
            8: 0.50   # Extrem seltene Klasse 8 (reduziert von 0.9)
        }

        # Rare classes definition basierend auf Label-Analyse
        self.rare_classes = [4, 5, 6, 7, 8]  # <0.01% of pixels each
        self.ultra_rare_classes = [6, 7, 8]  # <0.001% of pixels each

    def apply_spatial_augmentation(self, img_tensor, mask_tensor, intensity='standard'):
        """Applies spatial augmentations that preserve semantic content"""

        # Combine image and mask for synchronized transforms
        combined = torch.cat([img_tensor, mask_tensor.unsqueeze(0).float()], dim=0)

        augmentation_list = []

        if intensity in ['standard', 'aggressive']:
            # Basic spatial transforms
            augmentation_list.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
            ])

        if intensity == 'aggressive':
            # More intensive augmentations for rare classes
            augmentation_list.extend([
                T.RandomRotation(degrees=90, interpolation=T.InterpolationMode.BILINEAR),
                T.RandomApply([T.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.3),
            ])

        # Apply transforms
        if augmentation_list:
            transform = T.Compose(augmentation_list)
            combined = transform(combined)

        # Split back
        img_aug = combined[:IN_CHANNELS]
        mask_aug = combined[IN_CHANNELS:].squeeze(0).long()

        return img_aug, mask_aug

    def apply_spectral_augmentation(self, img_tensor, intensity='standard'):
        """Applies spectral augmentations specific to Sentinel-2 data"""

        augmented = img_tensor.clone()

        if intensity in ['standard', 'aggressive']:
            # Atmospheric variations simulation
            if random.random() < 0.3:
                brightness_factor = random.uniform(0.8, 1.2)
                augmented = torch.clamp(augmented * brightness_factor, 0, 1)

            # Band-specific noise (common in satellite imagery)
            if random.random() < 0.2:
                noise = torch.randn_like(augmented) * 0.01
                augmented = torch.clamp(augmented + noise, 0, 1)

        if intensity == 'aggressive':
            # More aggressive spectral changes for rare classes
            if random.random() < 0.4:
                # Simulate atmospheric haze
                haze_factor = random.uniform(0.95, 1.05)
                augmented = torch.clamp(augmented * haze_factor + random.uniform(-0.02, 0.02), 0, 1)

            # Band permutation (simulate different atmospheric conditions)
            if random.random() < 0.1:
                # Slight permutation of similar bands (e.g., swap red-edge bands)
                if IN_CHANNELS >= 6:
                    # Swap bands 4 and 5 (red-edge bands)
                    augmented[[4, 5]] = augmented[[5, 4]]

        return augmented

    def generate_multiple_versions(self, img_tensor, mask_tensor, target_class):
        """Generate multiple augmented versions for ultra-rare classes"""
        versions = []

        # Original version
        versions.append((img_tensor, mask_tensor, "ORIGINAL"))

        if target_class in self.ultra_rare_classes:
            # Generate 4-8 additional versions for ultra-rare classes
            num_versions = 6 if target_class == 8 else 4

            for i in range(num_versions):
                # Spatial augmentation
                img_aug, mask_aug = self.apply_spatial_augmentation(
                    img_tensor, mask_tensor, intensity='aggressive'
                )

                # Spectral augmentation
                img_aug = self.apply_spectral_augmentation(img_aug, intensity='aggressive')

                versions.append((img_aug, mask_aug, f"AUG_V{i+1}"))

        elif target_class in self.rare_classes:
            # Generate 2-3 versions for rare classes
            num_versions = 3

            for i in range(num_versions):
                img_aug, mask_aug = self.apply_spatial_augmentation(
                    img_tensor, mask_tensor, intensity='standard'
                )
                img_aug = self.apply_spectral_augmentation(img_aug, intensity='standard')

                versions.append((img_aug, mask_aug, f"AUG_V{i+1}"))

        return versions

# --------------------------------------------------
# 3 · Enhanced Dataset with Data Augmentation
# --------------------------------------------------
class SentinelKilnDataset(Dataset):
    def __init__(self, img_dir, mask_dir, patch_size=256, positive_ratio=0.7, augmentations=None):
        # Validierung der Verzeichnisse
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Bildverzeichnis nicht gefunden: {img_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Maskenverzeichnis nicht gefunden: {mask_dir}")

        # NEUE FUNKTION: Filtere ungültige Bild-Masken-Paare vor dem Training
        print("Filtering invalid images and masks...")
        self.img_paths = self._filter_valid_images(img_dir, mask_dir)
        assert self.img_paths, f"Keine gültigen Bilder in {img_dir} gefunden!"

        self.mask_dir   = mask_dir
        self.patch_size = patch_size
        self.positive_ratio = positive_ratio
        self.augmentations = augmentations

        # Cache kiln locations for targeted sampling
        print("Caching kiln locations for positive sampling...")
        self._cache_kiln_locations()
        print(f"Found {len(self.kiln_coords)} images with kiln pixels")

        # Berechne echte Normalisierungsstatistiken
        print("Computing real dataset statistics for normalization...")
        self.mean, self.std = self._compute_dataset_statistics()
        print(f"Dataset mean: {[f'{m:.4f}' for m in self.mean]}")
        print(f"Dataset std:  {[f'{s:.4f}' for s in self.std]}")

        # Verwende echte Statistiken für Normalisierung
        self.norm = T.Normalize(mean=self.mean, std=self.std)

    def _filter_valid_images(self, img_dir, mask_dir):
        """Filtert ungültige Bild-Masken-Paare vor dem Training"""
        all_img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        valid_paths = []
        filtered_count = 0

        print(f"Checking {len(all_img_paths)} image-mask pairs for validity...")

        for img_path in all_img_paths:
            mask_path = os.path.join(mask_dir, os.path.basename(img_path))

            try:
                # Check if mask file exists
                if not os.path.exists(mask_path):
                    print(f"FILTERED: Missing mask for {os.path.basename(img_path)}")
                    filtered_count += 1
                    continue

                # Check image validity
                img_valid, img_reason = self._is_image_valid(img_path)
                if not img_valid:
                    print(f"FILTERED: {os.path.basename(img_path)} - {img_reason}")
                    filtered_count += 1
                    continue

                # Check mask validity
                mask_valid, mask_reason = self._is_mask_valid(mask_path)
                if not mask_valid:
                    print(f"FILTERED: {os.path.basename(mask_path)} - {mask_reason}")
                    filtered_count += 1
                    continue

                # Both valid
                valid_paths.append(img_path)

            except Exception as e:
                print(f"FILTERED: {os.path.basename(img_path)} - Error: {e}")
                filtered_count += 1
                continue

        print(f"✅ Dataset filtering complete:")
        print(f"   Original images: {len(all_img_paths)}")
        print(f"   Valid images: {len(valid_paths)}")
        print(f"   Filtered out: {filtered_count}")
        print(f"   Retention rate: {len(valid_paths)/len(all_img_paths)*100:.1f}%")

        return valid_paths

    def _is_image_valid(self, img_path):
        """Prüft ob ein Bild gültig ist"""
        try:
            with rasterio.open(img_path) as src:
                img = src.read(out_dtype=np.float32)[:IN_CHANNELS]

            # Check 1: All NaN
            if np.all(np.isnan(img)):
                return False, "All NaN values"

            # Check 2: All zeros
            if np.all(img == 0):
                return False, "All zero values"

            # Check 3: Too many NaNs (>50%)
            nan_ratio = np.sum(np.isnan(img)) / img.size
            if nan_ratio > 0.5:
                return False, f"Too many NaNs ({nan_ratio*100:.1f}%)"

            # Check 4: Valid data range check
            valid_data = img[~np.isnan(img)]
            if len(valid_data) == 0:
                return False, "No valid pixels"

            # Check 5: Reasonable data range
            if valid_data.max() <= 0:
                return False, "All negative/zero values"

            # Check 6: File corruption indicators
            if img.shape[1] < 100 or img.shape[2] < 100:
                return False, f"Image too small: {img.shape}"

            return True, "Valid"

        except Exception as e:
            return False, f"Read error: {str(e)[:50]}"

    def _is_mask_valid(self, mask_path):
        """Prüft ob eine Maske gültig ist"""
        try:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)

            # Check 1: All NaN
            if np.all(np.isnan(mask)):
                return False, "All NaN values"

            # Check 2: Invalid label range
            unique_labels = np.unique(mask[~np.isnan(mask)])
            if len(unique_labels) == 0:
                return False, "No valid labels"

            # Check 3: Label range validation
            if unique_labels.min() < 0 or unique_labels.max() > 8:
                return False, f"Invalid label range: {unique_labels.min()}-{unique_labels.max()}"

            # Check 4: Only background (label 0)
            if len(unique_labels) == 1 and unique_labels[0] == 0:
                return False, "Only background pixels"

            # Check 5: File corruption
            if mask.shape[0] < 100 or mask.shape[1] < 100:
                return False, f"Mask too small: {mask.shape}"

            return True, "Valid"

        except Exception as e:
            return False, f"Read error: {str(e)[:50]}"

    def _compute_dataset_statistics(self, max_samples=50):
        """Berechnet Mean/Std für alle 8 Kanäle basierend auf einer Stichprobe der Daten"""
        print(f"Analyzing {min(max_samples, len(self.img_paths))} images for statistics...")

        # Sammle Pixel-Werte für alle Kanäle
        all_pixels = [[] for _ in range(IN_CHANNELS)]

        # Verwende eine Stichprobe der Bilder (nicht alle wegen Memory)
        sample_paths = self.img_paths[:max_samples] if len(self.img_paths) > max_samples else self.img_paths

        for i, img_path in enumerate(sample_paths):
            if i % 10 == 0:
                print(f"  Processing image {i+1}/{len(sample_paths)}")

            try:
                # Lade Bild (ohne Normalisierung)
                img = read_s2_image(img_path)  # Shape: [8, H, W]

                # Debug: Zeige Bildstatistiken für erstes Bild
                if i == 0:
                    print(f"    First image shape: {img.shape}")
                    print(f"    First image dtype: {img.dtype}")
                    print(f"    First image min/max: {img.min():.6f} / {img.max():.6f}")
                    print(f"    First image mean: {img.mean():.6f}")
                    for c in range(IN_CHANNELS):
                        channel_min = img[c].min()
                        channel_max = img[c].max()
                        channel_mean = img[c].mean()
                        print(f"    Channel {c}: min={channel_min:.6f}, max={channel_max:.6f}, mean={channel_mean:.6f}")

                # Sammle alle Pixel für jeden Kanal (downsampling für Speed)
                step = max(1, img.shape[1] // 100)  # Jeder 100. Pixel bei großen Bildern
                for c in range(IN_CHANNELS):
                    pixels = img[c, ::step, ::step].flatten()

                    # Debug für erstes Bild
                    if i == 0:
                        print(f"    Channel {c} before filtering: {len(pixels)} pixels, range: {pixels.min():.6f}-{pixels.max():.6f}")

                    # Entferne extreme Outliers (z.B. Wolken) - RELAXED FILTERING
                    # Original war zu strikt: pixels[(pixels >= 0) & (pixels <= 1)]
                    valid_pixels = pixels[(pixels >= -0.1) & (pixels <= 2.0)]  # Erweiterte Grenzen

                    if i == 0:
                        print(f"    Channel {c} after filtering: {len(valid_pixels)} pixels")
                        if len(valid_pixels) > 0:
                            print(f"    Channel {c} filtered range: {valid_pixels.min():.6f}-{valid_pixels.max():.6f}")

                    if len(valid_pixels) > 0:
                        all_pixels[c].extend(valid_pixels.tolist())

            except Exception as e:
                print(f"Warning: Could not process {img_path}: {e}")
                continue

        # Debug: Zeige wie viele Pixel gesammelt wurden
        for c in range(IN_CHANNELS):
            print(f"Channel {c}: Collected {len(all_pixels[c])} pixels total")

        # Berechne Statistiken pro Kanal
        means = []
        stds = []

        for c in range(IN_CHANNELS):
            if len(all_pixels[c]) > 100:  # Mindestens 100 Pixel
                channel_pixels = np.array(all_pixels[c])

                # Debug: Zeige rohe Statistiken
                raw_mean = float(np.mean(channel_pixels))
                raw_std = float(np.std(channel_pixels))
                print(f"Channel {c} raw stats: mean={raw_mean:.6f}, std={raw_std:.6f}")

                # Robuste Statistiken (10%-90% Perzentile um weniger Outliers zu verlieren)
                p10, p90 = np.percentile(channel_pixels, [10, 90])
                filtered_pixels = channel_pixels[(channel_pixels >= p10) & (channel_pixels <= p90)]

                if len(filtered_pixels) > 0:
                    mean_val = float(np.mean(filtered_pixels))
                    std_val = float(np.std(filtered_pixels))
                    print(f"Channel {c} filtered stats: mean={mean_val:.6f}, std={std_val:.6f}")
                else:
                    mean_val = raw_mean
                    std_val = raw_std
                    print(f"Channel {c}: Using raw stats (no pixels after percentile filtering)")

                means.append(mean_val)
                stds.append(max(std_val, 1e-6))  # Verhindere Division durch 0
            else:
                # Fallback für Sentinel-2 falls keine oder zu wenig Daten
                sentinel2_defaults = {
                    0: (0.15, 0.10),   # B1 (Coastal)
                    1: (0.12, 0.08),   # B2 (Blue)
                    2: (0.11, 0.07),   # B3 (Green)
                    3: (0.13, 0.08),   # B4 (Red)
                    4: (0.25, 0.12),   # B5 (Red Edge)
                    5: (0.35, 0.15),   # B6 (Red Edge)
                    6: (0.38, 0.16),   # B7 (Red Edge)
                    7: (0.30, 0.14),   # B8 (NIR)
                }
                default_mean, default_std = sentinel2_defaults.get(c, (0.2, 0.1))
                means.append(default_mean)
                stds.append(default_std)
                print(f"Warning: Channel {c} has only {len(all_pixels[c])} pixels, using Sentinel-2 default: mean={default_mean}, std={default_std}")

        return means, stds

    def _cache_kiln_locations(self):
        """Cache pixels with kiln classes for targeted sampling - KORRIGIERT basierend auf Label-Analyse"""
        self.kiln_coords = {}
        self.class6_coords = {}  # Für extrem seltene Klasse 6 (original 7)
        self.class5_coords = {}  # Für sehr seltene Klasse 5 (original 6)
        self.class3_coords = {}  # Für seltene Klasse 3 (original 4)

        for img_path in self.img_paths:
            mask_path = os.path.join(self.mask_dir, os.path.basename(img_path))
            try:
                with rasterio.open(mask_path) as src:
                    mask_raw = src.read(1)  # Original 0-8 Labels lesen

                # CRITICAL FIX: Masken haben bereits Labels 0-8!
                # Konvertiere korrekt: 0→-1, 1-8→0-7
                mask = torch.where(torch.from_numpy(mask_raw) == 0,
                                 torch.tensor(-1),
                                 torch.from_numpy(mask_raw) - 1)

                # Find alle Kiln-Pixel (0-7 nach Mapping)
                kiln_pixels = np.where((mask >= 0) & (mask <= 7))
                if len(kiln_pixels[0]) > 0:
                    self.kiln_coords[img_path] = list(zip(kiln_pixels[0], kiln_pixels[1]))

                # Spezielle Caches für extrem seltene Klassen
                for target_class, cache_name in [(6, 'class6_coords'), (5, 'class5_coords'), (3, 'class3_coords')]:
                    class_pixels = np.where(mask == target_class)
                    if len(class_pixels[0]) > 0:
                        getattr(self, cache_name)[img_path] = list(zip(class_pixels[0], class_pixels[1]))
                        print(f"Found {len(class_pixels[0])} class {target_class} pixels in {os.path.basename(img_path)}")

            except Exception as e:
                print(f"Warning: Could not process {mask_path}: {e}")

        print(f"Total images with class 6: {len(self.class6_coords)}")
        print(f"Total images with class 5: {len(self.class5_coords)}")
        print(f"Total images with class 3: {len(self.class3_coords)}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = read_s2_image(img_path)
        mask_path = os.path.join(self.mask_dir, os.path.basename(img_path))

        # Validierung ob Maskendatei existiert
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Maskendatei nicht gefunden: {mask_path}")

        try:
            with rasterio.open(mask_path) as src:
                mask_raw = src.read(1)  # Original 1-9 Labels lesen
        except Exception as e:
            raise RuntimeError(f"Fehler beim Lesen der Maske {mask_path}: {e}")

        # CRITICAL FIX: Konsistente Label-Behandlung basierend auf Label-Analyse
        # Debug: Prüfe ursprüngliche Label-Verteilung
        unique_raw = np.unique(mask_raw)
        if idx % 100 == 0:  # Debug output
            print(f"DEBUG Sample {idx}: Original mask labels: {unique_raw}")

        # KORRIGIERT: Masken haben bereits Labels 0-8, KEINE Subtraktion nötig für Background!
        # Korrekte Konversion: 0→-1 (background), 1-8→0-7 (classes)
        mask = torch.from_numpy(mask_raw).long()
        mask = torch.where(mask == 0, torch.tensor(-1), mask - 1)  # 0→-1, 1-8→0-7

        # Validierung der Label-Bereiche
        valid_mask = mask[mask != -1]
        if len(valid_mask) > 0:
            assert valid_mask.min() >= 0 and valid_mask.max() < NUM_CLASSES, \
                f"Invalid labels after conversion: {valid_mask.min()}-{valid_mask.max()}"

        # ERWEITERTE Targeting-Strategien basierend auf Label-Analyse
        _, H, W = img.shape

        # Strategie 1: Class 6 Super-Targeting (20% für extrem seltene Klasse 6)
        use_class6_sampling = (random.random() < 0.20 and
                              img_path in self.class6_coords and
                              len(self.class6_coords[img_path]) > 0)

        # Strategie 2: Class 5 Targeting (15% für sehr seltene Klasse 5)
        use_class5_sampling = (not use_class6_sampling and
                              random.random() < 0.15 and
                              img_path in self.class5_coords and
                              len(self.class5_coords[img_path]) > 0)

        # Strategie 3: Class 3 Targeting (10% für seltene Klasse 3)
        use_class3_sampling = (not use_class6_sampling and not use_class5_sampling and
                              random.random() < 0.10 and
                              img_path in self.class3_coords and
                              len(self.class3_coords[img_path]) > 0)

        # Strategie 4: Normale positive sampling (35% der Zeit)
        use_positive_sampling = (not any([use_class6_sampling, use_class5_sampling, use_class3_sampling]) and
                                random.random() < 0.35 and
                                img_path in self.kiln_coords and
                                len(self.kiln_coords[img_path]) > 0)

        if use_class6_sampling:
            center_y, center_x = random.choice(self.class6_coords[img_path])
            top = max(0, min(H - self.patch_size, center_y - self.patch_size // 2))
            left = max(0, min(W - self.patch_size, center_x - self.patch_size // 2))
            sampling_type = "CLASS6_ULTRA_TARGET"
        elif use_class5_sampling:
            center_y, center_x = random.choice(self.class5_coords[img_path])
            top = max(0, min(H - self.patch_size, center_y - self.patch_size // 2))
            left = max(0, min(W - self.patch_size, center_x - self.patch_size // 2))
            sampling_type = "CLASS5_SUPER_TARGET"
        elif use_class3_sampling:
            center_y, center_x = random.choice(self.class3_coords[img_path])
            top = max(0, min(H - self.patch_size, center_y - self.patch_size // 2))
            left = max(0, min(W - self.patch_size, center_x - self.patch_size // 2))
            sampling_type = "CLASS3_TARGET"
        elif use_positive_sampling:
            # Sample around kiln pixel (alle Klassen 0-7)
            center_y, center_x = random.choice(self.kiln_coords[img_path])
            top = max(0, min(H - self.patch_size, center_y - self.patch_size // 2))
            left = max(0, min(W - self.patch_size, center_x - self.patch_size // 2))
            sampling_type = "POSITIVE"
        else:
            # Random sampling (20% der Zeit)
            top = random.randint(0, H - self.patch_size)
            left = random.randint(0, W - self.patch_size)
            sampling_type = "RANDOM"

        # Extract patch
        img_patch = img[:, top:top+self.patch_size, left:left+self.patch_size]
        mask_patch = mask[top:top+self.patch_size, left:left+self.patch_size]

        # REDUCED DATA AUGMENTATION STRATEGY (less aggressive)
        if self.augmentations is not None:
            # Analyze patch for rare classes
            unique_classes = torch.unique(mask_patch[mask_patch != -1])
            rare_classes = [4, 5, 6, 7, 8]
            ultra_rare_classes = [6, 7, 8]

            has_ultra_rare = any(cls.item() in ultra_rare_classes for cls in unique_classes)
            has_rare = any(cls.item() in rare_classes for cls in unique_classes)

            # REDUCED augmentation probabilities
            if has_ultra_rare:
                # Ultra-rare classes get moderate augmentation (reduced from 90% to 60%)
                primary_class = max([cls.item() for cls in unique_classes if cls.item() in ultra_rare_classes])

                if random.random() < 0.6:  # REDUCED from 0.9
                    img_patch, mask_patch = self.augmentations.apply_spatial_augmentation(
                        img_patch, mask_patch, intensity='standard'  # REDUCED from 'aggressive'
                    )
                    img_patch = self.augmentations.apply_spectral_augmentation(
                        img_patch, intensity='standard'  # REDUCED from 'aggressive'
                    )
                    sampling_type += f"_ULTRA_AUG_C{primary_class}"

            elif has_rare:
                # Rare classes get light augmentation (reduced from 70% to 40%)
                primary_class = max([cls.item() for cls in unique_classes if cls.item() in rare_classes])

                if random.random() < 0.4:  # REDUCED from 0.7
                    img_patch, mask_patch = self.augmentations.apply_spatial_augmentation(
                        img_patch, mask_patch, intensity='standard'
                    )
                    img_patch = self.augmentations.apply_spectral_augmentation(
                        img_patch, intensity='standard'
                    )
                    sampling_type += f"_RARE_AUG_C{primary_class}"

            else:
                # Common classes get minimal augmentation (reduced from 30% to 15%)
                if random.random() < 0.15:  # REDUCED from 0.3
                    img_patch, mask_patch = self.augmentations.apply_spatial_augmentation(
                        img_patch, mask_patch, intensity='standard'
                    )
                    sampling_type += "_COMMON_AUG"

        # Debug output every 100th sample
        if idx % 100 == 0:
            unique_classes = torch.unique(mask_patch[mask_patch != -1])
            kiln_pixels = len(mask_patch[(mask_patch >= 0) & (mask_patch <= 8)])  # KORRIGIERT: 0-8
            class8_pixels = len(mask_patch[mask_patch == 8])
            total_valid = len(mask_patch[mask_patch != -1])
            print(f"Sample {idx}: {sampling_type} - Kiln: {kiln_pixels}/{total_valid}, Class 8: {class8_pixels}, Classes: {unique_classes.tolist()}")

        return self.norm(img_patch), mask_patch

# --------------------------------------------------
# 4 · UNet (8 Eingangs-Kanäle) - Vollständige Architektur
# --------------------------------------------------
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

        # Decoder (Upsampling Path)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)  # Upsample
        self.dec4 = DoubleConv(1024, 512)               # 1024 (512+512)→512

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)   # Upsample
        self.dec3 = DoubleConv(512, 256)                # 512 (256+256)→256

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)   # Upsample
        self.dec2 = DoubleConv(256, 128)                # 256 (128+128)→128

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)    # Upsample
        self.dec1 = DoubleConv(128, 64)                 # 128 (64+64)→64

        # Output Layer
        self.out = nn.Conv2d(64, NUM_CLASSES, 1)

        # Pooling
        self.pool = nn.MaxPool2d(2)

        # Class-weighted Loss Function für starkes Klassenungleichgewicht
        # KORRIGIERT: Gewichte für 8 Klassen (0-7 nach Mapping) basierend auf echten Häufigkeiten
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
        self.loss = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)

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

        # Decoder Path mit Skip Connections
        x = self.up4(x)                    # 32x32x512
        x = torch.cat([x, c4], dim=1)      # 32x32x1024
        x = self.dec4(x)                   # 32x32x512

        x = self.up3(x)                    # 64x64x256
        x = torch.cat([x, c3], dim=1)      # 64x64x512
        x = self.dec3(x)                   # 64x64x256

        x = self.up2(x)                    # 128x128x128
        x = torch.cat([x, c2], dim=1)      # 128x128x256
        x = self.dec2(x)                   # 128x128x128

        x = self.up1(x)                    # 256x256x64
        x = torch.cat([x, c1], dim=1)      # 256x256x128
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

        # Debug: Analysiere die Label-Werte (weniger verbose)
        unique_labels = torch.unique(y)
        if len(unique_labels) > 0:
            print(f"Unique labels in batch: {unique_labels.tolist()[:10]}...")  # Limit output

        # Prüfe auf ungültige Label-Werte für CrossEntropy
        if y.min() < -1 or y.max() >= NUM_CLASSES:
            print(f"ERROR: Invalid label values! min={y.min()}, max={y.max()}")
            print(f"Expected range: -1 to {NUM_CLASSES-1}")
            raise ValueError("Invalid label values detected")

        # CRITICAL FIX: Prüfe auf "nur ignore_index" Batch
        valid_labels = y[y != -1]  # Alle nicht-ignore Labels
        if len(valid_labels) == 0:
            print("WARNING: Skipping batch with only ignore labels")
            # KORRIGIERT: Return None statt dummy loss
            return None

        logits = self(x)
        loss = self.loss(logits, y)

        # NaN-Check für Loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: NaN/Inf loss detected!")
            print(f"Loss value: {loss}")
            print(f"Valid pixels in batch: {len(valid_labels)}")
            # KORRIGIERT: Return None statt dummy loss
            return None

        return loss

    def training_step(self, batch, idx):
        loss = self.shared_step(batch)
        if loss is not None:  # CRITICAL FIX: Skip dummy batches
            self.log("train_loss", loss, prog_bar=True)
            return loss
        else:
            return None  # Skip this batch

    def validation_step(self, batch, idx):
        loss = self.shared_step(batch)
        if loss is not None:  # CRITICAL FIX: Skip dummy batches
            self.log("val_loss", loss, prog_bar=True)
            return loss
        else:
            return None  # Skip this batch

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

# --------------------------------------------------
# 5 · Enhanced Training & Evaluation with Data Augmentation
# --------------------------------------------------

def build_augmented_loaders():
    """Build data loaders with advanced augmentation strategy - SIMPLIFIED"""

    # Initialize augmentation system
    augmentations = SentinelAugmentations()

    # Create training dataset WITH augmentation
    train_ds_full = SentinelKilnDataset(IMG_DIR, MASK_DIR, PATCH_SIZE,
                                       positive_ratio=0.7, augmentations=augmentations)

    # Create validation dataset WITHOUT augmentation (for fair evaluation)
    val_ds_full = SentinelKilnDataset(IMG_DIR, MASK_DIR, PATCH_SIZE,
                                     positive_ratio=0.3, augmentations=None)

    # Split dataset
    val_len = max(1, int(0.1 * len(train_ds_full)))
    train_ds, val_ds = random_split(train_ds_full, [len(train_ds_full)-val_len, val_len])

    print(f"Training samples: {len(train_ds)} (with augmentation)")
    print(f"Validation samples: {len(val_ds)} (no augmentation)")

    # SIMPLIFIED: Use standard DataLoader without complex sampling for now
    num_workers = min(4, os.cpu_count() or 2)  # Reduced workers for stability

    # Training loader with standard shuffle
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Standard shuffle instead of balanced sampler
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Validation loader with standard sampling
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader

def build_standard_loaders():
    """Build standard data loaders without augmentation for comparison"""
    ds = SentinelKilnDataset(IMG_DIR, MASK_DIR, PATCH_SIZE, augmentations=None)
    val_len = max(1, int(0.1 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds)-val_len, val_len])

    num_workers = min(8, os.cpu_count() or 4)
    kwargs = dict(batch_size=BATCH_SIZE,
                  num_workers=num_workers,
                  pin_memory=torch.cuda.is_available())

    return DataLoader(train_ds, shuffle=True, **kwargs), DataLoader(val_ds, shuffle=False, **kwargs)

# --------------------------------------------------
# 2.8 · Dataset Label Analysis Tool
# --------------------------------------------------

def analyze_dataset_labels():
    """CRITICAL: Analysiere echte Label-Verteilung vor Training"""
    print("=" * 60)
    print("CRITICAL ANALYSIS: Dataset Label Distribution")
    print("=" * 60)

    label_counts = {}
    total_pixels = 0

    mask_files = glob.glob(os.path.join(MASK_DIR, "*.tif"))
    print(f"Analyzing {len(mask_files)} mask files...")

    for i, mask_path in enumerate(mask_files[:50]):  # Sample first 50 for speed
        try:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                unique, counts = np.unique(mask, return_counts=True)
                for label, count in zip(unique, counts):
                    label_counts[label] = label_counts.get(label, 0) + count
                    total_pixels += count
        except Exception as e:
            print(f"Error reading {mask_path}: {e}")
            continue

    print("\nOriginal dataset label distribution:")
    for label in sorted(label_counts.keys()):
        percentage = (label_counts[label] / total_pixels) * 100
        print(f"  Label {label}: {label_counts[label]:,} pixels ({percentage:.4f}%)")

    print(f"\nTotal analyzed pixels: {total_pixels:,}")

    # Check for critical issues
    if 0 not in label_counts:
        print("WARNING: No background pixels (label 0) found!")
    if max(label_counts.keys()) > 9:
        print(f"ERROR: Labels > 9 found! Max label: {max(label_counts.keys())}")
    if min(label_counts.keys()) < 0:
        print(f"ERROR: Negative labels found! Min label: {min(label_counts.keys())}")

    print("=" * 60)
    return label_counts

# ==================== MAIN TRAINING EXECUTION ====================

print("Starting Enhanced Training with Data Augmentation...")
print("=" * 60)

# CRITICAL: Analyze dataset labels before training
print("🔍 STEP 1: Analyzing dataset label distribution...")
label_distribution = analyze_dataset_labels()
print("=" * 60)

# Choose training strategy:
USE_AUGMENTATION = False  # CRITICAL FIX: Start with False for baseline test

if USE_AUGMENTATION:
    print("✅ Using ENHANCED AUGMENTATION strategy")
    train_loader, val_loader = build_augmented_loaders()
    model_suffix = "_augmented"
else:
    print("📊 Using STANDARD training for baseline (RECOMMENDED FOR FIRST TEST)")
    train_loader, val_loader = build_standard_loaders()
    model_suffix = "_baseline"

print(f"📈 Train batches: {len(train_loader)}")
print(f"📊 Validation batches: {len(val_loader)}")
print("=" * 60)

# Modell und Training konfigurieren
model = UNet()
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
