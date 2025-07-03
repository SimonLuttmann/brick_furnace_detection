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
IMG_DIR  = os.path.join(DATA_ROOT, "Image")      # Korrigiert: "Image" statt "images"
MASK_DIR = os.path.join(DATA_ROOT, "Mask")       # Korrigiert: "Mask" statt "mask"
PATCH_SIZE  = 256
BATCH_SIZE  = 8
LR          = 1e-4          # Reduzierte Lernrate für Stabilität
EPOCHS      = 50            # Erhöht für bessere Konvergenz
NUM_CLASSES = 10
IN_CHANNELS = 8          # <── neu: 8 Kanäle
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

    Sentinel‑2‑GeoTIFFs können in einigen Bändern (z. B. durch Wolkenmasken
    oder fehlende Datenstreifen) `NaN`‑Pixel enthalten. Für ein CNN sind
    NaNs fatal, weil sie sich bei jeder Operation fortpflanzen → am Ende
    wird der gesamte Tensor zu `NaN`.  Wir konvertieren fehlende
    Werte daher explizit zu `0` — ein neutraler, „nicht‑informierender"
    Wert nach der DN‑Skalierung.
    """
    try:
        with rasterio.open(path) as src:
            img = src.read(out_dtype=np.float32)[:IN_CHANNELS]
    except Exception as e:
        raise RuntimeError(f"Fehler beim Lesen von {path}: {e}")

    # DN‑Werte nach ESA‑Empfehlung auf 0‑1 skalieren
    img = img / 10000.0

    # Alle NaNs durch 0 ersetzen, damit Torch‑Operationen stabil bleiben
    img = np.nan_to_num(img, nan=0.0)
    return torch.from_numpy(img)

# --------------------------------------------------
# 3 · Dataset
# --------------------------------------------------
class SentinelKilnDataset(Dataset):
    def __init__(self, img_dir, mask_dir, patch_size=256, positive_ratio=0.7):
        # Validierung der Verzeichnisse
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Bildverzeichnis nicht gefunden: {img_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Maskenverzeichnis nicht gefunden: {mask_dir}")

        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        assert self.img_paths, f"Keine Bilder in {img_dir}!"
        self.mask_dir   = mask_dir
        self.patch_size = patch_size
        self.positive_ratio = positive_ratio

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
        """Cache pixels with kiln classes for targeted sampling"""
        self.kiln_coords = {}
        for img_path in self.img_paths:
            mask_path = os.path.join(self.mask_dir, os.path.basename(img_path))
            try:
                with rasterio.open(mask_path) as src:
                    mask = src.read(1) - 1  # Convert to 0-9 range

                # Find kiln pixels (classes 1-9)
                kiln_pixels = np.where((mask >= 1) & (mask <= 9))
                if len(kiln_pixels[0]) > 0:
                    self.kiln_coords[img_path] = list(zip(kiln_pixels[0], kiln_pixels[1]))
            except Exception as e:
                print(f"Warning: Could not process {mask_path}: {e}")

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
                mask = torch.from_numpy(src.read(1)).long() - 1  # -1…9
        except Exception as e:
            raise RuntimeError(f"Fehler beim Lesen der Maske {mask_path}: {e}")

        # Targeted sampling strategy
        _, H, W = img.shape

        # Debug sampling decision
        use_positive_sampling = (random.random() < self.positive_ratio and
                                img_path in self.kiln_coords and
                                len(self.kiln_coords[img_path]) > 0)

        # Targeted sampling for kilns (70% of the time)
        if use_positive_sampling:
            # Sample around kiln pixel
            center_y, center_x = random.choice(self.kiln_coords[img_path])
            top = max(0, min(H - self.patch_size, center_y - self.patch_size // 2))
            left = max(0, min(W - self.patch_size, center_x - self.patch_size // 2))
            sampling_type = "POSITIVE"
        else:
            # Random sampling (30% of the time)
            top = random.randint(0, H - self.patch_size)
            left = random.randint(0, W - self.patch_size)
            sampling_type = "RANDOM"

        # Extract patch
        img_patch = img[:, top:top+self.patch_size, left:left+self.patch_size]
        mask_patch = mask[top:top+self.patch_size, left:left+self.patch_size]

        # Debug output every 100th sample
        if idx % 100 == 0:
            unique_classes = torch.unique(mask_patch[mask_patch != -1])
            kiln_pixels = len(mask_patch[(mask_patch >= 1) & (mask_patch <= 9)])
            total_valid = len(mask_patch[mask_patch != -1])
            print(f"Sample {idx}: {sampling_type} sampling - Kiln pixels: {kiln_pixels}/{total_valid}, Classes: {unique_classes.tolist()}")

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
        # Berechnete Gewichte basierend auf Klassenverteilung
        class_weights = torch.tensor([
            0.1,   # Background (häufig)
            3.0,   # Kiln classes (selten)
            4.0, 4.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0
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
        return loss

    def validation_step(self, batch, idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

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
# 5 · Training & Evaluation
# --------------------------------------------------

def build_loaders():
    ds = SentinelKilnDataset(IMG_DIR, MASK_DIR, PATCH_SIZE)
    val_len = max(1, int(0.1 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds)-val_len, val_len])

    # Begrenzte Worker-Anzahl für Stabilität (max 8)
    num_workers = min(8, os.cpu_count() or 4)

    kwargs = dict(batch_size=BATCH_SIZE,
                  num_workers=num_workers,
                  pin_memory=torch.cuda.is_available())
    return DataLoader(train_ds, shuffle=True, **kwargs), DataLoader(val_ds, shuffle=False, **kwargs)

# Initialisierung der DataLoader
train_loader, val_loader = build_loaders()
print(f"Train: {len(train_loader)}  Val: {len(val_loader)}")

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
torch.save(model.state_dict(), "unet_kiln_sentinel2_8ch.pt")
print("Gewichte gespeichert → unet_kiln_sentinel2_8ch.pt")
