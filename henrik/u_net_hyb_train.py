import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import tifffile as tiff
from tqdm import tqdm
import kornia
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- ConvNeXt-Stage ---
class ConvNeXtStage(nn.Module):
    def __init__(self, dim, depth, drop_path_rate=0.0):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
                nn.GroupNorm(1, dim),
                nn.Conv2d(dim, 4 * dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(4 * dim, dim, kernel_size=1),
                nn.Dropout2d(drop_path_rate),
            ))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)

# --- Hybrid U-Net++ mit Deep Supervision ---
class HybridUNetPP(nn.Module):
    def __init__(self, in_ch=8, base_features=32, num_classes=9,
                 bottleneck_depth=3, bottleneck_block_depth=2, drop_path_rate=0.2):
        super().__init__()
        dims = [base_features * (2 ** i) for i in range(4)]
        # Encoder
        self.enc1 = self.conv_block(in_ch, dims[0])        # 64x64
        self.enc2 = self.conv_block(dims[0], dims[1])      # 32x32
        self.enc3 = self.conv_block(dims[1], dims[2])      # 16x16
        self.enc4 = self.conv_block(dims[2], dims[3])      # 8x8
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[
            ConvNeXtStage(dims[3], depth=bottleneck_block_depth, drop_path_rate=drop_path_rate)
            for _ in range(bottleneck_depth)
        ])  # 8x8

        # Decoder mit Deep Supervision
        self.up3 = nn.ConvTranspose2d(dims[3], dims[2], kernel_size=2, stride=2)
        self.dec3 = self.conv_block(dims[2] + dims[3], dims[2])
        self.aux_out3 = nn.Conv2d(dims[2], num_classes, kernel_size=1)  # 16x16

        self.up2 = nn.ConvTranspose2d(dims[2], dims[1], kernel_size=2, stride=2)
        self.dec2 = self.conv_block(dims[1] + dims[2], dims[1])
        self.aux_out2 = nn.Conv2d(dims[1], num_classes, kernel_size=1)  # 32x32

        self.up1 = nn.ConvTranspose2d(dims[1], dims[0], kernel_size=2, stride=2)
        self.dec1 = self.conv_block(dims[0] + dims[1], dims[0])
        self.aux_out1 = nn.Conv2d(dims[0], num_classes, kernel_size=1)  # 64x64

        # Final Output (direkt 64x64)
        self.final = nn.Conv2d(dims[0], num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                     # 64x64
        e2 = self.enc2(self.pool(e1))         # 32x32
        e3 = self.enc3(self.pool(e2))         # 16x16
        e4 = self.enc4(self.pool(e3))         # 8x8

        # Bottleneck
        b = self.bottleneck(e4)               # 8x8

        # Decoder
        d3 = self.up3(b)                      # 16x16
        e4_up = F.interpolate(e4, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e4_up], dim=1))  # 16x16
        aux3 = self.aux_out3(d3)              # 16x16

        d2 = self.up2(d3)                     # 32x32
        e3_up = F.interpolate(e3, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e3_up], dim=1))  # 32x32
        aux2 = self.aux_out2(d2)              # 32x32

        d1 = self.up1(d2)                     # 64x64
        e2_up = F.interpolate(e2, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e2_up], dim=1))  # 64x64
        aux1 = self.aux_out1(d1)              # 64x64

        # Final Output
        final_out = self.final(d1)            # 64x64

        return final_out, aux3, aux2, aux1



# --- Augmentations ---
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.05),
    A.OneOf([
        A.ChannelDropout(channel_drop_range=(1, 1), p=0.1),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, p=0.1)
    ], p=0.1),
    A.Resize(64, 64),
    ToTensorV2()
], is_check_shapes=False)

val_transform = A.Compose([
    A.Resize(64, 64),
    ToTensorV2()
])

# Dataset
class PatchDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, in_channels=8):
        self.images    = sorted(os.listdir(image_dir))
        self.masks     = sorted(os.listdir(mask_dir))
        self.img_dir   = image_dir
        self.msk_dir   = mask_dir
        self.transform = transform
        self.in_ch     = in_channels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        img = tiff.imread(os.path.join(self.img_dir, self.images[idx])).astype(np.float32)
        msk = tiff.imread(os.path.join(self.msk_dir, self.masks[idx])).astype(np.int64)
        # Replace NaN/Inf, normalize
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        # Ensure HWC format
        if img.ndim == 3 and img.shape[0] == self.in_ch:
            # CHW -> HWC
            img_hwc = np.transpose(img, (1, 2, 0))
        elif img.ndim == 3:
            # assume HWC
            img_hwc = img
        else:
            # single-channel -> HWC
            img_hwc = np.expand_dims(img, axis=-1)
        if self.transform:
            aug = self.transform(image=img_hwc, mask=msk)
            image_t = aug['image']  # Tensor CHW with correct channels
            mask_t = torch.as_tensor(aug['mask'], dtype=torch.long)
            return image_t, mask_t
        # fallback without transform
        img_chw = np.moveaxis(img_hwc, -1, 0)
        return torch.from_numpy(img_chw), torch.from_numpy(msk).long()

# --- IoU-Metric ---
def compute_iou(preds, targets, num_classes=9):
    preds, targets = preds.view(-1), targets.view(-1)
    ious = []
    for c in range(num_classes):
        inter = ((preds == c) & (targets == c)).sum().item()
        union = ((preds == c) | (targets == c)).sum().item()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0

# Loss-Funktion
def dice_loss(logits, targets, eps=1e-6):
    C = logits.shape[1]
    t_onehot = F.one_hot(targets, C).permute(0,3,1,2).float()
    p = F.softmax(logits, dim=1)
    inter = torch.sum(p * t_onehot, dim=(0,2,3))
    union = torch.sum(p + t_onehot, dim=(0,2,3))
    dice  = (2 * inter + eps) / (union + eps)
    # dice[0] ist Background, den wir hier ggf. ignorieren:
    return 1.0 - dice[1:].mean()

def boundary_loss(logits, targets):
    """
    Boundary-Loss:
    - gt_edges: Sobel-Kanten der One-Hot–gt
    - pred_edges: Sobel-Kanten der Summe aller Nicht-BG-Wahrscheinlichkeiten
    """
    # 1) Ground-Truth–Kanten (B×1×H×W)
    with torch.no_grad():
        # Ein-Hot–Encoding der GT (B×C×H×W)
        C = logits.shape[1]
        gt_onehot = F.one_hot(targets, C).permute(0,3,1,2).float()
        # Randbild: Sobel auf Hintergrund ausgeschlossene Klassen summiert
        gt_mask = gt_onehot[:,1:,:,:].sum(dim=1, keepdim=True)
        gt_edges = kornia.filters.sobel(gt_mask)

    # 2) pred_edges
    probas    = F.softmax(logits, dim=1)
    pred_mask = probas[:,1:,:,:].sum(dim=1, keepdim=True)
    pred_edges= kornia.filters.sobel(pred_mask)

    # 3) logit‐Transformation (clamp nötig!)
    le = torch.logit(pred_edges.clamp(1e-6,1-1e-6))

    # 4) autocast-sichere BCE
    return F.binary_cross_entropy_with_logits(le, gt_edges)


def ce_dice_boundary_loss(logits, targets, weight=None,
                         ce_w=0.6, dice_w=0.4, bnd_w=0.2):
    ce   = F.cross_entropy(logits, targets, weight=weight)
    dice = dice_loss(logits, targets)
    bnd  = boundary_loss(logits, targets)
    return ce_w*ce + dice_w*dice + bnd_w*bnd

# --- Main ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=210)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridUNetPP(in_ch=8, base_features=32).to(device)

    class_probs = torch.tensor(
        [0.9897, 0.00332, 0.00500, 0.00071, 0.00020, 0.00036, 0.00010, 0.00006, 0.00055],
        device=device, dtype=torch.float32
    )
    weights = (1 / torch.log(1.02 + class_probs)).float()
    weights = weights / weights.sum() * len(weights)
    criterion = lambda logits, targets: ce_dice_boundary_loss(
        logits, targets,
        weight=weights,
        ce_w=0.6,    # wie stark CrossEntropy
        dice_w=0.4,  # wie stark Dice
        bnd_w=0.2    # wie stark Boundary Loss
    )
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=30,       # Länge des ersten Zyklus in Epochen
    T_mult=2,     # Zyklusverdopplung danach: 30→60→120…
    eta_min=1e-8  # minimale Lernrate
    )
    scaler = GradScaler()

    img_dir = "/scratch/tmp/sluttman/Brick_Patches_Filtered/Image"
    msk_dir = "/scratch/tmp/sluttman/Brick_Patches_Filtered/Mask"
    
    # Split und jeweils mit den passenden Augmentations laden
    full_ds = PatchDataset(img_dir, msk_dir, transform=None)  # Dummy transform zum Split
    tlen = int(0.8 * len(full_ds))
    vlen = len(full_ds) - tlen
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_ds)), [tlen, vlen], generator=torch.Generator().manual_seed(42)
    )

    # Jetzt mit transform laden
    train_ds = torch.utils.data.Subset(
        PatchDataset(img_dir, msk_dir, transform=train_transform), train_indices
    )
    val_ds = torch.utils.data.Subset(
        PatchDataset(img_dir, msk_dir, transform=val_transform), val_indices
    )

    # DataLoader erstellen
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Training Loop ---
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, msks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} Training"):
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            with autocast():
                final_out, aux3, aux2, aux1 = model(imgs)

                # Upsample auxiliary outputs to match target size
                aux3_up = F.interpolate(aux3, size=msks.shape[1:], mode='bilinear', align_corners=False)
                aux2_up = F.interpolate(aux2, size=msks.shape[1:], mode='bilinear', align_corners=False)
                aux1_up = F.interpolate(aux1, size=msks.shape[1:], mode='bilinear', align_corners=False)

                # Weighted Deep Supervision Loss
                loss = (1.0 * criterion(final_out, msks)
                        + 0.5 * criterion(aux1_up, msks)
                        + 0.3 * criterion(aux2_up, msks)
                        + 0.2 * criterion(aux3_up, msks))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        torch.cuda.empty_cache()

        # --- Validation ---
        model.eval()
        val_loss, ious, f1s = 0.0, [], []
        with torch.no_grad():
            for imgs, msks in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} Validation"):
                imgs, msks = imgs.to(device), msks.to(device)
                with autocast():
                    final_out, aux3, aux2, aux1 = model(imgs)

                    # Upsample auxiliary outputs
                    aux3_up = F.interpolate(aux3, size=msks.shape[1:], mode='bilinear', align_corners=False)
                    aux2_up = F.interpolate(aux2, size=msks.shape[1:], mode='bilinear', align_corners=False)
                    aux1_up = F.interpolate(aux1, size=msks.shape[1:], mode='bilinear', align_corners=False)

                    # Weighted Deep Supervision Loss
                    val_loss_batch = (1.0 * criterion(final_out, msks)
                                      + 0.5 * criterion(aux1_up, msks)
                                      + 0.3 * criterion(aux2_up, msks)
                                      + 0.2 * criterion(aux3_up, msks))
                    val_loss += val_loss_batch.item()

                    preds = final_out.argmax(dim=1)
                    ious.append(compute_iou(preds, msks))
                    y_true = msks.view(-1).cpu().numpy()
                    y_pred = preds.view(-1).cpu().numpy()
                    mask = y_true > 0
                    if mask.sum() > 0:
                        f1s.append(f1_score(y_true[mask], y_pred[mask], average='weighted', labels=np.unique(y_true[mask])))

        avg_iou = np.mean(ious)
        avg_f1 = np.mean(f1s)
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | IoU: {avg_iou:.4f} | F1: {avg_f1:.4f}")

        scheduler.step(epoch)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            os.makedirs("/scratch/tmp/sluttman/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "/scratch/tmp/sluttman/checkpoints/best_hybrid_unetpp.pth")
            print(f"🚀 New best F1: {best_f1:.4f}")

    torch.save(model.state_dict(), "/scratch/tmp/sluttman/checkpoints/hybrid_unetpp_final.pth")
