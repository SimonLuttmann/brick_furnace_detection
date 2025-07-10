import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import kornia
import tifffile as tiff
from tqdm import tqdm
from sklearn.metrics import f1_score

# 🧱 U-Net Architecture
class UNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=9, base_features=64):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, base_features)
        self.enc2 = self._conv_block(base_features, base_features * 2)
        self.enc3 = self._conv_block(base_features * 2, base_features * 4)
        self.enc4 = self._conv_block(base_features * 4, base_features * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._conv_block(base_features * 8, base_features * 16)
        self.up4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, 2, stride=2)
        self.dec4 = self._conv_block(base_features * 16, base_features * 8)
        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_features * 8, base_features * 4)
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_features * 4, base_features * 2)
        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, 2, stride=2)
        self.dec1 = self._conv_block(base_features * 2, base_features)
        self.final = nn.Conv2d(base_features, out_channels, 1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

# Dataset (ohne Augmentations)
class PatchDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.images  = sorted(os.listdir(image_dir))
        self.masks   = sorted(os.listdir(mask_dir))
        self.img_dir = image_dir
        self.msk_dir = mask_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = tiff.imread(os.path.join(self.img_dir, self.images[idx])).astype(np.float32)
        msk = tiff.imread(os.path.join(self.msk_dir, self.masks[idx])).astype(np.int64)
        # NaN/Inf ersetzen + Normierung
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        # Kanal-Achse nach vorne
        img = np.moveaxis(img, -1, 0)
        return torch.from_numpy(img), torch.from_numpy(msk).long()

# IoU-Metrik
def compute_iou(preds, targets, num_classes=9):
    preds   = preds.view(-1)
    targets = targets.view(-1)
    ious = []
    for c in range(num_classes):
        p = preds == c
        t = targets == c
        inter = (p & t).sum().item()
        uni   = (p | t).sum().item()
        if uni > 0:
            ious.append(inter / uni)
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

    # 2) Pred-Kanten (B×1×H×W)
    probas = F.softmax(logits, dim=1)
    pred_mask = probas[:,1:,:,:].sum(dim=1, keepdim=True)
    pred_edges = kornia.filters.sobel(pred_mask)

    # 3) BCE zwischen diesen Kanten
    return F.binary_cross_entropy(pred_edges, gt_edges)

#def ce_dice_loss(logits, targets, weight=None, ce_weight=1.0, dice_weight=1.0):
#    ce_term   = F.cross_entropy(logits, targets, weight=weight)
#    dice_term = dice_loss(logits, targets)
#    return ce_weight * ce_term + dice_weight * dice_term

def ce_dice_boundary_loss(logits, targets, weight=None,
                         ce_w=0.6, dice_w=0.4, bnd_w=0.2):
    ce   = F.cross_entropy(logits, targets, weight=weight)
    dice = dice_loss(logits, targets)
    bnd  = boundary_loss(logits, targets)
    return ce_w*ce + dice_w*dice + bnd_w*bnd

# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train basic U-Net (no augmentations)')
    parser.add_argument('--epochs',     type=int, default=210)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = UNet().to(device)

    # Loss + Optimizer
    class_probs = torch.tensor(
        [0.9897, 0.00332, 0.00500, 0.00071, 0.00020, 0.00036, 0.00010, 0.00006, 0.00055],
        device=device
    )
    weights = 1 / torch.log(1.02 + class_probs)
    weights = weights / weights.sum() * len(weights)
#    criterion = lambda logits, targets: ce_dice_loss(
#    logits, targets,
#    weight=weights,
#    ce_weight=0.6,    # wie stark CrossEntropy
#    dice_weight=0.4   # wie stark Dice
#    )
    criterion = lambda logits, targets: ce_dice_boundary_loss(
        logits, targets,
        weight=weights,
        ce_w=0.6,    # wie stark CrossEntropy
        dice_w=0.4,  # wie stark Dice
        bnd_w=0.2    # wie stark Boundary Loss
    )
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
#    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-8, verbose=True
#    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=30,       # Länge des ersten Zyklus in Epochen
    T_mult=2,     # Zyklusverdopplung danach: 30→60→120…
    eta_min=1e-8  # minimale Lernrate
    )

    # Daten laden
    img_dir = "/scratch/tmp/sluttman/Brick_Patches_Filtered/Image"
    msk_dir = "/scratch/tmp/sluttman/Brick_Patches_Filtered/Mask"
    ds      = PatchDataset(img_dir, msk_dir)
    tlen    = int(0.8 * len(ds))
    vlen    = len(ds) - tlen
    train_ds, val_ds = random_split(
        ds, [tlen, vlen], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for imgs, msks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} Training"):
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            out   = model(imgs)
            loss  = criterion(out, msks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validierung
        model.eval()
        val_loss = 0.0
        ious, f1s = [], []
        with torch.no_grad():
            for imgs, msks in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} Validation"):
                imgs, msks = imgs.to(device), msks.to(device)
                out = model(imgs)
                val_loss += criterion(out, msks).item()
                preds = out.argmax(dim=1)
                ious.append(compute_iou(preds, msks))
                y_true = msks.view(-1).cpu().numpy()
                y_pred = preds.view(-1).cpu().numpy()
                mask   = y_true > 0
                f1s.append(f1_score(y_true[mask], y_pred[mask], average='weighted', labels=np.unique(y_true[mask])))

        avg_iou = np.mean(ious)
        avg_f1 = np.mean(f1s)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f} | "
            f"IoU: {avg_iou:.4f} | F1: {avg_f1:.4f}"
        )

#       scheduler.step(avg_f1)
        scheduler.step(epoch)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.state_dict(), "/scratch/tmp/sluttman/checkpoints/best_unet_noaug.pth")
            print(f"🚀 New best F1: {best_f1:.4f}")

    os.makedirs("/scratch/tmp/sluttman/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "/scratch/tmp/sluttman/checkpoints/unet_noaug_final.pth")
