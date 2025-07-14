import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import kornia
import tifffile as tiff
from tqdm import tqdm
from sklearn.metrics import f1_score
from unet import UNet


class ImageDataset(Dataset):
    def __init__(self, img_dir, msk_dir):
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.images  = sorted(os.listdir(img_dir))
        self.masks   = sorted(os.listdir(msk_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = tiff.imread(os.path.join(self.img_dir, self.images[idx])).astype(np.float32)
        msk = tiff.imread(os.path.join(self.msk_dir, self.masks[idx])).astype(np.int64)
        # NaN/Inf → 0
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        # Normalisierung
        mi, ma = img.min(), img.max()
        if ma > mi:
            img = (img - mi) / (ma - mi)
        # Kanäle vorne
        img = np.moveaxis(img, -1, 0)
        
        return torch.from_numpy(img), torch.from_numpy(msk)

# IoU-Metrik
def compute_iou(preds, targets, num_classes=9):
    p = preds.view(-1); t = targets.view(-1)
    ious = []
    for c in range(1, num_classes):
        pi = (p==c); ti = (t==c)
        inter = (pi & ti).sum().item()
        uni   = (pi | ti).sum().item()
        if uni>0: ious.append(inter/uni)
    return np.mean(ious) if ious else 0.0


def dice_loss(logits, targets, eps=1e-6):
    C = logits.shape[1]
    oh = F.one_hot(targets, C).permute(0,3,1,2).float()
    p  = F.softmax(logits, dim=1)
    inter = torch.sum(p * oh, dim=(0,2,3))
    union = torch.sum(p + oh, dim=(0,2,3))
    dice  = (2*inter + eps)/(union + eps)
    return 1 - dice[1:].mean()

def boundary_loss(logits, targets):
    with torch.no_grad():
        C = logits.shape[1]
        oh = F.one_hot(targets, C).permute(0,3,1,2).float()
        gm = oh[:,1:,:,:].sum(1,True)
        ge = kornia.filters.sobel(gm)
    prob = F.softmax(logits,1)
    pm = prob[:,1:,:,:].sum(1,True)
    pe = kornia.filters.sobel(pm)
    return F.binary_cross_entropy(pe, ge)

def ce_dice_boundary_loss(logits, targets, weight=None, ce_w=0.6, dice_w=0.4, bnd_w=0.2):
    ce   = F.cross_entropy(logits, targets, weight=weight)
    dice = dice_loss(logits, targets)
    bnd  = boundary_loss(logits, targets)
    return ce_w*ce + dice_w*dice + bnd_w*bnd

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = UNet().to(device)
    # Gewichtung für CE
    class_probs = torch.tensor([
    0.9992773,  # Background
    0.0002316,  # Brick_Furnace_1
    0.0003396,  # Brick_Furnace_2
    0.0000540,  # Brick_Furnace_3
    0.0000149,  # Brick_Furnace_4
    0.0000261,  # Brick_Furnace_5
    0.0000059,  # Brick_Furnace_6
    0.0000045,  # Brick_Furnace_7
    0.0000458   # Brick_Furnace_8
], device=device)
    weights = 1/torch.log(1.02+class_probs)
    weights = weights/weights.sum()*len(weights)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-8
    )

    # Direkte Pfade für Train/Test
    train_img = '/scratch/tmp/sluttman/Brick_Data_Train/Split/Image/train'
    train_msk = '/scratch/tmp/sluttman/Brick_Data_Train/Split/Mask/train'
    test_img  = '/scratch/tmp/sluttman/Brick_Data_Train/Split/Image/test'
    test_msk  = '/scratch/tmp/sluttman/Brick_Data_Train/Split/Mask/test'

    train_ds = ImageDataset(train_img, train_msk)
    test_ds  = ImageDataset(test_img,  test_msk)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False,num_workers=8, pin_memory=True)

    best_f1 = 0.0
    for epoch in range(1, 211):
        # Training
        model.train(); tloss=0.0
        for imgs, msks in tqdm(train_loader, desc=f'Epoch {epoch} Training'):
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = ce_dice_boundary_loss(out, msks, weight=weights)
            loss.backward(); optimizer.step(); tloss+=loss.item()
        
        # Test-Evaluation
        model.eval(); vloss=0.0; ious=[]; f1s=[]
        with torch.no_grad():
            for imgs, msks in tqdm(test_loader, desc=f'Epoch {epoch} Test'):
                imgs, msks = imgs.to(device), msks.to(device)
                out = model(imgs)
                vloss += ce_dice_boundary_loss(out, msks, weight=weights).item()
                preds = out.argmax(1)
                ious.append(compute_iou(preds, msks))
                y_true = msks.view(-1).cpu().numpy()
                y_pred = preds.view(-1).cpu().numpy()
                mask   = y_true>0
                if mask.sum()>0:
                    f1s.append(f1_score(y_true[mask], y_pred[mask], average='weighted', zero_division=0))
        avg_iou = np.mean(ious); avg_f1=np.mean(f1s) if f1s else 0.0
        print(f'Epoch {epoch}/210 | Train Loss: {tloss/len(train_loader):.4f} | '
              f'Test Loss: {vloss/len(test_loader):.4f} | IoU: {avg_iou:.4f} | F1: {avg_f1:.4f}')
        scheduler.step(epoch)
        if avg_f1>best_f1:
            best_f1=avg_f1
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_unet_256.pth')
            print(f"New best F1: {best_f1:.4f}")
    torch.save(model.state_dict(), 'checkpoints/unet_noaug_256_final.pth')
