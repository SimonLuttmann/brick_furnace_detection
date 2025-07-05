import os
import sys

print(f"Python version: {sys.version}", flush=True)
from ...data_loader import load_split
import numpy as np
import torch
from ...utils import *

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

print_available_memory()
MODEL_NAME = (
    "simple_unet_v2"  # every run needs a unique name to avoid overwriting previous runs
)
STORAGE_PATH = "/scratch/tmp/sluttman/"
CREATOR_NAME = "simon_luttmann"
DATA_PATH = "/scratch/tmp/sluttman/Brick_Data_Train/"


SEED = 42  # For reproducibility
LR_ADAM = 1e-4  # Learning rate for Adam optimizer
BATCH_SIZE = 8  # Batch size for training
MAX_EPOCHS = 30  # Number of epochs for training


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=8, num_classes=9):
        super(SimpleUNet, self).__init__()

        # Encoder (Contracting Path)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder (Expanding Path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)  # 512 + 512 from skip connection

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)  # 256 + 256 from skip connection

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)  # 128 + 128 from skip connection

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)  # 64 + 64 from skip connection

        # Final output layer
        self.final_conv = nn.Conv2d(64, num_classes, 1)

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

    def conv_block(self, in_channels, out_channels):
        """Simple convolutional block: Conv -> ReLU -> Conv -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # 256x256x64
        enc1_pool = self.pool(enc1)  # 128x128x64

        enc2 = self.enc2(enc1_pool)  # 128x128x128
        enc2_pool = self.pool(enc2)  # 64x64x128

        enc3 = self.enc3(enc2_pool)  # 64x64x256
        enc3_pool = self.pool(enc3)  # 32x32x256

        enc4 = self.enc4(enc3_pool)  # 32x32x512
        enc4_pool = self.pool(enc4)  # 16x16x512

        # Bottleneck
        bottleneck = self.bottleneck(enc4_pool)  # 16x16x1024

        # Decoder with skip connections
        up4 = self.upconv4(bottleneck)  # 32x32x512
        concat4 = torch.cat([up4, enc4], dim=1)  # 32x32x1024
        dec4 = self.dec4(concat4)  # 32x32x512

        up3 = self.upconv3(dec4)  # 64x64x256
        concat3 = torch.cat([up3, enc3], dim=1)  # 64x64x512
        dec3 = self.dec3(concat3)  # 64x64x256

        up2 = self.upconv2(dec3)  # 128x128x128
        concat2 = torch.cat([up2, enc2], dim=1)  # 128x128x256
        dec2 = self.dec2(concat2)  # 128x128x128

        up1 = self.upconv1(dec2)  # 256x256x64
        concat1 = torch.cat([up1, enc1], dim=1)  # 256x256x128
        dec1 = self.dec1(concat1)  # 256x256x64

        # Final output
        output = self.final_conv(dec1)  # 256x256x9

        return output


if __name__ == "__main__":
    print("Initializing Simple UNet model...", flush=True)
    # Set seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = SimpleUNet()

    # Check for GPU/CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    print("Model initialized successfully.", flush=True)

    # Prepare paths
    model_dir = validate_paths(STORAGE_PATH, CREATOR_NAME, MODEL_NAME)

    # --- Training loop for demonstration ---
    # Suggestion: Use CrossEntropyLoss for multi-class segmentation
    print("Initializeing loss function and optimizer...", flush=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_ADAM)
    print("Loss function and optimizer initialized.", flush=True)
    print()
    print_available_memory()
    print("Loading data...", flush=True)
    # Load data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        image_paths_train,
        image_paths_test,
        label_paths_train,
        label_paths_test,
    ) = load_split(DATA_PATH, load_into_ram=True, verbose=True, as_numpy=True)
    print(f"Training data loaded: {len(X_train)} samples")
    print(f"Test data loaded: {len(X_test)} samples", flush=True)

    # Clean NaNs in X_train, X_test, y_train, y_test
    print("Cleaning NaNs in training and test data...", flush=True)
    np.nan_to_num(X_train, copy=False, nan=0.0)
    np.nan_to_num(X_test, copy=False, nan=0.0)
    print("NaN cleaning done.", flush=True)
    # Move channels to first dimension for PyTorch: (N, 8, 256, 256)
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))
    # Convert to torch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()
    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("after dataloader initialization")
    print_available_memory()
    best_eval_loss = float("inf")
    start_epoch = 0

    # --- Training loop ---
    print("Starting training...", flush=True)
    model.train()
    for epoch in range(start_epoch, MAX_EPOCHS):
        epoch_loss = 0.0
        batch_count = 0
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = loss_fn(outputs, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            batch_count += 1
            if batch_count % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{MAX_EPOCHS}, Batch {batch_count}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )
        avg_loss = epoch_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}", flush=True)

        # --- Evaluation phase ---
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                outputs = model(xb)
                loss = loss_fn(outputs, yb)
                eval_loss += loss.item() * xb.size(0)
        avg_eval_loss = eval_loss / len(test_dataset)
        print(f"Epoch {epoch+1}/10, Eval Loss: {avg_eval_loss:.4f}", flush=True)

        # Save best model and training state
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            best_weights_path = os.path.join(model_dir, f"weights_best_model.pth")
            # Save model, optimizer, and epoch only
            state_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_eval_loss": best_eval_loss,
            }
            save_model_state(state_dict, best_weights_path)

            print(
                f"Best model and state saved at {best_weights_path} (Eval Loss: {best_eval_loss:.4f})",
                flush=True,
            )

        model.train()

    # Save final model weights and state
    weights_path = os.path.join(model_dir, "final_weights.pth")
    state_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": MAX_EPOCHS,
        "best_eval_loss": best_eval_loss,
    }
    save_model_state(state_dict, weights_path)
    print(f"Model weights and state saved at {weights_path}")
