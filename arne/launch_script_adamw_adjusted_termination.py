import os
import sys
from simon.simple_model.unet import SimpleUNet
print(f"Python version: {sys.version}", flush=True)
from data_loader import load_split
import numpy as np
import torch
from utils import *
from performance_analysis import calculate_and_log_performance
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

print_available_memory()

STORAGE_PATH = "/scratch/tmp/tstraus2/"
CREATOR_NAME = "arne"
DATA_PATH = "/scratch/tmp/tstraus2/Brick_Data_Train/"


SEED = 42  # For reproducibility
LR_ADAM = 1e-4  # Learning rate for Adam optimizer
WD_ADAM = 5e-5 
BATCH_SIZE = 8  # Batch size for training
MAX_EPOCHS = 200  # Number of epochs for training
TERMINATTE_EARLY = 20  # Early stopping patience in epochs now based on 
#LR_DECAY_EPOCHS = 100


MODEL_NAME = (
    f"v3_unet_adamW_weightedLoss2_seed{SEED}_lr{LR_ADAM}_wd{WD_ADAM}_bs{BATCH_SIZE}_epochs{MAX_EPOCHS}"  # every run needs a unique name to avoid overwriting previous runs
)
if __name__ == "__main__":
    print("Initializing Simple UNet model...", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = SimpleUNet()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    model = model.to(device)

    print("Model initialized successfully.", flush=True)
    model_dir = validate_paths(STORAGE_PATH, CREATOR_NAME, MODEL_NAME)

    print("writer initialization...", flush=True)
    writer = SummaryWriter(f'{STORAGE_PATH}/{CREATOR_NAME}/tensorboard_logs/{MODEL_NAME}')


    print("Initializing loss function and optimizer...", flush=True)
    # Gegebene relative Häufigkeiten
    relative_class_distribution = {
        "0": 0.9992917404276939,
        "1": 0.00023072790429295306,
        "2": 0.0003373913337268344,
        "3": 5.163629370999624e-05,
        "4": 1.370890871428422e-05,
        "5": 2.5589622487345214e-05,
        "6": 6.1842013235232595e-06,
        "7": 4.705814632226824e-06,
        "8": 3.8315493418988456e-05
    }

    # Frequenzen in korrekter Reihenfolge in ein numpy-Array
    class_frequencies = np.array([relative_class_distribution[str(i)] for i in range(9)])
    # Log-inverse Gewichtung mit Stabilizer alpha
    alpha = 1.01 
    log_inv_weights = 1.0 / np.log(alpha + class_frequencies)
    # Optional: Normieren, damit Summe 1 ist (oder Summe = num_classes)
    normalized_weights = log_inv_weights / log_inv_weights.sum()
    # In Torch-Tensor umwandeln
    class_weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32).to(device)
    # In Loss-Funktion übergeben
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_ADAM, weight_decay=WD_ADAM)
    

    print_available_memory()
    print("Loading data...", flush=True)

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

    print(f"Training data loaded: {len(X_train)} samples", flush=True)
    print(f"Test data loaded: {len(X_test)} samples", flush=True)

    np.nan_to_num(X_train, copy=False, nan=0.0)
    np.nan_to_num(X_test, copy=False, nan=0.0)

    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print_available_memory()
    best_val_weighted_f1 = float("-inf")
    start_epoch = 0
    epochs_no_improve = 0  # Track epochs without improvement

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
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        


        model.eval()
        eval_loss = 0.0
        num_classes = model.final_conv.out_channels
        # Collect all predictions and true labels
        all_preds = []
        all_labels = []


        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                outputs = model(xb)
                loss = loss_fn(outputs, yb)
                eval_loss += loss.item() * xb.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy().reshape(-1))
                all_labels.extend(yb.cpu().numpy().reshape(-1))
                
        avg_eval_loss = eval_loss / len(test_dataset)
        print(f"Epoch {epoch+1}/10, Eval Loss: {avg_eval_loss:.4f}", flush=True)
        writer.add_scalar('Loss/Val', avg_eval_loss, epoch)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        print("Performance metrics will be calculated", flush=True)
        val_weighted_f1 =calculate_and_log_performance(all_labels=all_labels, all_preds=all_preds, num_classes=num_classes, epoch=epoch, writer=writer)
        print("Performance metrics calculated and logged.", flush=True)        
        print(MODEL_NAME, flush=True)
        print(f"val_weighted{val_weighted_f1}", flush=True)
        print(f"best_val_weighted_f1: {best_val_weighted_f1}", flush=True)
  
        if val_weighted_f1 > best_val_weighted_f1:
            best_val_weighted_f1 = val_weighted_f1
            best_weights_path = os.path.join(model_dir, f"weights_best_model.pth")
            
            
            save_model_state(model.state_dict(), best_weights_path)
            print(
                f"Best model and state saved at {best_weights_path} (f1_weighted: {best_val_weighted_f1:.4f})",
                flush=True,
            )
            epochs_no_improve = 0  # Reset counter on improvement
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= TERMINATTE_EARLY:
            print(f"Early stopping: No improvement in {TERMINATTE_EARLY} epochs.", flush=True)
            break

        model.train()
        print_available_memory()
    
    writer.close()

    weights_path = os.path.join(model_dir, "final_weights.pth")
    save_model_state(model.state_dict(), weights_path)
    print(f"Model weights and state saved at {weights_path}", flush=True)
