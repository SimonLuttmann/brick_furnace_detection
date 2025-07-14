import torch
#from simon.simple_model.unet import SimpleUNet
#from henrik.u_net_base_train256 import UNet
from Tim.DetectionV4_final import EnhancedUNet
from data_loader import *
from torch.utils.data import TensorDataset, DataLoader
from performance_analysis import *
import numpy as np
DATA_PATH = "/scratch/tmp/sluttman/Brick_Data_Train/"


model = EnhancedUNet()
BATCH_SIZE=8
model.load_state_dict(torch.load('enhanced_unet_v4_final_stabilized_kiln_sentinel2.pt'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
model = model.to(device)

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


print(f"Test data loaded: {len(X_test)} samples", flush=True)


np.nan_to_num(X_test, copy=False, nan=0.0)


X_test = np.transpose(X_test, (0, 3, 1, 2))

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()

print("before dataloader", flush=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.eval()
eval_loss = 0.0
num_classes = 9
# Collect all predictions and true labels
all_preds = []
all_labels = []


with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        outputs = model(xb)
               
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy().reshape(-1))
        all_labels.extend(yb.cpu().numpy().reshape(-1))
        


all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print("Performance metrics will be calculated", flush=True)
val_weighted_f1 =calculate_and_log_performance(all_labels=all_labels, all_preds=all_preds, num_classes=num_classes, epoch=1, writer=None)
print("Performance metrics calculated and logged.", flush=True)        