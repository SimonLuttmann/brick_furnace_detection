# TODO
1. [x] weighted F1 Score without background implementation
1. [ ]  Loss Function with weighted Classes (inverse of class share of total pixel) 
1. [x] AdamW?
1. [x] AdamW second parameter weight decay?
1. [x] LR Decay function  optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
1. [ ] Data Augmentation
1. [ ] Batch Normalization
1. [ ] Batch Size


Train Models
1. [ ] Raw training of U-Net simply adjust LR, Batch Size, 
1. [ ] Training of U-Net with weighted Loss Fuction adjust LR, Batch Size





1. [ ] ask chatgpt about adamw and weight loss function



Do
256x256 2560m * 2560m = 6553600 m²
64x64 640 * 640 = 409600 m² 
Brick Killn ~ 500 m² max

1. [x] check for black pixels if there are prediction made -> yes there are just very few. We will ignore them
2. [x] try adamw without sceduler and try differet weight decay values
1. [ ] try with weighted loss function with unet
2. [ ] create data augmentated images of train dataset (rotation)
3. [ ] switch termination criteria to weighted f1 val
