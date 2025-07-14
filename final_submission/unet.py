import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=9, base_features=64):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, base_features)
        self.enc2 = self._conv_block(base_features, base_features*2)
        self.enc3 = self._conv_block(base_features*2, base_features*4)
        self.enc4 = self._conv_block(base_features*4, base_features*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._conv_block(base_features*8, base_features*16)
        self.up4 = nn.ConvTranspose2d(base_features*16, base_features*8, 2, stride=2)
        self.dec4 = self._conv_block(base_features*16, base_features*8)
        self.up3 = nn.ConvTranspose2d(base_features*8, base_features*4, 2, stride=2)
        self.dec3 = self._conv_block(base_features*8, base_features*4)
        self.up2 = nn.ConvTranspose2d(base_features*4, base_features*2, 2, stride=2)
        self.dec2 = self._conv_block(base_features*4, base_features*2)
        self.up1 = nn.ConvTranspose2d(base_features*2, base_features, 2, stride=2)
        self.dec1 = self._conv_block(base_features*2, base_features)
        self.final = nn.Conv2d(base_features, out_channels, 1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True)
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