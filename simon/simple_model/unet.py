import torch

import torch.nn as nn

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