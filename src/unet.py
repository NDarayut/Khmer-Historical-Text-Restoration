import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UNet, self).__init__()

        # Encoder (Downsampling)
        self.encoder1 = self.conv_block(input_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (Upsampling)
        self.decoder4 = self.upconv_block(1024, 512)
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, 64)
        
        # Skip connection channel reducers
        self.reduce_d4 = self.conv_block(1024, 512)
        self.reduce_d3 = self.conv_block(512, 256)
        self.reduce_d2 = self.conv_block(256, 128)
        self.reduce_d1 = self.conv_block(128, 64)

        # Output layer
        self.output = nn.Conv2d(64, output_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block
    
    def upconv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Decoder
        d4 = self.decoder4(b)
        d4 = torch.cat([d4, e4], dim=1)  # Concatenate skip connection
        d4 = self.reduce_d4(d4)  # Use predefined reducer

        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.reduce_d3(d3)  # Use predefined reducer

        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.reduce_d2(d2)  # Use predefined reducer

        d1 = self.decoder1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.reduce_d1(d1)  # Use predefined reducer
        
        # Output layer
        out = self.output(d1)
        
        return out
