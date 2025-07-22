import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

##############################################
# 1. Define Basic Building Blocks
##############################################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
        )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class AttentionGate(nn.Module):
    """
    Attention Gate as in Attention U-Net.
    g: gating signal (from decoder, coarser features)
    x: skip connection features (from encoder, finer features)
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # Multiply attention coefficients with the encoder features
        return x * psi

##############################################
# 2. Define the Attention Residual U-Net
##############################################

class AttentionResidualUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        """
        A simplified U-Net with residual blocks and attention gates on skip connections.
        """
        super(AttentionResidualUNet, self).__init__()
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: create residual blocks and save skip connection outputs
        prev_channels = in_channels
        for feature in features:
            self.encoder_blocks.append(ResidualBlock(prev_channels, feature))
            prev_channels = feature

        # Bottleneck: an extra residual block
        self.bottleneck = ResidualBlock(features[-1], features[-1]*2)

        # Decoder: upsampling layers, attention gates, and residual blocks
        self.upconv_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        rev_features = features[::-1]
        decoder_in_channels = features[-1]*2
        for feature in rev_features:
            self.upconv_blocks.append(
                nn.ConvTranspose2d(decoder_in_channels, feature, kernel_size=2, stride=2)
            )
            # Attention gate: gating signal from decoder and skip connection from encoder
            self.attention_gates.append(
                AttentionGate(F_g=feature, F_l=feature, F_int=feature // 2)
            )
            # After concatenation, channels double
            self.decoder_blocks.append(
                ResidualBlock(feature * 2, feature)
            )
            decoder_in_channels = feature

        # Final convolution to get desired output channels
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for enc in self.encoder_blocks:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse for decoder

        for idx in range(len(self.upconv_blocks)):
            x = self.upconv_blocks[idx](x)
            skip_connection = skip_connections[idx]
            # Apply attention gate on skip connection features
            att_gate = self.attention_gates[idx]
            skip_connection = att_gate(g=x, x=skip_connection)
            # Concatenate skip connection features with upsampled features
            x = torch.cat([skip_connection, x], dim=1)
            x = self.decoder_blocks[idx](x)

        return torch.sigmoid(self.final_conv(x))
