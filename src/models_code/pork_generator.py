import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Encoder block: uses LeakyReLU as before
# -------------------------------
class pix2pix_conv_block(nn.Module):
    """
    A double convolution block using 3x3 filters.
    Uses LeakyReLU activations (negative slope 0.2).
    Optionally applies BatchNorm.

    The first convolution outputs 'base_channels'
    and the second outputs 'base_channels * 2'.
    """
    def __init__(self, in_channels, base_channels, use_batchnorm=True):
        super(pix2pix_conv_block, self).__init__()
        layers = []
        # First convolution: outputs base_channels
        layers.append(nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=True))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(base_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        # Second convolution: outputs base_channels * 2
        layers.append(nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1, bias=True))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(base_channels * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

# -------------------------------
# Decoder conv block: uses only ReLU
# -------------------------------
class pix2pix_conv_block_decoder(nn.Module):
    """
    A double convolution block for the decoder using 3x3 filters.
    Uses ReLU activations.
    Optionally applies BatchNorm.

    The first convolution outputs 'base_channels'
    and the second outputs 'base_channels * 2'.
    """
    def __init__(self, in_channels, base_channels, use_batchnorm=True):
        super(pix2pix_conv_block_decoder, self).__init__()
        layers = []
        # First convolution: outputs base_channels
        layers.append(nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=True))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(base_channels))
        layers.append(nn.ReLU(inplace=True))
        # Second convolution: outputs base_channels * 2
        layers.append(nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1, bias=True))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(base_channels * 2))
        layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

# -------------------------------
# Upsampling block (decoder): already uses ReLU
# -------------------------------
class pix2pix_up_conv_decoder(nn.Module):
    """
    Decoder block with a 4x4 transpose convolution followed by BatchNorm and ReLU.
    This block does not double the filter size during decoding.
    """
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super(pix2pix_up_conv_decoder, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.up = nn.Sequential(*layers)

    def forward(self, x):
        return self.up(x)

# -------------------------------
# Modified UNetPix2PixGenerator
# -------------------------------
class UNetGenerator(nn.Module):
    def __init__(self, img_channels=1, output_channels=1):
        super(UNetGenerator, self).__init__()
        # Encoder (input resolution: 48x48)
        self.enc1 = pix2pix_conv_block(img_channels, 4, use_batchnorm=False)  # Output: 8 channels, 48x48
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                     # 48 -> 24
        self.enc2 = pix2pix_conv_block(8, 8, use_batchnorm=True)                # Output: 16 channels, 24x24
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                     # 24 -> 12
        self.enc3 = pix2pix_conv_block(16, 16, use_batchnorm=True)              # Output: 32 channels, 12x12

        # Reduce latent channels from 32 to 16:
        self.reduce = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)

        # Decoder
        self.up1 = pix2pix_up_conv_decoder(16, 16, use_dropout=True)           # 12x12 -> 24x24
        self.reduce1 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)    # Reduce channels before dec1
        self.dec1 = pix2pix_conv_block_decoder(16, 16, use_batchnorm=True)      # Uses ReLU; concatenated with enc2 (16+16)

        self.up2 = pix2pix_up_conv_decoder(32, 8, use_dropout=True)            # 24x24 -> 48x48 (input from dec1 is 32 channels)
        self.reduce2 = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0)     # Reduce channels before dec2
        self.dec2 = pix2pix_conv_block_decoder(8, 8, use_batchnorm=True)        # Uses ReLU; concatenated with enc1 (8+8)

        self.final_conv = nn.Conv2d(16, output_channels, kernel_size=1, stride=1, padding=0)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)       # [B, 8, 48, 48]
        p1 = self.pool1(x1)     # [B, 8, 24, 24]
        x2 = self.enc2(p1)      # [B, 16, 24, 24]
        p2 = self.pool2(x2)     # [B, 16, 12, 12]
        x3 = self.enc3(p2)      # [B, 32, 12, 12]  <- Latent space before reduction
        x3 = self.reduce(x3)    # [B, 16, 12, 12]  <- Reduced latent space

        # Decoder
        up1 = self.up1(x3)      # [B, 16, 24, 24]
        cat1 = torch.cat([up1, x2], dim=1)  # [B, 32, 24, 24]
        cat1 = self.reduce1(cat1)  # Reduce channels from 32 to 16 [B, 16, 24, 24]
        dec1 = self.dec1(cat1)     # [B, 32, 24, 24] (decoder conv block doubles channels)

        up2 = self.up2(dec1)    # [B, 8, 48, 48]
        cat2 = torch.cat([up2, x1], dim=1)  # [B, 16, 48, 48]
        cat2 = self.reduce2(cat2)  # Reduce channels from 16 to 8 [B, 8, 48, 48]
        dec2 = self.dec2(cat2)     # [B, 16, 48, 48]

        out = self.final_conv(dec2)   # [B, output_channels, 48, 48]
        out = self.final_activation(out)
        return out

# -------------------------------
# (Optional) Weight Initialization Function
# -------------------------------
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in classname or 'Linear' in classname):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
