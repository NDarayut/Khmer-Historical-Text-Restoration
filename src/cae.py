import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),  # Conv2D layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # MaxPooling2D layer (adjusted padding)

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # Adjusted padding

            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # Adjusted padding
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Upsampling

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'), 

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'), 

            nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1),  # Output layer
            nn.Sigmoid()  # Sigmoid for pixel values in [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)  # Pass through the encoder
        x = self.decoder(x)  # Pass through the decoder
        return x
