import boto3
import pandas as pd
import torch
from io import StringIO
from PIL import Image
import numpy as np
from torchvision import transforms

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, train_csv_s3_key, target_csv_s3_key, s3_bucket, transform=None):
        self.s3_client = boto3.client('s3')
        self.s3_bucket = s3_bucket
        self.train_csv_s3_key = train_csv_s3_key
        self.target_csv_s3_key = target_csv_s3_key
        self.transform = transform
        
        # Fetch train CSV
        train_obj = self.s3_client.get_object(Bucket=s3_bucket, Key=train_csv_s3_key)
        train_data = train_obj['Body'].read().decode('utf-8')
        self.train_df = pd.read_csv(StringIO(train_data))  # Read without header
        
        # Fetch target CSV
        target_obj = self.s3_client.get_object(Bucket=s3_bucket, Key=target_csv_s3_key)
        target_data = target_obj['Body'].read().decode('utf-8')
        self.target_df = pd.read_csv(StringIO(target_data))  # Read without header

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        # Fetch noisy image (train.csv)
        noisy_image_row = self.train_df.iloc[idx].values  # Get pixel values as numpy array
        # Fetch corresponding clean image (target.csv)
        clean_image_row = self.target_df.iloc[idx].values  # Get pixel values as numpy array

        # Ensure rows are the correct size
        if len(noisy_image_row) != 48 * 48 or len(clean_image_row) != 48 * 48:
            raise ValueError(
                f"Unexpected row sizes: noisy={len(noisy_image_row)}, clean={len(clean_image_row)}. Expected 2304."
            )
        
        # Reshape to 48x48
        noisy_image = noisy_image_row.reshape(48, 48).astype(np.uint8)
        clean_image = clean_image_row.reshape(48, 48).astype(np.uint8)
        
        # Convert numpy arrays to PIL Images
        noisy_image = Image.fromarray(noisy_image)
        clean_image = Image.fromarray(clean_image)

        # Apply transformations (if provided)
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        noisy_image = torch.tensor(noisy_image, dtype=torch.float)
        clean_image = torch.tensor(clean_image, dtype=torch.float)

        return noisy_image, clean_image
