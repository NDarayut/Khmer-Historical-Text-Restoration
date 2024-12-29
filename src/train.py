import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import boto3
from io import StringIO
from torchvision import transforms
from data import CustomDataset

# Utility functions to handle training
def save_model(model, model_path, s3_client, s3_bucket, artifacts_s3_folder):
    # Save model locally
    torch.save(model.state_dict(), model_path)
    # Upload model to S3
    s3_client.upload_file(model_path, s3_bucket, os.path.join(artifacts_s3_folder, os.path.basename(model_path)))
    print(f'Model artifact saved to S3 at: s3://{s3_bucket}/{artifacts_s3_folder}/{os.path.basename(model_path)}')

def save_logs(log_file_local, s3_client, s3_bucket, logs_s3_folder):
    s3_client.upload_file(log_file_local, s3_bucket, os.path.join(logs_s3_folder, 'training_log_1.log'))
    print(f'Logs saved to S3 at: s3://{s3_bucket}/{logs_s3_folder}/training_log_1.log')

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, epochs, log_file_local):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Compute loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Logging
        log_message = f'Epoch [{epoch}/{epochs}], Step [{batch_idx}/{len(train_loader)}], MSE Loss: {loss.item()}'
        print(log_message)
        
        # Save log message to file
        with open(log_file_local, 'a') as f:
            f.write(log_message + '\n')

def setup_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_data_loader(s3_bucket, batch_size, transform):
    # Prepare dataset and DataLoader
    train_dataset = CustomDataset(
        train_csv_s3_key='train/train.csv', 
        target_csv_s3_key='train/target.csv', 
        s3_bucket=s3_bucket, 
        transform=transform
    )
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def setup_model(model_class, input_channels, output_channels, device):
    model = model_class(input_channels=input_channels, output_channels=output_channels).to(device)
    return model

def setup_optimizer(model, learning_rate):
    return optim.Adam(model.parameters(), lr=learning_rate)

def setup_criterion():
    return nn.MSELoss()

def train(model_class, input_channels, output_channels, batch_size, epochs, learning_rate, transform, s3_bucket, logs_dir, artifacts_s3_folder, logs_s3_folder):
    # Initialize SageMaker environment variables
    input_path = os.environ.get('SM_CHANNEL_TRAINING', '/train')
    output_path = os.environ.get('SM_MODEL_DIR', '/model')

    # Device setup
    device = setup_device()

    # Initialize log and model paths
    log_dir = logs_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file_local = os.path.join(log_dir, 'training_log_1.log')
    open(log_file_local, 'w').close()

    model = setup_model(model_class, input_channels, output_channels, device)
    criterion = setup_criterion()
    optimizer = setup_optimizer(model, learning_rate)

    # Data loading
    train_loader = setup_data_loader(s3_bucket, batch_size, transform)

    # S3 client setup
    s3_client = boto3.client('s3')

    # Training loop
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, epochs, log_file_local)

    # Saving and uploading artifacts
    model_file_local = '/opt/ml/model/cae_v1.pth'  # Saving model to the model directory
    save_model(model, model_file_local, s3_client, s3_bucket, artifacts_s3_folder)
    save_logs(log_file_local, s3_client, s3_bucket, logs_s3_folder)


# Example of how to use this modular code for training
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 128
    epochs = 10
    learning_rate = 1e-3
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((48, 48))
    ])
    
    # S3 bucket and folders
    s3_bucket = 'khmer-historical-manuscript'
    artifacts_s3_folder = 'artifacts/'
    logs_s3_folder = 'logs/'
