from unet import UNet  # Import a different model
from train import train  # Import the train function from train.py
from torchvision import transforms

# Hyperparameters
batch_size = 64
epochs = 10
learning_rate = 1e-4
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((48, 48))
])

# S3 bucket and folders
s3_bucket = 'khmer-historical-manuscript'
artifacts_s3_folder = 'artifacts/'
logs_s3_folder = 'logs/'

# Train the model
train(UNet, input_channels=1, output_channels=10, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, transform=transform, s3_bucket=s3_bucket, logs_dir='/opt/ml/code/logs', artifacts_s3_folder=artifacts_s3_folder, logs_s3_folder=logs_s3_folder)
