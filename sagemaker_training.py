import sagemaker
from sagemaker.pytorch import PyTorch

# Set up the SageMaker session
sagemaker_session = sagemaker.Session()

# Get the execution role
role = "arn:aws:iam::867718131774:role/SageMaker_train"

# Define the PyTorch estimator
estimator = PyTorch(
    entry_point='script.py',  # The training script
    source_dir='./src',       # Directory containing your scripts
    role=role,
    framework_version='2.0.1',  # Adjust according to your PyTorch version
    py_version='py310',           # Python version, adjust as needed
    instance_count=1,
    instance_type='ml.p3.2xlarge',  # GPU instance
    output_path='s3://khmer-historical-manuscript/output/',  # Output location for model artifacts
    sagemaker_session=sagemaker_session
)

# Start the training job
estimator.fit({'training': 's3://khmer-historical-manuscript/train'})
