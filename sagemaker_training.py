import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
import os

def create_sagemaker_estimator():
    role = get_execution_role()

    # Set up the SageMaker PyTorch Estimator
    estimator = PyTorch(
        entry_point="src/train.py",           # Training script to execute
        source_dir=".",                       # Directory with your code
        role=role,                            # IAM role
        instance_type="ml.p3.2xlarge",        # GPU instance type (adjust as needed)
        instance_count=1,                     # Number of instances
        framework_version="1.10",             # PyTorch version
        py_version="py38",                    # Python version
        script_mode=True,                     # Running the training script in script mode
        hyperparameters={                     # Optional: Include hyperparameters here
            'batch_size': 64,
            'epochs': 10,
        },
        dependencies=["requirements.txt"],     # Dependencies to install in the training container
    )
    
    return estimator

if __name__ == "__main__":
    estimator = create_sagemaker_estimator()

    # Submit the training job
    estimator.fit(wait=True)
