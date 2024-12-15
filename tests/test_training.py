import sys
from pathlib import Path
import pytest
import torch
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from train import train_model, save_results
from model import Net

def test_save_results(tmp_path):
    """Test results saving functionality"""
    results_file = tmp_path / "test_results.json"
    
    # Save some test results
    save_results(
        epoch=0,
        train_loss=0.5,
        val_loss=0.4,
        val_acc=98.0,
        test_loss=0.3,
        test_acc=99.0,
        results_file=results_file
    )
    
    # Check if file exists and content is correct
    assert results_file.exists()
    with open(results_file) as f:
        results = json.load(f)
    
    assert len(results) == 1
    assert results[0]['test_accuracy'] == 99.0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_to_device():
    """Test model device placement"""
    model = Net().to('cuda')
    assert next(model.parameters()).is_cuda

def test_data_transforms():
    """Test data augmentation transforms"""
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    
    # Create a dummy PIL image
    img = Image.fromarray(np.uint8(np.ones((28, 28)) * 255))
    
    transform = transforms.Compose([
        transforms.RandomRotation((-15, 15)),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1), 
            scale=(0.9, 1.1),
            shear=(-5, 5)
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.2)
    ])
    
    # Apply transforms
    transformed = transform(img)
    
    # Check output
    assert isinstance(transformed, torch.Tensor), "Output should be a tensor"
    assert transformed.shape == (1, 28, 28), "Transform should maintain image shape"
    assert transformed.dtype == torch.float32, "Output should be float32"