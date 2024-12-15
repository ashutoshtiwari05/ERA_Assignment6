import sys
from pathlib import Path
import pytest
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from model import Net

def test_model_parameters():
    """Test that model has less than 20k parameters"""
    model = Net()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count < 20000, f"Model has {param_count:,} parameters (exceeds 20k limit)"

def test_model_architecture():
    """Test model architecture requirements"""
    model = Net()
    
    # Check for BatchNorm
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model should use Batch Normalization"
    
    # Check for Dropout
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should use Dropout"
    
    # Check for GAP
    has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())
    assert has_gap, "Model should use Global Average Pooling"

def test_model_output():
    """Test model output shape and type"""
    model = Net()
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    
    assert output.shape == (batch_size, 10), f"Expected output shape (4, 10), got {output.shape}"
    assert torch.allclose(output.exp().sum(1), torch.ones(batch_size)), "Output should be log-softmax" 