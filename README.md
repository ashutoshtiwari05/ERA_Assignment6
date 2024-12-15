# MNIST Classification with PyTorch

[![Model Checks](https://github.com/ashutosh-tiwari15/ERAV3/actions/workflows/model_checks.yml/badge.svg)](https://github.com/ashutosh-tiwari15/ERAV3/actions/workflows/model_checks.yml)

A CNN implementation for MNIST digit classification achieving >99.4% accuracy with less than 20k parameters in less than 20 epochs.

## Test Results & Logs 📊

### Quick Links
- 📈 [Detailed Test Logs](Assignment_6/results.json) - Complete training history and test metrics
- 🔍 [Latest Test Run](https://github.com/ashutosh-tiwari15/ERAV3/actions) - Most recent model check results
- 📊 [Test Implementation](Assignment_6/tests/test_training.py) - Test suite code

### Latest Metrics
```json
{
    "final_test_accuracy": "99.42%",
    "final_test_loss": 0.0198,
    "best_validation_accuracy": "99.38%",
    "total_parameters": 18752,
    "total_epochs": 15,
    "training_time": "8m 42s"
}
```

### Training History
- Epoch 15: Test Acc=99.42%, Loss=0.0198
- Epoch 14: Test Acc=99.36%, Loss=0.0205
- Epoch 13: Test Acc=99.31%, Loss=0.0213
- Epoch 12: Test Acc=99.28%, Loss=0.0221

## Latest Results

### Test Performance
```
Test Accuracy: 99.42%
Test Loss: 0.0198
```

### Model Stats
- Total Parameters: 18,752
- Training Time: 15 epochs
- Best Validation Accuracy: 99.38%
- Peak Learning Rate: 0.1
- Final Learning Rate: 1e-4

### Key Files
- [Test Implementation](tests/test_training.py)
- [Model Architecture Tests](tests/test_model.py)
- [Training Logs](results.json)
- [Model Checkpoints](checkpoints/best_model.pth)

## Project Overview

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification with specific constraints:
- Parameters: < 20k
- Target Accuracy: > 99.4%
- Training Duration: < 20 epochs
- Dataset Split: Training (59,950) / Validation (50) / Test (10,000)

## Architecture

### Model Structure
- Input Block: Conv(1→12→16) + BatchNorm + ReLU + Dropout
- First Block: Conv(16→20) + BatchNorm + ReLU + MaxPool
- Transition Block: Conv(20→24) + BatchNorm + ReLU + Dropout
- Second Block: Conv(24→24) + BatchNorm + ReLU + MaxPool
- Output Block: Conv(24→20→10) + GAP

### Key Components
- MaxPooling: Used for spatial dimension reduction
- Batch Normalization: After each convolution
- Dropout: Progressive (0.05 → 0.1)
- Learning Rate: OneCycleLR scheduler
- Parameters: ~18k total

## Training Configuration

### Data Augmentation
```python
transforms.Compose([
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
```

### Optimizer Settings
```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True
)

scheduler = OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=20,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    div_factor=25,
    final_div_factor=1e4,
    anneal_strategy='cos'
)
```

## Project Structure
```
.
├── README.md
├── model.py              # Neural network architecture
├── train.py             # Training script
├── utils/
│   └── checkpoint.py    # Checkpoint management
├── tests/
│   ├── test_model.py    # Model architecture tests
│   └── test_training.py # Training pipeline tests
├── .github/
│   ├── workflows/       # GitHub Actions
│   └── scripts/         # CI/CD scripts
├── requirements.txt     # Dependencies
├── Makefile            # Build automation
└── environment.py      # Environment verification
```

## Setup and Usage

1. Create and activate virtual environment:
```bash
make setup
source venv-a6/bin/activate  # Unix
# or
.\venv-a6\Scripts\activate  # Windows
```

2. Verify environment:
```bash
make env-update
```

3. Run tests:
```bash
make test
```

4. Start training:
```bash
make train
```

5. View results:
```bash
make view-results
```

## Available Commands

- `make setup`: Create virtual environment and install dependencies
- `make train`: Run model training
- `make test`: Run all tests
- `make test-model`: Run model architecture tests
- `make test-training`: Run training pipeline tests
- `make check-model`: Check model architecture requirements
- `make check-params`: Verify parameter count
- `make view-results`: Display best training results
- `make clean`: Clean Python cache files
- `make clean-checkpoints`: Remove saved checkpoints

## Monitoring and Results

Training progress includes:
- Real-time progress bar with loss and batch time
- Per-epoch metrics:
  - Training loss
  - Validation loss and accuracy (50 samples)
  - Test loss and accuracy (10k samples)
- Automatic checkpoint management (keeps last 5)
- Results logging to JSON for analysis

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- tqdm
- pytest
- numpy

## License

MIT License