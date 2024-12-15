import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms
from torch.utils.data import random_split
from model import Net
from utils.checkpoint import CheckpointManager
from tqdm import tqdm
import time
import json
from pathlib import Path

def count_parameters(model):
    """Count and print trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    table = [["Layer", "Parameters"]]
    for name, p in model.named_parameters():
        if p.requires_grad:
            table.append([name, str(format(p.numel(), ","))])
    
    # Print table
    print("\nModel Parameter Count:")
    print("-" * 50)
    for row in table:
        print(f"{row[0]:<40} {row[1]:>10}")
    print("-" * 50)
    print(f"Total Trainable Parameters: {total_params:,}")
    print("-" * 50 + "\n")
    return total_params

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, batch_idx, num_batches):
    if epoch < 5:  # First 5 epochs for warmup
        total_steps = 5 * num_batches
        current_step = epoch * num_batches + batch_idx
        lr = 0.01 * current_step / total_steps
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def save_results(epoch, train_loss, val_loss, val_acc, test_loss, test_acc, results_file='results.json'):
    """Save training results to a JSON file"""
    result = {
        'epoch': epoch,
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'val_accuracy': float(val_acc),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc)
    }
    
    # Load existing results or create new list
    if Path(results_file).exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = []
    
    # Append new result
    results.append(result)
    
    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

def train_model():
    # Data Loading and Splitting
    transform_train = transforms.Compose([
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
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load full training dataset
    full_train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform_train)
    
    # Split into train and validation (50 samples for validation)
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [len(full_train_dataset) - 50, 50],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Load test dataset (10k samples)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform_test)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Create model and print parameter count
    model = Net().to(device)
    num_params = count_parameters(model)
    if num_params > 20_000:
        print(f"Warning: Model has {num_params:,} parameters (exceeds 20k limit)!")
    print("-" * 50)
    
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
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(model, optimizer)
    
    # Training Loop
    best_accuracy = 0
    results = []
    
    for epoch in range(20):
        # Training phase
        model.train()
        losses = AverageMeter()
        batch_times = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        end = time.time()
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = F.nll_loss(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            losses.update(loss.item(), data.size(0))
            batch_times.update(time.time() - end)
            end = time.time()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'batch_time': f'{batch_times.avg:.3f}s'
            })
        
        # Validation phase (50 samples)
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_accuracy = 100. * val_correct / len(val_loader.dataset)
        
        # Test phase (10k samples)
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * test_correct / len(test_loader.dataset)
        
        # After computing metrics
        print(f'\nEpoch {epoch}:')
        print(f'Training Loss: {losses.avg:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')
        print('-' * 50)
        
        # Save results
        save_results(
            epoch=epoch,
            train_loss=losses.avg,
            val_loss=val_loss,
            val_acc=val_accuracy,
            test_loss=test_loss,
            test_acc=test_accuracy
        )
        
        # Save checkpoint every 5 epochs or if best accuracy
        if (epoch + 1) % 5 == 0 or test_accuracy > best_accuracy:
            checkpoint_manager.save(epoch, test_loss, test_accuracy)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                print(f'New best accuracy: {best_accuracy:.2f}%')
        
        scheduler.step()
    
    # Print final results
    print('\nTraining completed!')
    print(f'Best Test Accuracy: {best_accuracy:.2f}%')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model() 