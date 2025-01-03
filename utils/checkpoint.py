import os
import torch
from pathlib import Path
import re

class CheckpointManager:
    def __init__(self, model, optimizer, save_dir='checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.best_accuracy = 0.0
        
    def _get_epoch_number(self, checkpoint_path):
        """Extract epoch number from checkpoint filename"""
        match = re.search(r'epoch_(\d+)\.pt', str(checkpoint_path))
        return int(match.group(1)) if match else -1
        
    def save(self, epoch, loss, accuracy):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy
        }
        
        # Save regular checkpoint
        path = self.save_dir / f'checkpoint_epoch_{epoch:02d}.pt'
        torch.save(checkpoint, path)
        
        # Save best model if accuracy improved
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            best_path = self.save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'accuracy': accuracy,
                'loss': loss
            }, best_path)
        
        # Keep only last 5 checkpoints using numeric sorting
        checkpoints = sorted(
            [p for p in self.save_dir.glob('checkpoint_epoch_*.pt')],
            key=self._get_epoch_number
        )
        if len(checkpoints) > 5:
            os.remove(checkpoints[0])
            
    def load_latest(self):
        checkpoints = sorted(
            [p for p in self.save_dir.glob('checkpoint_epoch_*.pt')],
            key=self._get_epoch_number
        )
        if not checkpoints:
            return None
            
        checkpoint = torch.load(checkpoints[-1])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint

    def load_best(self):
        """Load the best model checkpoint"""
        best_path = self.save_dir / 'best_model.pth'
        if not best_path.exists():
            return None
            
        checkpoint = torch.load(best_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint