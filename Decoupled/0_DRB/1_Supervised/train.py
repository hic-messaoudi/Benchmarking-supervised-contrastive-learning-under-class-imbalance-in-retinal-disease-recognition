"""
Supervised Training Script for Binary Classification
Trains an EfficientNet model using standard BCE loss with 5-fold cross-validation.
"""

import os
import sys
import random
import time
import warnings

import numpy as np
import torch
import timm
from PIL import Image
from glob import glob
from natsort import natsorted
from sklearn.utils import shuffle
from torchvision import transforms
from torcheval.metrics.functional import binary_auroc, binary_auprc

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuration
SEED = 42
BATCH_SIZE = 256
DEVICE = 'cuda:0'
NUM_WORKERS = 8
EPOCHS = 1
LEARNING_RATE = 3e-4
PATIENCE = 20

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PARENT_DIR, 'DR')


def seed_everything(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImageDataset(torch.utils.data.Dataset):
    """Dataset for loading and augmenting images."""
    
    def __init__(self, paths, labels, augment=False):
        self.paths = paths
        self.labels = labels
        self.augment = augment
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.RandomResizedCrop(size=(256, 256), scale=(0.7, 1.0), ratio=(0.75, 1.33))
            ], p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur((3, 3), sigma=(0.5, 2.0))
            ], p=0.2),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.5),
            transforms.RandomGrayscale(p=0.2),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        label = self.labels[idx].astype(np.float32)
        
        if self.augment:
            img = self.augmentation(img)
        
        img = self.normalize(transforms.ToTensor()(np.array(img)))
        return img.float(), torch.tensor(label)


def load_fold_data(fold_num: int):
    """Load data for a specific fold."""
    neg_paths = natsorted(glob(os.path.join(DATA_DIR, f'Fold{fold_num}', '0', '*.jpg')))
    pos_paths = natsorted(glob(os.path.join(DATA_DIR, f'Fold{fold_num}', '1', '*.jpg')))
    
    paths = neg_paths + pos_paths
    labels = np.concatenate([np.zeros(len(neg_paths)), np.ones(len(pos_paths))])
    
    return shuffle(paths, labels, random_state=SEED)


def create_fold_datasets(fold_idx: int, all_data: list):
    """Create train and validation datasets for a fold."""
    train_paths, train_labels = [], []
    
    for i, (paths, labels) in enumerate(all_data):
        if i != fold_idx:
            train_paths.extend(paths)
            train_labels.extend(labels)
    
    train_paths = np.array(train_paths)
    train_labels = np.array(train_labels)
    val_paths, val_labels = all_data[fold_idx]
    
    train_dataset = ImageDataset(train_paths, train_labels, augment=True)
    val_dataset = ImageDataset(np.array(val_paths), np.array(val_labels), augment=False)
    
    return train_dataset, val_dataset


class Trainer:
    """Training class for supervised learning."""
    
    def __init__(self, device=DEVICE):
        self.device = device
        self.patience = PATIENCE
        self.criterion = torch.nn.BCELoss()
        self.model = timm.create_model(
            'tf_efficientnet_b0.ns_jft_in1k', 
            pretrained=True, 
            num_classes=1
        ).to(self.device)

    def train_epoch(self, train_loader, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = torch.sigmoid(self.model(images)).squeeze()
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            all_preds.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
        
        preds = torch.cat(all_preds).flatten().to(self.device)
        labels = torch.cat(all_labels).flatten().to(self.device).int()
        
        return {
            'loss': total_loss / len(train_loader),
            'auroc': binary_auroc(preds, labels).item(),
            'auprc': binary_auprc(preds, labels).item()
        }

    @torch.no_grad()
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        all_preds, all_labels = [], []
        
        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = torch.sigmoid(self.model(images)).squeeze()
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())
        
        preds = torch.cat(all_preds).flatten().to(self.device)
        labels = torch.cat(all_labels).flatten().to(self.device).int()
        
        return {
            'auroc': binary_auroc(preds, labels).item(),
            'auprc': binary_auprc(preds, labels).item()
        }

    def fit(self, train_loader, val_loader, fold_num, epochs=EPOCHS, lr=LEARNING_RATE):
        """Train the model."""
        seed_everything(SEED)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        best_auroc = 0
        best_auprc = 0
        no_improve = 0
        
        for epoch in range(epochs):
            start = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader, optimizer)
            duration = time.time() - start
            
            print(f"Epoch {epoch+1}/{epochs} ({duration:.0f}s) - "
                  f"Loss: {train_metrics['loss']:.4f}, "
                  f"AUROC: {train_metrics['auroc']:.4f}, "
                  f"AUPRC: {train_metrics['auprc']:.4f}")
            
            # Skip validation if training metrics are too low
            if train_metrics['auroc'] < 0.3:
                continue
            
            # Validation
            val_metrics = self.validate(val_loader)
            print(f"  Val AUROC: {val_metrics['auroc']:.4f}, Val AUPRC: {val_metrics['auprc']:.4f}")
            
            improved = False
            
            # Save best AUROC model
            if val_metrics['auroc'] > best_auroc:
                best_auroc = val_metrics['auroc']
                improved = True
                print(f"  New best AUROC: {best_auroc:.4f}")
                torch.save(self.model, os.path.join(CURRENT_DIR, f'Supervised_fold_{fold_num}.pt'))
            
            # Save best AUPRC model
            if val_metrics['auprc'] > best_auprc:
                best_auprc = val_metrics['auprc']
                improved = True
                print(f"  New best AUPRC: {best_auprc:.4f}")
                torch.save(self.model, os.path.join(CURRENT_DIR, f'Supervised_foldp_{fold_num}.pt'))
            
            # Early stopping
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break


def main():
    """Main training function."""
    seed_everything(SEED)
    
    # Load all fold data
    all_data = [load_fold_data(i) for i in range(1, 6)]
    
    # Train for each fold
    for fold in range(5):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold+1}")
        print('='*50)
        
        train_dataset, val_dataset = create_fold_datasets(fold, all_data)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, drop_last=False
        )
        
        trainer = Trainer()
        trainer.fit(train_loader, val_loader, fold+1)
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
