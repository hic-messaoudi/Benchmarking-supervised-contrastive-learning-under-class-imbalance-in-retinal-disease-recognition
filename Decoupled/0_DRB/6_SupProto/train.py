"""
Decoupled SupProto (Supervised Prototypical Contrastive) Training Script
Contrastive learning with class prototypes.
"""

import os
import sys
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from glob import glob
from natsort import natsorted
from sklearn.utils import shuffle
from torchvision import transforms

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuration
SEED = 42
BATCH_SIZE = 128
DEVICE = 'cuda:0'
NUM_WORKERS = 8
EPOCHS = 1
LEARNING_RATE = 3e-4
PATIENCE = 20
FEATURE_DIM = 256

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


class ContrastiveDataset(torch.utils.data.Dataset):
    """Dataset that returns two augmented views of each image."""
    
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
            ], p=0.7),
            transforms.RandomApply([
                transforms.GaussianBlur((3, 3), sigma=(0.5, 2.0))
            ], p=0.3),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        
        img1 = self.normalize(transforms.ToTensor()(self.augmentation(img)))
        img2 = self.normalize(transforms.ToTensor()(self.augmentation(img)))
        return img1, img2, label


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
    
    train_dataset = ContrastiveDataset(np.array(train_paths), np.array(train_labels), augment=True)
    val_paths, val_labels = all_data[fold_idx]
    val_dataset = ContrastiveDataset(np.array(val_paths), np.array(val_labels), augment=False)
    
    return train_dataset, val_dataset


class SupProtoLoss(nn.Module):
    """Supervised Prototypical Contrastive Loss."""
    
    def __init__(self, num_classes=2, feature_dim=FEATURE_DIM, temperature=0.1, device=DEVICE):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.device = device
        
        # Initialize class prototypes on the correct device
        self.register_buffer('prototypes', torch.zeros(num_classes, feature_dim, device=device))
        self.register_buffer('prototype_counts', torch.zeros(num_classes, device=device))

    def update_prototypes(self, features, labels):
        """Update class prototypes with momentum."""
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = (labels.squeeze() == c)
                if mask.sum() > 0:
                    class_features = features[mask]
                    mean_features = F.normalize(class_features.mean(dim=0), dim=0)
                    
                    if self.prototype_counts[c] == 0:
                        self.prototypes[c] = mean_features
                    else:
                        momentum = 0.9
                        self.prototypes[c] = F.normalize(
                            momentum * self.prototypes[c] + (1 - momentum) * mean_features, dim=0
                        )
                    self.prototype_counts[c] += mask.sum()

    def forward(self, features, labels):
        """Compute prototypical contrastive loss."""
        features = F.normalize(features, dim=1)
        self.update_prototypes(features, labels)
        
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        
        # Sample-to-sample contrastive (supervised)
        mask = torch.eq(labels, labels.T).float().to(self.device)
        
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(self.device), 0
        )
        mask = mask * logits_mask
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        sample_loss = -mean_log_prob_pos.mean()
        
        # Sample-to-prototype contrastive
        proto_sim = torch.matmul(features, self.prototypes.T) / self.temperature
        proto_labels = labels.squeeze().long()
        proto_loss = F.cross_entropy(proto_sim, proto_labels)
        
        # Combined loss
        loss = sample_loss + proto_loss
        return loss


class ContrastiveEncoder(nn.Module):
    """Encoder with backbone and projection head only (no classifier)."""
    
    def __init__(self, backbone_name="tf_efficientnet_b0.ns_jft_in1k", feature_dim=FEATURE_DIM):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.backbone.classifier = nn.Identity()
        self.projector = nn.Linear(1280, feature_dim)

    def forward(self, x):
        features = self.backbone(x)
        proj = F.normalize(self.projector(features), dim=-1)
        return proj


class Trainer:
    """Training class for Decoupled SupProto."""
    
    def __init__(self, device=DEVICE):
        self.device = device
        self.patience = PATIENCE
        self.contrastive_criterion = SupProtoLoss(device=device)
        self.model = ContrastiveEncoder().to(device)

    def train_epoch(self, train_loader, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for img1, img2, labels in train_loader:
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)
            
            images = torch.cat([img1, img2], dim=0)
            labels_cat = torch.cat([labels, labels], dim=0)
            proj = self.model(images)
            
            loss = self.contrastive_criterion(proj, labels_cat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return {'supproto': total_loss / len(train_loader)}

    @torch.no_grad()
    def validate(self, val_loader):
        """Validate using SupProto loss."""
        self.model.eval()
        total_loss = 0
        
        for img1, img2, labels in val_loader:
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)
            
            images = torch.cat([img1, img2], dim=0)
            labels_cat = torch.cat([labels, labels], dim=0)
            proj = self.model(images)
            
            loss = self.contrastive_criterion(proj, labels_cat)
            total_loss += loss.item()
        
        return {'supproto': total_loss / len(val_loader)}

    def fit(self, train_loader, val_loader, fold_num, epochs=EPOCHS, lr=LEARNING_RATE):
        """Train the model."""
        seed_everything(SEED)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        best_loss = float('inf')
        no_improve = 0
        
        for epoch in range(epochs):
            start = time.time()
            
            train_metrics = self.train_epoch(train_loader, optimizer)
            duration = time.time() - start
            
            print(f"Epoch {epoch+1}/{epochs} ({duration:.0f}s) - "
                  f"Train Loss: {train_metrics['supproto']:.4f}")
            
            val_metrics = self.validate(val_loader)
            print(f"  Val Loss: {val_metrics['supproto']:.4f}")
            
            if val_metrics['supproto'] < best_loss:
                best_loss = val_metrics['supproto']
                no_improve = 0
                print(f"  New best loss: {best_loss:.4f}")
                torch.save(self.model, os.path.join(CURRENT_DIR, f'SupProto_fold_{fold_num}.pt'))
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break


def main():
    """Main training function."""
    seed_everything(SEED)
    
    all_data = [load_fold_data(i) for i in range(1, 6)]
    
    # Stage 1: Contrastive Learning
    print("\n" + "="*60)
    print("STAGE 1: Contrastive Learning (SupProto)")
    print("="*60)
    
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
            num_workers=4, drop_last=True
        )
        
        trainer = Trainer()
        trainer.fit(train_loader, val_loader, fold+1)
    
    print("\nContrastive training completed!")
    
    # Stage 2: Finetuning (Classification)
    print("\n" + "="*60)
    print("STAGE 2: Finetuning (Classification)")
    print("="*60)
    
    for fold in range(5):
        print(f"\n{'='*50}")
        print(f"Finetuning Fold {fold+1}")
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
        
        finetuner = Finetuner(fold+1)
        finetuner.fit(train_loader, val_loader, fold+1)
    
    print("\nFinetuning completed!")


# =============================================================================
# Stage 2: Finetuning Classes
# =============================================================================

class Classifier(nn.Module):
    """Classifier using frozen pretrained backbone."""
    
    def __init__(self, backbone_path):
        super().__init__()
        encoder = torch.load(backbone_path, map_location=DEVICE, weights_only=False)
        self.backbone = encoder.backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(1280, 1)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.classifier(features)


class Finetuner:
    """Finetuning class for classification stage."""
    
    def __init__(self, fold_num, device=DEVICE):
        self.device = device
        self.patience = PATIENCE
        self.criterion = nn.BCEWithLogitsLoss()
        backbone_path = os.path.join(CURRENT_DIR, f'SupProto_fold_{fold_num}.pt')
        self.model = Classifier(backbone_path).to(device)

    def train_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss = 0
        
        for img1, img2, labels in train_loader:
            img1 = img1.to(self.device)
            labels = labels.to(self.device).squeeze()
            
            optimizer.zero_grad()
            outputs = self.model(img1).squeeze()
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        all_preds, all_labels = [], []
        
        for img1, img2, labels in val_loader:
            img1 = img1.to(self.device)
            labels = labels.to(self.device)
            
            outputs = torch.sigmoid(self.model(img1)).squeeze()
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())
        
        from torcheval.metrics.functional import binary_auroc, binary_auprc
        all_preds = torch.cat(all_preds).flatten()
        all_labels = torch.cat(all_labels).flatten().int()
        
        return {'auroc': binary_auroc(all_preds, all_labels).item(), 
                'auprc': binary_auprc(all_preds, all_labels).item()}

    def fit(self, train_loader, val_loader, fold_num, epochs=EPOCHS, lr=LEARNING_RATE):
        seed_everything(SEED)
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=lr)
        
        best_auroc = 0
        no_improve = 0
        
        for epoch in range(epochs):
            start = time.time()
            train_loss = self.train_epoch(train_loader, optimizer)
            duration = time.time() - start
            
            print(f"Epoch {epoch+1}/{epochs} ({duration:.0f}s) - Train Loss: {train_loss:.4f}")
            
            val_metrics = self.validate(val_loader)
            print(f"  Val AUROC: {val_metrics['auroc']:.4f}, AUPRC: {val_metrics['auprc']:.4f}")
            
            if val_metrics['auroc'] > best_auroc:
                best_auroc = val_metrics['auroc']
                no_improve = 0
                print(f"  New best AUROC: {best_auroc:.4f}")
                torch.save(self.model, os.path.join(CURRENT_DIR, f'SupProto_classifier_fold_{fold_num}.pt'))
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break


if __name__ == '__main__':
    main()
