"""
SupProto (Supervised Contrastive with Prototypes) Training Script
Extends supervised contrastive learning with class prototypes.
"""

import os
import sys
import random
import time
import warnings
from typing import Optional, Tuple

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
from torcheval.metrics.functional import binary_auroc, binary_auprc

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
DATA_DIR = os.path.join(PARENT_DIR, 'AIROGS')


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
        
        if self.augment:
            img1 = self.normalize(transforms.ToTensor()(self.augmentation(img)))
            img2 = self.normalize(transforms.ToTensor()(self.augmentation(img)))
            return img1, img2, label
        else:
            img = self.normalize(transforms.ToTensor()(img))
            return img, label


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
    
    train_labels_arr = np.array(train_labels)
    neg_count = np.sum(train_labels_arr == 0)
    pos_count = np.sum(train_labels_arr == 1)
    neg_weight = max(pos_count, 1) / neg_count
    
    train_dataset = ContrastiveDataset(np.array(train_paths), train_labels_arr, augment=True)
    val_paths, val_labels = all_data[fold_idx]
    val_dataset = ContrastiveDataset(np.array(val_paths), np.array(val_labels), augment=False)
    
    return train_dataset, val_dataset, neg_weight


class SupProtoLoss(nn.Module):
    """Supervised Contrastive Learning with Prototypes."""
    
    def __init__(self, temperature=0.1, negatives_weight=1.0, eps=0.1, 
                 minority_cls=0, device=DEVICE):
        super().__init__()
        self.temperature = temperature
        self.negatives_weight = negatives_weight
        self.eps = eps
        self.minority_cls = minority_cls
        self.device = device
        self.prototypes = None

    def set_prototypes(self, prototypes: torch.Tensor):
        """Set class prototypes."""
        self.prototypes = prototypes.to(self.device)

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """Compute SupProto loss."""
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        # Convert labels to one-hot format
        labels_min = (labels == 1).to(torch.float32)
        labels_maj = (labels == 0).to(torch.float32)
        labels_onehot = torch.cat((labels_min, labels_maj), dim=1)
        
        batch_size = features.shape[0]
        num_prototypes = self.prototypes.size(0) if self.prototypes is not None else 0

        # Create supervision mask
        labels_dim = labels_onehot[:, 1].contiguous().view(-1, 1)
        mask_sup = torch.eq(labels_dim, labels_dim.T).float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.prototypes is not None:
            contrast_feature = torch.cat([contrast_feature, self.prototypes], dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # Compute similarity matrix
        sims = torch.matmul(anchor_feature, contrast_feature.T)
        anchor_dot_contrast = torch.div(sims, self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Create masks
        mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        mask = mask.repeat(anchor_count, contrast_count)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device), 0
        )
        
        mask_sup = mask_sup.repeat(anchor_count, contrast_count)
        mask_sup = mask_sup * logits_mask
        mask = mask * logits_mask

        # Handle prototypes
        if self.prototypes is not None:
            selected_prototypes_mask = labels_onehot.to(self.device)
            selected_prototypes_mask = torch.cat(
                [selected_prototypes_mask, selected_prototypes_mask], dim=0
            )
            
            logits_mask = torch.cat([
                logits_mask, 
                torch.zeros_like(selected_prototypes_mask).to(self.device)
            ], dim=1)
            
            logits = logits[:-num_prototypes, :]

        # Compute loss
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(self.negatives_weight * exp_logits.sum(1, keepdim=True))

        # Apply prototype pull mechanism
        if self.prototypes is not None:
            m_pull = torch.ones_like(selected_prototypes_mask, dtype=torch.float32).to(self.device)
            
            cond2 = sims[:-num_prototypes, -1] <= (sims[:-num_prototypes, -2] + self.eps)
            cond1 = sims[:-num_prototypes, -2] <= (sims[:-num_prototypes, -1] + self.eps)
            
            p2_mask = torch.logical_and(cond2, selected_prototypes_mask[:, -1].bool())
            p1_mask = torch.logical_and(cond1, selected_prototypes_mask[:, -2].bool())
            selected_prototypes_mask = torch.stack((p1_mask, p2_mask), dim=1).float()
            
            m_pull = m_pull * selected_prototypes_mask
            mask = torch.cat([mask, m_pull], dim=1)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = -mean_log_prob_pos.mean()
        return loss


class ContrastiveModel(nn.Module):
    """Model with backbone, projection head, and classifier."""
    
    def __init__(self, backbone_name="tf_efficientnet_b0.ns_jft_in1k", feature_dim=FEATURE_DIM):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.backbone.classifier = nn.Identity()
        self.projector = nn.Linear(1280, feature_dim)
        self.classifier = nn.Linear(1280, 1)

    def forward(self, x):
        features = self.backbone(x)
        proj = F.normalize(self.projector(features), dim=-1)
        logits = self.classifier(features)
        
        if self.training:
            return proj, logits
        return logits


class Trainer:
    """Training class for SupProto."""
    
    def __init__(self, neg_weight, device=DEVICE):
        self.device = device
        self.patience = PATIENCE
        self.contrastive_criterion = SupProtoLoss(negatives_weight=neg_weight, device=device)
        self.classification_criterion = nn.BCELoss()
        self.model = ContrastiveModel().to(device)
        self._init_prototypes()

    def _init_prototypes(self):
        """Initialize class prototypes."""
        prototype_1 = torch.randn(FEATURE_DIM).to(self.device)
        prototype_1 = F.normalize(prototype_1, dim=0)
        prototype_2 = -prototype_1  # Opposite direction
        self.contrastive_criterion.set_prototypes(
            torch.stack([prototype_1, prototype_2], dim=0)
        )

    def train_epoch(self, train_loader, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_con_loss = 0
        total_cls_loss = 0
        
        for img1, img2, labels in train_loader:
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)
            
            images = torch.cat([img1, img2], dim=0)
            proj, logits = self.model(images)
            
            m = proj.shape[0] // 2
            contrastive_features = torch.cat([
                proj[:m].unsqueeze(1), 
                proj[m:].unsqueeze(1)
            ], dim=1)
            
            con_loss = self.contrastive_criterion(contrastive_features, labels)
            cls_loss = (
                self.classification_criterion(torch.sigmoid(logits[:m]), labels) +
                self.classification_criterion(torch.sigmoid(logits[m:]), labels)
            )
            loss = con_loss + cls_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_con_loss += con_loss.item()
            total_cls_loss += cls_loss.item()
        
        n = len(train_loader)
        return {'contrastive': total_con_loss / n, 'classification': total_cls_loss / n}

    @torch.no_grad()
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        all_preds, all_labels = [], []
        
        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = torch.sigmoid(self.model(images))
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())
        
        preds = torch.cat(all_preds).flatten().to(self.device)
        labels = torch.cat(all_labels).flatten().to(self.device)
        
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
            
            train_metrics = self.train_epoch(train_loader, optimizer)
            duration = time.time() - start
            
            print(f"Epoch {epoch+1}/{epochs} ({duration:.0f}s) - "
                  f"Con Loss: {train_metrics['contrastive']:.4f}, "
                  f"Cls Loss: {train_metrics['classification']:.4f}")
            
            val_metrics = self.validate(val_loader)
            print(f"  Val AUROC: {val_metrics['auroc']:.4f}, Val AUPRC: {val_metrics['auprc']:.4f}")
            
            improved = False
            
            if val_metrics['auroc'] > best_auroc:
                best_auroc = val_metrics['auroc']
                improved = True
                print(f"  New best AUROC: {best_auroc:.4f}")
                torch.save(self.model, os.path.join(CURRENT_DIR, f'SupProto_fold_{fold_num}.pt'))
            
            if val_metrics['auprc'] > best_auprc:
                best_auprc = val_metrics['auprc']
                improved = True
                print(f"  New best AUPRC: {best_auprc:.4f}")
                torch.save(self.model, os.path.join(CURRENT_DIR, f'SupProto_foldp_{fold_num}.pt'))
            
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break


class FinetuneDataset(torch.utils.data.Dataset):
    """Simple dataset for finetuning (no augmentation pairs)."""
    
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        img = self.normalize(transforms.ToTensor()(img))
        return img, label


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
        
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images).squeeze()
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        all_preds, all_labels = [], []
        
        for images, labels in val_loader:
            images = images.to(self.device)
            outputs = torch.sigmoid(self.model(images)).squeeze()
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())
        
        preds = torch.cat(all_preds).flatten()
        labels = torch.cat(all_labels).flatten().int()
        
        return {
            'auroc': binary_auroc(preds, labels).item(),
            'auprc': binary_auprc(preds, labels).item()
        }

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
                torch.save(self.model, os.path.join(CURRENT_DIR, f'SupProtofS_fold_{fold_num}.pt'))
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break


def create_finetune_datasets(fold_idx: int, all_data: list):
    """Create train and validation datasets for finetuning."""
    train_paths, train_labels = [], []
    
    for i, (paths, labels) in enumerate(all_data):
        if i != fold_idx:
            train_paths.extend(paths)
            train_labels.extend(labels)
    
    train_dataset = FinetuneDataset(np.array(train_paths), np.array(train_labels))
    val_paths, val_labels = all_data[fold_idx]
    val_dataset = FinetuneDataset(np.array(val_paths), np.array(val_labels))
    
    return train_dataset, val_dataset


def main():
    """Main training function."""
    seed_everything(SEED)
    
    all_data = [load_fold_data(i) for i in range(1, 6)]
    
    # Stage 1: Contrastive + Classification Training
    print("\n" + "="*60)
    print("STAGE 1: Contrastive + Classification Training")
    print("="*60)
    
    for fold in range(5):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold+1}")
        print('='*50)
        
        train_dataset, val_dataset, neg_weight = create_fold_datasets(fold, all_data)
        print(f"Negative weight: {neg_weight:.4f}")
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, drop_last=True
        )
        
        trainer = Trainer(neg_weight=neg_weight)
        trainer.fit(train_loader, val_loader, fold+1)
    
    # Stage 2: Finetuning (frozen backbone + new classifier)
    print("\n" + "="*60)
    print("STAGE 2: Finetuning (Frozen Backbone)")
    print("="*60)
    
    for fold in range(5):
        print(f"\n{'='*50}")
        print(f"Finetuning Fold {fold+1}")
        print('='*50)
        
        train_dataset, val_dataset = create_finetune_datasets(fold, all_data)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, drop_last=True
        )
        
        finetuner = Finetuner(fold+1)
        finetuner.fit(train_loader, val_loader, fold+1)
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
