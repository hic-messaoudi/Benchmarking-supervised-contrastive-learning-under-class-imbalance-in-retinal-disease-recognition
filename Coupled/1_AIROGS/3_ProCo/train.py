"""
ProCo (Probabilistic Contrastive) Training Script
Trains a model using ProCo loss with von Mises-Fisher distribution for class prototypes.
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
from torcheval.metrics.functional import binary_auroc, binary_auprc
from scipy.special import ive

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
NUM_CLASSES = 2

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
    
    train_dataset = ContrastiveDataset(np.array(train_paths), np.array(train_labels), augment=True)
    val_paths, val_labels = all_data[fold_idx]
    val_dataset = ContrastiveDataset(np.array(val_paths), np.array(val_labels), augment=False)
    
    return train_dataset, val_dataset


def miller_recurrence(nu, x, device=DEVICE):
    """Miller's recurrence for computing Bessel functions."""
    I_n = torch.ones(1, dtype=torch.float64).to(device)
    I_n1 = torch.zeros(1, dtype=torch.float64).to(device)

    Estimat_n = [nu, nu+1]
    scale0 = 0 
    scale1 = 0 
    scale = 0

    for i in range(2*nu, 0, -1):
        I_n_tem, I_n1_tem = 2*i/x*I_n + I_n1, I_n
        if torch.isinf(I_n_tem).any():
            I_n1 /= I_n
            scale += torch.log(I_n)
            if i >= (nu+1):
                scale0 += torch.log(I_n)
                scale1 += torch.log(I_n)
            elif i == nu:
                scale0 += torch.log(I_n)

            I_n = torch.ones(1, dtype=torch.float64).to(device)
            I_n, I_n1 = 2*i/x*I_n + I_n1, I_n
        else:
            I_n, I_n1 = I_n_tem, I_n1_tem

        if i == nu:
            Estimat_n[0] = I_n1
        elif i == (nu+1):
            Estimat_n[1] = I_n1

    ive0 = torch.special.i0e(x.to(device))

    Estimat_n[0] = torch.log(ive0) + torch.log(Estimat_n[0]) - torch.log(I_n) + scale0 - scale
    Estimat_n[1] = torch.log(ive0) + torch.log(Estimat_n[1]) - torch.log(I_n) + scale1 - scale

    return Estimat_n[0], Estimat_n[1]


class LogRatioC(torch.autograd.Function):
    """Autograd function for log ratio of normalizing constants."""
    
    @staticmethod
    def forward(ctx, k, p, logc):
        nu, nu1 = miller_recurrence((p/2 - 1).int(), k.double())
        tensor = nu + k - (p/2 - 1) * torch.log(k+1e-20) - logc
        ctx.save_for_backward(torch.exp(nu1 - nu))
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad = ctx.saved_tensors[0]
        grad[grad > 1.0] = 1.0
        grad *= grad_output
        return grad, None, None


class EstimatorCV:
    """Class for estimating class-conditional von Mises-Fisher distributions."""
    
    def __init__(self, feature_num, class_num, device=DEVICE):
        self.class_num = class_num
        self.feature_num = feature_num
        self.device = device
        self.reset()

    def reset(self):
        """Reset estimator state."""
        self.Ave = F.normalize(torch.randn(self.class_num, self.feature_num), dim=1) * 0.9
        self.Amount = torch.zeros(self.class_num)
        self.kappa = torch.ones(self.class_num) * self.feature_num * 90 / 19
        
        tem = torch.from_numpy(
            ive(self.feature_num/2 - 1, self.kappa.cpu().numpy().astype(np.float64))
        ).to(self.kappa.device)
        self.logc = torch.log(tem+1e-300) + self.kappa - (self.feature_num/2 - 1) * torch.log(self.kappa+1e-300)
        
        self.Ave = self.Ave.to(self.device)
        self.Amount = self.Amount.to(self.device)
        self.kappa = self.kappa.to(self.device)
        self.logc = self.logc.to(self.device)

    def update_CV(self, features, labels):
        """Update class-conditional averages."""
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C).to(self.device)
        onehot.scatter_(1, labels.long().view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)
        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1
        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)
        weight_AV = sum_weight_AV.div(sum_weight_AV + self.Amount.view(C, 1).expand(C, A))
        weight_AV[weight_AV != weight_AV] = 0

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()
        self.Amount += onehot.sum(0)

    def update_kappa(self):
        """Update concentration parameters."""
        R = torch.linalg.norm(self.Ave, dim=1)
        self.kappa = self.feature_num * R / (1 - R**2)
        self.kappa[self.kappa > 1e5] = 1e5
        self.kappa[self.kappa < 0] = 1e5

        nu, _ = miller_recurrence(torch.tensor(self.feature_num/2 - 1).int(), self.kappa.double())
        self.logc = nu + self.kappa - (self.feature_num/2 - 1) * torch.log(self.kappa+1e-20)


class ProCoLoss(nn.Module):
    """ProCo loss for probabilistic contrastive learning."""
    
    def __init__(self, feature_dim=FEATURE_DIM, temperature=0.1, num_classes=NUM_CLASSES, device=DEVICE):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.feature_num = feature_dim
        self.device = device
        self.estimator_old = EstimatorCV(feature_dim, num_classes, device)
        self.estimator = EstimatorCV(feature_dim, num_classes, device)

    def hook_before_epoch(self):
        """Exchange estimators before each epoch."""
        self.estimator_old.Ave = self.estimator.Ave
        self.estimator_old.Amount = self.estimator.Amount
        self.estimator_old.kappa = self.estimator.kappa
        self.estimator_old.logc = self.estimator.logc
        self.estimator.reset()

    def forward(self, features, labels):
        """Compute ProCo loss."""
        self.estimator_old.update_CV(features.detach(), labels)
        self.estimator.update_CV(features.detach(), labels)
        self.estimator_old.update_kappa()

        Ave = self.estimator_old.Ave.detach()
        Ave_norm = F.normalize(Ave, dim=1)
        logc = self.estimator_old.logc.detach()
        kappa = self.estimator_old.kappa.detach()

        tem = kappa.reshape(-1, 1) * Ave_norm
        tem = tem.unsqueeze(0) + features.unsqueeze(1) / self.temperature
        kappa_new = torch.linalg.norm(tem, dim=2)

        contrast_logits = LogRatioC.apply(kappa_new, torch.tensor(self.feature_num), logc)
        return nn.CrossEntropyLoss()(contrast_logits, labels[:, 0].long())


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
    """Training class for ProCo."""
    
    def __init__(self, device=DEVICE):
        self.device = device
        self.patience = PATIENCE
        self.contrastive_criterion = ProCoLoss(device=device).to(device)
        self.classification_criterion = nn.BCELoss()
        self.model = ContrastiveModel().to(device)

    def train_epoch(self, train_loader, optimizer):
        """Train for one epoch."""
        self.model.train()
        self.contrastive_criterion.hook_before_epoch()
        
        total_con_loss = 0
        total_cls_loss = 0
        
        for img1, img2, labels in train_loader:
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)
            
            images = torch.cat([img1, img2], dim=0)
            proj, logits = self.model(images)
            
            m = proj.shape[0] // 2
            
            # ProCo loss only on first view
            con_loss = self.contrastive_criterion(proj[:m], labels)
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
                torch.save(self.model, os.path.join(CURRENT_DIR, f'ProCo_fold_{fold_num}.pt'))
            
            if val_metrics['auprc'] > best_auprc:
                best_auprc = val_metrics['auprc']
                improved = True
                print(f"  New best AUPRC: {best_auprc:.4f}")
                torch.save(self.model, os.path.join(CURRENT_DIR, f'ProCo_foldp_{fold_num}.pt'))
            
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
        backbone_path = os.path.join(CURRENT_DIR, f'ProCo_fold_{fold_num}.pt')
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
                torch.save(self.model, os.path.join(CURRENT_DIR, f'ProCofS_fold_{fold_num}.pt'))
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
