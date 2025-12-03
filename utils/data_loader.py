"""
Data loading utilities for k-fold cross-validation.
"""
import os
import torch
import numpy as np
from glob import glob
from natsort import natsorted
from sklearn.utils import shuffle
from PIL import Image
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for image classification with optional augmentation.
    
    Args:
        paths: List of image file paths
        labs: List/array of labels
        transform: Optional torchvision transforms
        augment: Whether to apply data augmentation
        return_two_views: If True, return two augmented views (for contrastive learning)
    """
    def __init__(self, paths, labs, transform=None, augment=False, return_two_views=False):
        self.paths = paths
        self.labs = labs
        self.transform = transform
        self.augment = augment
        self.return_two_views = return_two_views
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
        
        # Define augmentation transforms
        self.aug_transforms1 = transforms.Compose([
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
        
        self.aug_transforms2 = transforms.Compose([
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
   
    def preprocess(self, img):
        """Apply normalization to image tensor."""
        img = self.normalize(img)
        return img
  
    def __getitem__(self, index):
        # Select sample
        id_path = self.paths[index]
        id_labs = self.labs[index]
        
        # Open image
        img = Image.open(id_path).convert('RGB')
        
        # Create label tensor
        y = torch.tensor(id_labs, dtype=torch.float32).unsqueeze(0)
        
        if self.augment and self.return_two_views:
            # For contrastive learning: return two augmented views
            img1 = self.aug_transforms1(img)
            img2 = self.aug_transforms2(img)
            
            img1 = self.preprocess(transforms.ToTensor()(img1))
            img2 = self.preprocess(transforms.ToTensor()(img2))
            
            return img1, img2, y
            
        elif self.augment:
            # Single view with augmentation
            img = self.aug_transforms1(img)
            x = self.preprocess(transforms.ToTensor()(img))
            return x, y
            
        else:
            # No augmentation
            x = self.preprocess(transforms.ToTensor()(img))
            return x, y


def _get_fold_base(parent_dir):
    """
    Determine the correct fold base directory.
    DRB datasets use '01vsAll' subdirectory, AIROGS datasets use 'AIROGS' subdirectory.
    
    Args:
        parent_dir: Parent directory to search
        
    Returns:
        Path to the fold base directory
    """
    # Check for 01vsAll (DRB datasets)
    fold_base_01vsAll = f"{parent_dir}/DR"
    if os.path.exists(fold_base_01vsAll):
        return fold_base_01vsAll
    
    # Check for AIROGS directory
    fold_base_airogs = f"{parent_dir}/AIROGS"
    if os.path.exists(fold_base_airogs):
        return fold_base_airogs
    
    raise ValueError(f"Could not find fold data in {parent_dir}. Expected '01vsAll' or 'AIROGS' subdirectory.")


def load_fold_data(parent_dir, fold_num, seed=42):
    """
    Load training and validation data for a specific fold.
    
    Training uses all folds except fold_num, validation uses fold_num.
    
    Args:
        parent_dir: Parent directory containing 01vsAll or AIROGS subdirectory with folds
        fold_num: Fold number to use for validation (1-5)
        seed: Random seed for shuffling
        
    Returns:
        Tuple of (train_data, train_labels, val_data, val_labels)
    """
    fold_base = _get_fold_base(parent_dir)
    
    # Load all 5 folds
    all_folds_data = []
    all_folds_labels = []
    
    for i in range(1, 6):
        neg_class = natsorted(glob(f"{fold_base}/Fold{i}/0/*.jpg"))
        pos_class = natsorted(glob(f"{fold_base}/Fold{i}/1/*.jpg"))
        
        fold_data = neg_class + pos_class
        fold_labels = np.concatenate((np.zeros(len(neg_class)), np.ones(len(pos_class))))
        
        # Shuffle each fold
        fold_data, fold_labels = shuffle(fold_data, fold_labels, random_state=seed)
        
        all_folds_data.append(fold_data)
        all_folds_labels.append(fold_labels)
    
    # Validation fold (0-indexed, so fold_num-1)
    val_data = all_folds_data[fold_num - 1]
    val_labels = all_folds_labels[fold_num - 1]
    
    # Training folds (all except validation fold)
    train_data = []
    train_labels = []
    for i in range(5):
        if i != (fold_num - 1):
            train_data.extend(all_folds_data[i])
            train_labels.extend(all_folds_labels[i])
    
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    
    return train_data, train_labels, val_data, val_labels


def load_test_data(parent_dir, seed=42):
    """
    Load test data.
    
    Args:
        parent_dir: Parent directory containing 01vsAll or AIROGS subdirectory with Test folder
        seed: Random seed for shuffling
        
    Returns:
        Tuple of (test_data, test_labels)
    """
    fold_base = _get_fold_base(parent_dir)
    test_base = f"{fold_base}/Test"
    
    neg_class = natsorted(glob(f"{test_base}/0/*.jpg"))
    pos_class = natsorted(glob(f"{test_base}/1/*.jpg"))
    
    test_data = neg_class + pos_class
    test_labels = np.concatenate((np.zeros(len(neg_class)), np.ones(len(pos_class))))
    
    # Shuffle test data
    test_data, test_labels = shuffle(test_data, test_labels, random_state=seed)
    
    return np.array(test_data), test_labels


def load_validation_data(parent_dir, fold_num, seed=42):
    """
    Load validation data for a specific fold.
    
    Args:
        parent_dir: Parent directory containing 01vsAll or AIROGS subdirectory with folds
        fold_num: Fold number to use for validation (1-5)
        seed: Random seed for shuffling
        
    Returns:
        Tuple of (val_data, val_labels)
    """
    fold_base = _get_fold_base(parent_dir)
    
    neg_class = natsorted(glob(f"{fold_base}/Fold{fold_num}/0/*.jpg"))
    pos_class = natsorted(glob(f"{fold_base}/Fold{fold_num}/1/*.jpg"))
    
    val_data = neg_class + pos_class
    val_labels = np.concatenate((np.zeros(len(neg_class)), np.ones(len(pos_class))))
    
    # Shuffle data
    val_data, val_labels = shuffle(val_data, val_labels, random_state=seed)
    
    return np.array(val_data), val_labels


def create_validation_dataloader(val_data, val_labels, batch_size, num_workers=8):
    """
    Create PyTorch DataLoader for validation set.
    
    Args:
        val_data: Validation data paths
        val_labels: Validation labels
        batch_size: Batch size for dataloader
        num_workers: Number of workers for data loading
        
    Returns:
        Validation DataLoader
    """
    val_dataset = Dataset(val_data, val_labels, augment=False, return_two_views=False)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    return val_loader


def create_dataloaders(train_data, train_labels, val_data, val_labels, 
                       batch_size, num_workers=8, augment_train=True,
                       return_two_views=False):
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        train_data: Training data paths
        train_labels: Training labels
        val_data: Validation data paths
        val_labels: Validation labels
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        augment_train: Whether to augment training data
        return_two_views: Whether to return two views for contrastive learning
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = Dataset(train_data, train_labels, augment=augment_train, 
                           return_two_views=return_two_views)
    val_dataset = Dataset(val_data, val_labels, augment=False, return_two_views=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        drop_last=False
    )
    
    return train_loader, val_loader


def create_test_dataloader(test_data, test_labels, batch_size, num_workers=8):
    """
    Create PyTorch Dataloader for test set.
    
    Args:
        test_data: Test data paths
        test_labels: Test labels
        batch_size: Batch size for dataloader
        num_workers: Number of workers for data loading
        
    Returns:
        Test DataLoader
    """
    test_dataset = Dataset(test_data, test_labels, augment=False, return_two_views=False)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    return test_loader
