"""
Common utilities shared across all training and evaluation scripts.
"""
import torch
import numpy as np
import random
import os


def seed_everything(seed: int):
    """
    Set seed for reproducibility across all random number generators.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_metric(value, decimals=4):
    """
    Format a metric value to specified number of decimal places.
    
    Args:
        value: Numeric value to format
        decimals: Number of decimal places (default: 4)
        
    Returns:
        Formatted string with specified decimal places
    """
    if isinstance(value, (int, float, np.number)):
        return f"{float(value):.{decimals}f}"
    elif isinstance(value, torch.Tensor):
        return f"{float(value.item()):.{decimals}f}"
    else:
        return str(value)


def print_section_header(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}\n")
