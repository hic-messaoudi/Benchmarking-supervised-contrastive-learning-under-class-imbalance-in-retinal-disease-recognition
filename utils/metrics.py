"""
Metrics calculation and statistical testing utilities.
"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score
from scipy import stats
from torcheval.metrics.functional import binary_auroc, binary_auprc
from utils.common import format_metric


def calculate_fold_metrics(predictions, labels, threshold=0.5):
    """
    Calculate comprehensive metrics for a single fold.
    
    Args:
        predictions: Model predictions (probabilities)
        labels: Ground truth labels
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        Dictionary containing all metrics
    """
    # Convert to numpy if tensors
    if isinstance(predictions, torch.Tensor):
        predictions_prob = predictions.cpu().numpy().flatten()
    else:
        predictions_prob = np.array(predictions).flatten()
        
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy().flatten()
    else:
        labels_np = np.array(labels).flatten()
    
    # Binary predictions
    predictions_binary = (predictions_prob >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(labels_np, predictions_binary)
    
    # Class-wise recall
    recall_per_class = recall_score(labels_np, predictions_binary, average=None, zero_division=0)
    recall_0 = recall_per_class[0] if len(recall_per_class) > 0 else 0.0
    recall_1 = recall_per_class[1] if len(recall_per_class) > 1 else 0.0
    
    # Macro F1
    f1_macro = f1_score(labels_np, predictions_binary, average='macro', zero_division=0)
    
    # AUROC and AUPRC
    try:
        auroc = binary_auroc(torch.from_numpy(predictions_prob), 
                            torch.from_numpy(labels_np)).item()
    except:
        auroc = 0.0
    
    try:
        auprc = binary_auprc(torch.from_numpy(predictions_prob), 
                            torch.from_numpy(labels_np)).item()
    except:
        auprc = 0.0
    
    return {
        'accuracy': accuracy,
        'recall_0': recall_0,
        'recall_1': recall_1,
        'f1_macro': f1_macro,
        'auroc': auroc,
        'auprc': auprc
    }


def compute_statistics(fold_metrics_list):
    """
    Compute mean and standard deviation across multiple folds.
    
    Args:
        fold_metrics_list: List of dictionaries, each containing metrics for one fold
        
    Returns:
        Dictionary with mean and std for each metric
    """
    if not fold_metrics_list:
        return {}
    
    # Get all metric names
    metric_names = fold_metrics_list[0].keys()
    
    statistics = {}
    for metric_name in metric_names:
        values = [fold_metrics[metric_name] for fold_metrics in fold_metrics_list]
        statistics[f'{metric_name}_mean'] = np.mean(values)
        statistics[f'{metric_name}_std'] = np.std(values, ddof=1)  # Sample std dev
        
    return statistics


def perform_statistical_tests(baseline_metrics, test_metrics, metric_name='auroc'):
    """
    Perform statistical significance tests comparing two methods.
    
    Uses paired t-test and Wilcoxon signed-rank test.
    
    Args:
        baseline_metrics: List of metric dictionaries for baseline method (one per fold)
        test_metrics: List of metric dictionaries for test method (one per fold)
        metric_name: Name of the metric to compare (default: 'auroc')
        
    Returns:
        Dictionary containing p-values and comparison statistics
    """
    if len(baseline_metrics) != len(test_metrics):
        raise ValueError("Baseline and test metrics must have same number of folds")
    
    # Extract metric values from each fold
    baseline_values = [m[metric_name] for m in baseline_metrics]
    test_values = [m[metric_name] for m in test_metrics]
    
    # Paired t-test
    try:
        t_statistic, t_pvalue = stats.ttest_rel(test_values, baseline_values)
    except:
        t_pvalue = 1.0
    
    # Wilcoxon signed-rank test
    try:
        wilcoxon_statistic, wilcoxon_pvalue = stats.wilcoxon(test_values, baseline_values)
    except:
        wilcoxon_pvalue = 1.0
    
    return {
        'paired_t_test_pvalue': t_pvalue,
        'wilcoxon_pvalue': wilcoxon_pvalue,
        'baseline_mean': np.mean(baseline_values),
        'test_mean': np.mean(test_values),
        'difference': np.mean(test_values) - np.mean(baseline_values),
        'baseline_std': np.std(baseline_values, ddof=1),
        'test_std': np.std(test_values, ddof=1)
    }


def format_results_table(fold_metrics_list, statistics=None, decimals=4):
    """
    Format evaluation results as a pretty-printed table.
    
    Args:
        fold_metrics_list: List of metric dictionaries (one per fold)
        statistics: Optional dictionary with mean/std statistics
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    output = []
    
    # Per-fold results
    for i, metrics in enumerate(fold_metrics_list, 1):
        fold_str = (
            f"Fold {i}: "
            f"Accuracy={format_metric(metrics['accuracy'], decimals)}, "
            f"Recall_0={format_metric(metrics['recall_0'], decimals)}, "
            f"Recall_1={format_metric(metrics['recall_1'], decimals)}, "
            f"F1_Macro={format_metric(metrics['f1_macro'], decimals)}, "
            f"AUROC={format_metric(metrics['auroc'], decimals)}, "
            f"AUPRC={format_metric(metrics['auprc'], decimals)}"
        )
        output.append(fold_str)
    
    # Statistics
    if statistics:
        output.append("\nCross-Validation Mean ± Std:")
        output.append(
            f"  Accuracy   = {format_metric(statistics['accuracy_mean'], decimals)} "
            f"± {format_metric(statistics['accuracy_std'], decimals)}"
        )
        output.append(
            f"  Recall_0   = {format_metric(statistics['recall_0_mean'], decimals)} "
            f"± {format_metric(statistics['recall_0_std'], decimals)}"
        )
        output.append(
            f"  Recall_1   = {format_metric(statistics['recall_1_mean'], decimals)} "
            f"± {format_metric(statistics['recall_1_std'], decimals)}"
        )
        output.append(
            f"  F1_Macro   = {format_metric(statistics['f1_macro_mean'], decimals)} "
            f"± {format_metric(statistics['f1_macro_std'], decimals)}"
        )
        output.append(
            f"  AUROC      = {format_metric(statistics['auroc_mean'], decimals)} "
            f"± {format_metric(statistics['auroc_std'], decimals)}"
        )
        output.append(
            f"  AUPRC      = {format_metric(statistics['auprc_mean'], decimals)} "
            f"± {format_metric(statistics['auprc_std'], decimals)}"
        )
    
    return '\n'.join(output)


def save_results_to_csv(fold_metrics_list, statistics, save_path):
    """
    Save evaluation results to a CSV file.
    
    Args:
        fold_metrics_list: List of metric dictionaries (one per fold)
        statistics: Dictionary with mean/std statistics
        save_path: Path to save CSV file
    """
    import csv
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Fold', 'Accuracy', 'Recall_0', 'Recall_1', 'F1_Macro', 'AUROC', 'AUPRC'
        ])
        
        # Per-fold results
        for i, metrics in enumerate(fold_metrics_list, 1):
            writer.writerow([
                i,
                format_metric(metrics['accuracy'], 4),
                format_metric(metrics['recall_0'], 4),
                format_metric(metrics['recall_1'], 4),
                format_metric(metrics['f1_macro'], 4),
                format_metric(metrics['auroc'], 4),
                format_metric(metrics['auprc'], 4)
            ])
        
        # Mean row
        writer.writerow([
            'Mean',
            format_metric(statistics['accuracy_mean'], 4),
            format_metric(statistics['recall_0_mean'], 4),
            format_metric(statistics['recall_1_mean'], 4),
            format_metric(statistics['f1_macro_mean'], 4),
            format_metric(statistics['auroc_mean'], 4),
            format_metric(statistics['auprc_mean'], 4)
        ])
        
        # Std row
        writer.writerow([
            'Std',
            format_metric(statistics['accuracy_std'], 4),
            format_metric(statistics['recall_0_std'], 4),
            format_metric(statistics['recall_1_std'], 4),
            format_metric(statistics['f1_macro_std'], 4),
            format_metric(statistics['auroc_std'], 4),
            format_metric(statistics['auprc_std'], 4)
        ])
