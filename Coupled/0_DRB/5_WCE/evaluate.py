"""
Evaluation script for WCE method (no contrastive features).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import os
import sys
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

from utils import (
    seed_everything,
    load_test_data,
    load_validation_data,
    create_test_dataloader,
    create_validation_dataloader,
    calculate_fold_metrics,
    compute_statistics,
    format_results_table,
    save_results_to_csv,
    print_section_header,
    format_metric
)


# ============== DeLong Test ==============

def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]
    aucs = np.zeros(k)
    score = np.zeros((k, m + n))
    for r in range(k):
        r_sorted = np.argsort(predictions_sorted_transposed[r, :])
        score[r, :] = compute_midrank(predictions_sorted_transposed[r, r_sorted])[np.argsort(r_sorted)]
        aucs[r] = (np.sum(score[r, :m]) - m*(m+1)/2.0) / (m*n)
    v01 = (score[:, :m] - np.arange(1, m+1)) / n
    v10 = 1.0 - (score[:, m:] - np.arange(1, n+1)) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx/m + sy/n
    return aucs, delongcov


def delong_test(y_true, y_pred1, y_pred2):
    y_true = np.asarray(y_true).flatten()
    y_pred1 = np.asarray(y_pred1).flatten()
    y_pred2 = np.asarray(y_pred2).flatten()
    order = np.argsort(-y_true)
    y_pred1_sorted = y_pred1[order]
    y_pred2_sorted = y_pred2[order]
    label_1_count = int(np.sum(y_true))
    predictions_sorted_transposed = np.vstack([y_pred1_sorted, y_pred2_sorted])
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    diff = aucs[0] - aucs[1]
    var = delongcov[0,0] + delongcov[1,1] - 2*delongcov[0,1]
    if var <= 0:
        return aucs[0], aucs[1], 0.0, 1.0
    z = diff / np.sqrt(var)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs[0], aucs[1], z, p_value


# ============== Model Architecture ==============

class ModelArchitecture(nn.Module):
    def __init__(self, backbone_name="tf_efficientnet_b0.ns_jft_in1k"):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.classifier = nn.Linear(1280, 1)
    
    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


# ============== Evaluation Functions ==============

def evaluate_fold(model, data_loader, device):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = torch.sigmoid(model(images.to(device))).to(torch.float32)
            preds.append(outputs.cpu())
            labs.append(labels.cpu())
    return torch.cat(preds, dim=0), torch.cat(labs, dim=0)


def evaluate_ensemble(models, test_loader, device):
    all_preds, labels = [], None
    for model in models:
        preds, labs = evaluate_fold(model, test_loader, device)
        all_preds.append(preds)
        if labels is None: labels = labs
    return torch.stack(all_preds).mean(dim=0), labels


def load_supervised_baseline_cv(parent_dir, device, batch_size):
    supervised_dir = os.path.join(parent_dir, '1_Supervised')
    if not os.path.exists(supervised_dir): return None, None, None
    cv_metrics, all_preds, all_labels = [], [], []
    for fold in range(1, 6):
        model_path = os.path.join(supervised_dir, f'Supervised_fold_{fold}.pt')
        if not os.path.exists(model_path): return None, None, None
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        val_data, val_labels = load_validation_data(parent_dir, fold)
        val_loader = create_validation_dataloader(val_data, val_labels, batch_size=batch_size, num_workers=8)
        preds, labs = evaluate_fold(model, val_loader, device)
        cv_metrics.append(calculate_fold_metrics(preds, labs))
        all_preds.append(preds)
        all_labels.append(labs)
    return cv_metrics, torch.cat(all_preds, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()


def load_supervised_ensemble_test(parent_dir, test_loader, device):
    supervised_dir = os.path.join(parent_dir, '1_Supervised')
    if not os.path.exists(supervised_dir): return None, None, None
    models = []
    for fold in range(1, 6):
        model_path = os.path.join(supervised_dir, f'Supervised_fold_{fold}.pt')
        if not os.path.exists(model_path): return None, None, None
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        models.append(model)
    pred_avg, labels = evaluate_ensemble(models, test_loader, device)
    return calculate_fold_metrics(pred_avg, labels), pred_avg.numpy(), labels.numpy()


def format_pvalue(p):
    if p == 0: return "< 1×10⁻¹⁰"
    if p >= 0.01: return f"{p:.4f}"
    exp = int(np.floor(np.log10(abs(p))))
    mantissa = p / (10 ** exp)
    superscripts = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻'}
    exp_str = ''.join(superscripts[c] for c in str(exp))
    return f"{mantissa:.1f}×10{exp_str}"


def print_stat_comparison_cv(method_metrics, baseline_metrics, method_name, method_preds, method_labels, sup_preds, sup_labels):
    if baseline_metrics is None: 
        print("  (Supervised baseline not available)")
        return
    method_aurocs = [m['auroc'] for m in method_metrics]
    sup_aurocs = [m['auroc'] for m in baseline_metrics]
    method_mean, method_std = np.mean(method_aurocs), np.std(method_aurocs)
    sup_mean, sup_std = np.mean(sup_aurocs), np.std(sup_aurocs)
    t_stat, t_pval = stats.ttest_rel(method_aurocs, sup_aurocs)
    try: w_stat, w_pval = stats.wilcoxon(method_aurocs, sup_aurocs)
    except: w_pval = 1.0
    _, _, _, p_delong = delong_test(method_labels, method_preds, sup_preds) if sup_preds is not None else (0,0,0,1.0)
    print(f"\n  CV Comparison (AUROC): {method_name} vs Supervised")
    print(f"    {method_name}: {method_mean:.4f} ± {method_std:.4f}")
    print(f"    Supervised: {sup_mean:.4f} ± {sup_std:.4f}")
    print(f"    Difference: {method_mean - sup_mean:+.4f}")
    print(f"    p-values: t-test={format_pvalue(t_pval)}, Wilcoxon={format_pvalue(w_pval)}, DeLong={format_pvalue(p_delong)}")


def print_all_metrics(metrics, title=""):
    print(f"{title}Accuracy={format_metric(metrics['accuracy'])}, "
          f"Recall_0={format_metric(metrics['recall_0'])}, Recall_1={format_metric(metrics['recall_1'])}, "
          f"F1_Macro={format_metric(metrics['f1_macro'])}, AUROC={format_metric(metrics['auroc'])}, "
          f"AUPRC={format_metric(metrics['auprc'])}")


def print_ensemble_comparison(method_preds, method_labels, sup_preds, sup_labels, method_auroc, sup_auroc, method_name):
    if sup_preds is None:
        print("  (Supervised baseline not available)")
        return
    _, _, _, p_delong = delong_test(method_labels, method_preds, sup_preds)
    print(f"\n  Ensemble Test AUROC Comparison: {method_name} vs Supervised")
    print(f"    {method_name}: {method_auroc:.4f}")
    print(f"    Supervised: {sup_auroc:.4f}")
    print(f"    Difference: {method_auroc - sup_auroc:+.4f}")
    print(f"    DeLong p-value: {format_pvalue(p_delong)}")


def main():
    device = 'cuda:0'
    batch_size = 256
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    seed_everything(42)
    test_data, test_labels = load_test_data(parent_dir)
    test_loader = create_test_dataloader(test_data, test_labels, batch_size=batch_size, num_workers=8)
    
    supervised_cv, sup_cv_preds, sup_cv_labels = load_supervised_baseline_cv(parent_dir, device, batch_size)
    supervised_test, sup_test_preds, sup_test_labels = load_supervised_ensemble_test(parent_dir, test_loader, device)
    supervised_test_auroc = supervised_test['auroc'] if supervised_test else None
    
    print_section_header("WCE Model Evaluation")
    
    models, cv_metrics = [], []
    cv_preds_list, cv_labels_list = [], []
    for fold in range(1, 6):
        model_path = os.path.join(current_dir, f'WCE_fold_{fold}.pt')
        if not os.path.exists(model_path): continue
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        models.append(model)
        
        val_data, val_labels = load_validation_data(parent_dir, fold)
        val_loader = create_validation_dataloader(val_data, val_labels, batch_size=batch_size, num_workers=8)
        preds, labs = evaluate_fold(model, val_loader, device)
        cv_metrics.append(calculate_fold_metrics(preds, labs))
        cv_preds_list.append(preds)
        cv_labels_list.append(labs)
    
    if cv_metrics:
        cv_stats = compute_statistics(cv_metrics)
        print(format_results_table(cv_metrics, cv_stats))
        pooled_preds = torch.cat(cv_preds_list, dim=0).numpy()
        pooled_labels = torch.cat(cv_labels_list, dim=0).numpy()
        print_stat_comparison_cv(cv_metrics, supervised_cv, 'WCE', pooled_preds, pooled_labels, sup_cv_preds, sup_cv_labels)
        save_results_to_csv(cv_metrics, cv_stats, os.path.join(current_dir, 'results.csv'))
    
    if models:
        print_section_header("Ensemble on Test Set")
        pred_avg, labels = evaluate_ensemble(models, test_loader, device)
        ens = calculate_fold_metrics(pred_avg, labels)
        print_all_metrics(ens, "  ")
        print_ensemble_comparison(pred_avg.numpy(), labels.numpy(), sup_test_preds, sup_test_labels, ens['auroc'], supervised_test_auroc, 'WCE')
    
    print("\n" + "="*60 + "\nEvaluation complete!\n" + "="*60)


if __name__ == "__main__":
    main()
