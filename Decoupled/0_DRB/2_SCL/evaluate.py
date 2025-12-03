"""
Evaluation script for SCL (Decoupled) method.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
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
    perform_statistical_tests,
    format_results_table,
    save_results_to_csv,
    print_section_header,
    format_metric
)


# ============== DeLong Test ==============

def compute_midrank(x):
    """Compute midranks for DeLong test."""
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
    """Fast DeLong computation."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
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
    """
    Perform DeLong test to compare two AUROC values.
    Returns: auc1, auc2, z_stat, p_value
    """
    y_true = np.asarray(y_true).flatten()
    y_pred1 = np.asarray(y_pred1).flatten()
    y_pred2 = np.asarray(y_pred2).flatten()
    
    order = np.argsort(-y_true)  # Sort by label (1s first)
    y_true_sorted = y_true[order]
    y_pred1_sorted = y_pred1[order]
    y_pred2_sorted = y_pred2[order]
    
    label_1_count = int(np.sum(y_true))
    predictions_sorted_transposed = np.vstack([y_pred1_sorted, y_pred2_sorted])
    
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    
    # Compute z-statistic
    diff = aucs[0] - aucs[1]
    var = delongcov[0,0] + delongcov[1,1] - 2*delongcov[0,1]
    
    if var <= 0:
        return aucs[0], aucs[1], 0.0, 1.0
    
    z = diff / np.sqrt(var)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return aucs[0], aucs[1], z, p_value


# ============== Model Architectures ==============

class ContrastiveEncoder(nn.Module):
    """Encoder with backbone and projection head only (no classifier)."""
    
    def __init__(self, backbone_name="tf_efficientnet_b0.ns_jft_in1k", feature_dim=256):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.backbone.classifier = nn.Identity()
        self.projector = nn.Linear(1280, feature_dim)

    def forward(self, x):
        features = self.backbone(x)
        proj = F.normalize(self.projector(features), dim=-1)
        return proj


class ContrastiveModelArchitecture(nn.Module):
    def __init__(self, backbone_name="tf_efficientnet_b0.ns_jft_in1k", feature_dim=256):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.backbone.classifier = nn.Identity()
        self.feature_projector = nn.Linear(1280, feature_dim)
    
    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.feature_projector(x1)
        return F.normalize(x2, dim=-1)


class Classifier(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        loaded = torch.load(os.path.join(base_dir, backbone_name), map_location='cuda:0', weights_only=False)
        self.backbone = loaded.backbone
        self.classifier = nn.Sequential(nn.Linear(1280, 1))
    
    def forward(self, x):
        return self.classifier(self.backbone(x))


# ============== Evaluation Functions ==============

def extract_features(model, data_loader, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, labs in data_loader:
            out = model(images.to(device))
            features.append(out.cpu())
            labels.append(labs.cpu())
    return torch.cat(features, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


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


def plot_tsne_2d(features, labels, save_path, title):
    """2D t-SNE with class legend."""
    print(f"  Computing 2D t-SNE for {len(features)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    feat_2d = tsne.fit_transform(features)
    
    labels_flat = labels.flatten()
    plt.figure(figsize=(10, 8))
    
    mask0 = labels_flat == 0
    mask1 = labels_flat == 1
    plt.scatter(feat_2d[mask0, 0], feat_2d[mask0, 1], c='blue', alpha=0.6, s=10, label='Class 0', marker='o')
    plt.scatter(feat_2d[mask1, 0], feat_2d[mask1, 1], c='red', alpha=0.6, s=10, label='Class 1', marker='o')
    
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(loc='best', markerscale=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tsne_sphere(features, labels, save_path, title):
    """3D t-SNE on sphere with class legend."""
    print(f"  Computing 3D t-SNE (sphere) for {len(features)} samples...")
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(features)-1))
    feat_3d = tsne.fit_transform(features)
    
    norms = np.linalg.norm(feat_3d, axis=1, keepdims=True)
    feat_sphere = feat_3d / norms
    
    labels_flat = labels.flatten()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.2, linewidth=0.5)
    
    mask0 = labels_flat == 0
    mask1 = labels_flat == 1
    ax.scatter(feat_sphere[mask0, 0], feat_sphere[mask0, 1], feat_sphere[mask0, 2],
               c='blue', alpha=0.7, s=15, label='Class 0', marker='o')
    ax.scatter(feat_sphere[mask1, 0], feat_sphere[mask1, 1], feat_sphere[mask1, 2],
               c='red', alpha=0.7, s=15, label='Class 1', marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_box_aspect([1,1,1])
    ax.legend(loc='best', markerscale=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def load_supervised_baseline_cv(parent_dir, device, batch_size):
    """Load supervised CV metrics and predictions for comparison."""
    # Try local Decoupled first, then Coupled
    supervised_dir = os.path.join(parent_dir, '1_Supervised')
    data_parent = parent_dir
    if not os.path.exists(supervised_dir):
        coupled_parent = parent_dir.replace('Decoupled', 'Coupled')
        supervised_dir = os.path.join(coupled_parent, '1_Supervised')
        data_parent = coupled_parent
        if not os.path.exists(supervised_dir):
            # Try without number prefix (Coupled style)
            supervised_dir = os.path.join(coupled_parent, 'Supervised')
            if not os.path.exists(supervised_dir):
                return None, None, None
    
    cv_metrics = []
    all_preds, all_labels = [], []
    for fold in range(1, 6):
        model_path = os.path.join(supervised_dir, f'Supervised_fold_{fold}.pt')
        if not os.path.exists(model_path): return None, None, None
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        val_data, val_labels = load_validation_data(data_parent, fold)
        val_loader = create_validation_dataloader(val_data, val_labels, batch_size=batch_size, num_workers=8)
        preds, labs = evaluate_fold(model, val_loader, device)
        cv_metrics.append(calculate_fold_metrics(preds, labs))
        all_preds.append(preds)
        all_labels.append(labs)
    
    pooled_preds = torch.cat(all_preds, dim=0).numpy()
    pooled_labels = torch.cat(all_labels, dim=0).numpy()
    return cv_metrics, pooled_preds, pooled_labels


def load_supervised_ensemble_test(parent_dir, test_loader, device):
    """Load supervised ensemble test predictions for comparison."""
    # Try local Decoupled first, then Coupled
    supervised_dir = os.path.join(parent_dir, '1_Supervised')
    if not os.path.exists(supervised_dir):
        coupled_parent = parent_dir.replace('Decoupled', 'Coupled')
        supervised_dir = os.path.join(coupled_parent, '1_Supervised')
        if not os.path.exists(supervised_dir):
            # Try without number prefix (Coupled style)
            supervised_dir = os.path.join(coupled_parent, 'Supervised')
            if not os.path.exists(supervised_dir):
                return None, None, None
    
    models = []
    for fold in range(1, 6):
        model_path = os.path.join(supervised_dir, f'Supervised_fold_{fold}.pt')
        if not os.path.exists(model_path): return None, None, None
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        models.append(model)
    
    pred_avg, labels = evaluate_ensemble(models, test_loader, device)
    metrics = calculate_fold_metrics(pred_avg, labels)
    return metrics, pred_avg.numpy(), labels.numpy()


def format_pvalue(p):
    """Format p-value: scientific notation for small values (e.g., 3.2×10⁻⁴)."""
    if p == 0:
        return "< 1×10⁻¹⁰"
    if p >= 0.01:
        return f"{p:.4f}"
    exp = int(np.floor(np.log10(abs(p))))
    mantissa = p / (10 ** exp)
    superscripts = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻'}
    exp_str = ''.join(superscripts[c] for c in str(exp))
    return f"{mantissa:.1f}×10{exp_str}"


def print_stat_comparison_cv(method_metrics, baseline_metrics, method_name, method_pooled_preds, method_pooled_labels, sup_pooled_preds, sup_pooled_labels):
    """Compare CV AUROC: fold-by-fold + DeLong on pooled predictions."""
    if baseline_metrics is None or len(baseline_metrics) != len(method_metrics):
        print("  (Supervised baseline not available for comparison)")
        return
    
    # Fold-by-fold stats
    method_aurocs = [m['auroc'] for m in method_metrics]
    sup_aurocs = [m['auroc'] for m in baseline_metrics]
    
    method_mean = np.mean(method_aurocs)
    method_std = np.std(method_aurocs)
    sup_mean = np.mean(sup_aurocs)
    sup_std = np.std(sup_aurocs)
    diff = method_mean - sup_mean
    
    # Paired t-test and Wilcoxon
    t_stat, t_pval = stats.ttest_rel(method_aurocs, sup_aurocs)
    try:
        w_stat, w_pval = stats.wilcoxon(method_aurocs, sup_aurocs)
    except:
        w_pval = 1.0
    
    # DeLong on pooled CV predictions
    if sup_pooled_preds is not None and method_pooled_preds is not None:
        _, _, z_delong, p_delong = delong_test(method_pooled_labels, method_pooled_preds, sup_pooled_preds)
    else:
        p_delong = 1.0
    
    print(f"\n  CV Comparison (AUROC): {method_name} vs Supervised")
    print(f"    {method_name}: {method_mean:.4f} ± {method_std:.4f}")
    print(f"    Supervised: {sup_mean:.4f} ± {sup_std:.4f}")
    print(f"    Difference: {diff:+.4f}")
    print(f"    p-values: t-test={format_pvalue(t_pval)}, Wilcoxon={format_pvalue(w_pval)}, DeLong={format_pvalue(p_delong)}")


def print_all_metrics(metrics, title=""):
    """Print all metrics on one line."""
    print(f"{title}Accuracy={format_metric(metrics['accuracy'])}, "
          f"Recall_0={format_metric(metrics['recall_0'])}, Recall_1={format_metric(metrics['recall_1'])}, "
          f"F1_Macro={format_metric(metrics['f1_macro'])}, AUROC={format_metric(metrics['auroc'])}, "
          f"AUPRC={format_metric(metrics['auprc'])}")


def print_ensemble_comparison(method_preds, method_labels, sup_preds, sup_labels, method_auroc, sup_auroc, method_name):
    """Print ensemble test comparison with DeLong test."""
    if sup_preds is None:
        print("  (Supervised baseline not available)")
        return
    
    diff = method_auroc - sup_auroc
    _, _, z_stat, p_delong = delong_test(method_labels, method_preds, sup_preds)
    
    print(f"\n  Ensemble Test AUROC Comparison: {method_name} vs Supervised")
    print(f"    {method_name}: {method_auroc:.4f}")
    print(f"    Supervised: {sup_auroc:.4f}")
    print(f"    Difference: {diff:+.4f}")
    print(f"    DeLong p-value: {format_pvalue(p_delong)}")


def main():
    device = 'cuda:0'
    batch_size = 256
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    seed_everything(42)
    test_data, test_labels = load_test_data(parent_dir)
    test_loader = create_test_dataloader(test_data, test_labels, batch_size=batch_size, num_workers=8)
    
    # Load supervised baselines
    supervised_cv, sup_cv_preds, sup_cv_labels = load_supervised_baseline_cv(parent_dir, device, batch_size)
    supervised_test, sup_test_preds, sup_test_labels = load_supervised_ensemble_test(parent_dir, test_loader, device)
    supervised_test_auroc = supervised_test['auroc'] if supervised_test else None
    
    # =========================================================================
    # PART 1: Contrastive Encoder - t-SNE only
    # =========================================================================
    print_section_header("Part 1: Contrastive Encoder (SCL_fold_*.pt)")
    
    encoder_path = os.path.join(current_dir, 'SCL_fold_1.pt')
    if os.path.exists(encoder_path):
        print("\n  Generating t-SNE visualizations...")
        encoder = torch.load(encoder_path, map_location=device, weights_only=False)
        encoder.eval()
        features, labels_np = extract_features(encoder, test_loader, device)
        plot_tsne_2d(features, labels_np, os.path.join(current_dir, 'tsne_2d.png'), "SCL t-SNE 2D")
        plot_tsne_sphere(features, labels_np, os.path.join(current_dir, 'tsne_sphere.png'), "SCL t-SNE Sphere")
    
    # =========================================================================
    # PART 2: Finetuned Classifier
    # =========================================================================
    print_section_header("Part 2: Finetuned Classifier (SCL_classifier_fold_*.pt)")
    
    finetuned_models, cv_metrics = [], []
    cv_preds_list, cv_labels_list = [], []
    for fold in range(1, 6):
        model_path = os.path.join(current_dir, f'SCL_classifier_fold_{fold}.pt')
        if not os.path.exists(model_path): continue
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        finetuned_models.append(model)
        
        val_data, val_labels = load_validation_data(parent_dir, fold)
        val_loader = create_validation_dataloader(val_data, val_labels, batch_size=batch_size, num_workers=8)
        preds, labs = evaluate_fold(model, val_loader, device)
        cv_metrics.append(calculate_fold_metrics(preds, labs))
        cv_preds_list.append(preds)
        cv_labels_list.append(labs)
    
    if cv_metrics:
        print("\nCross-Validation Results:")
        cv_stats = compute_statistics(cv_metrics)
        print(format_results_table(cv_metrics, cv_stats))
        
        pooled_preds = torch.cat(cv_preds_list, dim=0).numpy()
        pooled_labels = torch.cat(cv_labels_list, dim=0).numpy()
        print_stat_comparison_cv(cv_metrics, supervised_cv, 'SCLS', pooled_preds, pooled_labels, sup_cv_preds, sup_cv_labels)
        save_results_to_csv(cv_metrics, cv_stats, os.path.join(current_dir, 'results.csv'))
    
    if finetuned_models:
        print_section_header("Finetuned Model - Ensemble Test")
        pred_avg, labels = evaluate_ensemble(finetuned_models, test_loader, device)
        ens = calculate_fold_metrics(pred_avg, labels)
        print_all_metrics(ens, "  ")
        print_ensemble_comparison(pred_avg.numpy(), labels.numpy(), sup_test_preds, sup_test_labels, ens['auroc'], supervised_test_auroc, 'SCLS')
    
    print("\n" + "="*60 + "\nEvaluation complete!\n" + "="*60)


if __name__ == "__main__":
    main()
