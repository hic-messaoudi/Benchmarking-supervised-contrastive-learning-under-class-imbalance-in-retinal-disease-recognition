"""
Evaluation script for Supervised Learning method.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import os
import sys

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


class SupervisedModelArchitecture(nn.Module):
    def __init__(self, backbone_name="tf_efficientnet_b0.ns_jft_in1k"):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.classifier = nn.Linear(1280, 1)
    
    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


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


def print_all_metrics(metrics, title=""):
    """Print all metrics on one line."""
    print(f"{title}Accuracy={format_metric(metrics['accuracy'])}, "
          f"Recall_0={format_metric(metrics['recall_0'])}, Recall_1={format_metric(metrics['recall_1'])}, "
          f"F1_Macro={format_metric(metrics['f1_macro'])}, AUROC={format_metric(metrics['auroc'])}, "
          f"AUPRC={format_metric(metrics['auprc'])}")


def main():
    device = 'cuda:0'
    batch_size = 256
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    seed_everything(42)
    test_data, test_labels = load_test_data(parent_dir)
    test_loader = create_test_dataloader(test_data, test_labels, batch_size=batch_size, num_workers=8)
    
    print_section_header("Supervised Model Evaluation")
    
    models, cv_metrics = [], []
    for fold in range(1, 6):
        model_path = os.path.join(current_dir, f'Supervised_fold_{fold}.pt')
        if not os.path.exists(model_path): continue
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        models.append(model)
        
        val_data, val_labels = load_validation_data(parent_dir, fold)
        val_loader = create_validation_dataloader(val_data, val_labels, batch_size=batch_size, num_workers=8)
        preds, labs = evaluate_fold(model, val_loader, device)
        cv_metrics.append(calculate_fold_metrics(preds, labs))
    
    if cv_metrics:
        cv_stats = compute_statistics(cv_metrics)
        print(format_results_table(cv_metrics, cv_stats))
        save_results_to_csv(cv_metrics, cv_stats, os.path.join(current_dir, 'results.csv'))
    
    if models:
        print_section_header("Ensemble on Test Set")
        pred_avg, labels = evaluate_ensemble(models, test_loader, device)
        ens = calculate_fold_metrics(pred_avg, labels)
        print_all_metrics(ens, "  ")
    
    print("\n" + "="*60 + "\nEvaluation complete!\n" + "="*60)


if __name__ == "__main__":
    main()
