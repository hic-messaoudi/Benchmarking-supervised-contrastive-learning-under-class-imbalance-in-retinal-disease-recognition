# Benchmarking Supervised Contrastive Learning Under Class Imbalance in Retinal Disease Recognition

> **Official implementation of the paper currently under review in Scientific Reports journal.**

## Abstract

Contrastive learning has emerged as a powerful paradigm for visual representation learning. However, its application to medical image classification, particularly in conditions of class imbalance, remains insufficiently investigated. Current evaluations of supervised contrastive approaches lack a comprehensive assessment in long-tailed medical imaging scenarios, where these methods demonstrate varying effectiveness depending on the degree of imbalance present. While individual models exhibit inconsistent performance, ensemble approaches consistently yield superior results and facilitate more reliable comparisons in skewed data distributions. In this article, we benchmark recent supervised contrastive learning techniques for diabetic retinopathy and glaucoma detection from fundus images across varying imbalance ratios, assessing both single and ensemble model configurations. This work presents the first systematic evaluation of supervised contrastive learning in retinal disease recognition, highlighting the advantages of ensemble-based strategies and offering practical insights for deploying these methods in imbalanced clinical imaging applications.

---

## Repository Structure

```
├── Coupled/                    # Coupled training approach (joint optimization)
│   ├── 0_DRB/                  # Diabetic Retinopathy Binary classification
│   │   ├── 1_Supervised/       # Standard supervised learning (BCE loss)
│   │   ├── 2_SCL/              # Supervised Contrastive Learning
│   │   ├── 3_ProCo/            # Probabilistic Contrastive Learning
│   │   ├── 4_SupMin/           # Supervised Minority Contrastive Learning
│   │   ├── 5_WCE/              # Weighted Cross-Entropy
│   │   ├── 6_SupProto/         # Supervised Contrastive with Prototypes
│   │   ├── 7_SCLHN/            # SCL with Hybrid Networks (decaying alpha)
│   │   └── DR/                 # Diabetic Retinopathy dataset (5-fold + test)
│   │
│   └── 1_AIROGS/               # AIROGS Glaucoma dataset
│       ├── 1_Supervised/
│       ├── 2_SCL/
│       ├── 3_ProCo/
│       ├── 4_SupMin/
│       ├── 5_WCE/
│       ├── 6_SupProto/
│       ├── 7_SCLHN/
│       └── AIROGS/             # AIROGS dataset (5-fold + test)
│
├── Decoupled/                  # Decoupled training approach (two-stage)
│   ├── 0_DRB/                  # Diabetic Retinopathy Binary classification
│   │   ├── 1_Supervised/
│   │   ├── 2_SCL/
│   │   ├── 3_ProCo/
│   │   ├── 4_SupMin/
│   │   ├── 6_SupProto/
│   │   └── DR/
│   │
│   └── 1_AIROGS/               # AIROGS Glaucoma dataset
│       ├── 1_Supervised/
│       ├── 2_SCL/
│       ├── 3_ProCo/
│       ├── 4_SupMin/
│       ├── 6_SupProto/
│       └── AIROGS/
│
└── utils/                      # Shared utilities
    ├── __init__.py
    ├── common.py               # Common functions (seeding, formatting)
    ├── data_loader.py          # Data loading utilities for k-fold CV
    ├── metrics.py              # Evaluation metrics and statistical tests
    └── model_factory.py        # Model creation utilities
```

---

## Training Approaches

### Coupled Training
In the **coupled** approach, the contrastive loss and classification loss are jointly optimized in a single training phase. The model learns representations and classification simultaneously.

### Decoupled Training
In the **decoupled** approach, training is performed in two stages:
1. **Stage 1**: Train the encoder using only the contrastive loss
2. **Stage 2**: Freeze the encoder and train a linear classifier on the learned representations

---

## Methods Implemented

| Method | Description |
|--------|-------------|
| **Supervised** | Standard supervised learning with Binary Cross-Entropy (BCE) loss |
| **WCE** | Weighted Cross-Entropy loss to handle class imbalance |
| **SCL** | Supervised Contrastive Learning - uses label information to pull same-class samples together |
| **ProCo** | Probabilistic Contrastive Learning with von Mises-Fisher distribution for class prototypes |
| **SupMin** | Supervised contrastive loss for minority class, unsupervised for majority class |
| **SupProto** | Supervised Contrastive Learning extended with class prototypes |
| **SCLHN** | SCL with Hybrid Networks - uses decaying alpha to balance contrastive and classification losses |

---

## Datasets

### Diabetic Retinopathy Binary (DRB)
- **Task**: Binary classification (No DR & Mild vs Moderate, Severe &  Proliferative DR)
- **Structure**: 5-fold cross-validation + held-out test set
- **Classes**: `0` (non-referable), `1` (referable)

### AIROGS (Glaucoma Detection)
- **Task**: Binary classification (glaucoma vs. no glaucoma)
- **Structure**: 5-fold cross-validation + held-out test set
- **Classes**: `0` (no glaucoma), `1` (glaucoma)

### Data Organization
```
Dataset/
├── Fold1/
│   ├── 0/          # Negative class images (*.jpg)
│   └── 1/          # Positive class images (*.jpg)
├── Fold2/
│   ├── 0/
│   └── 1/
├── Fold3/
│   ├── 0/
│   └── 1/
├── Fold4/
│   ├── 0/
│   └── 1/
├── Fold5/
│   ├── 0/
│   └── 1/
└── Test/
    ├── 0/
    └── 1/
```

---

## How It Works

### Training Pipeline

1. **Data Loading**: Images are loaded from the folder structure using 5-fold cross-validation
2. **Augmentation**: Training images undergo augmentation (random flips, crops, color jitter, etc.)
3. **Model**: EfficientNet-B0 backbone pretrained on ImageNet, with:
   - A projection head for contrastive methods (maps to 256-dim feature space)
   - A classification head for supervised/coupled training
4. **Training**: Models are trained for each fold with early stopping based on validation metrics
5. **Evaluation**: Both single-fold and ensemble predictions are evaluated

### Running Training

```bash
# Navigate to the desired method directory
cd Coupled/0_DRB/2_SCL/

# Run training
python train.py
```

### Running Evaluation

```bash
# Navigate to the desired method directory
cd Coupled/0_DRB/2_SCL/

# Run evaluation (loads saved models)
python evaluate.py
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEED` | 42 | Random seed for reproducibility |
| `BATCH_SIZE` | 128/256 | Training batch size |
| `EPOCHS` | Variable | Maximum training epochs |
| `LEARNING_RATE` | 3e-4 | Initial learning rate |
| `PATIENCE` | 20 | Early stopping patience |
| `FEATURE_DIM` | 256 | Projection head output dimension |
| `DEVICE` | cuda:0 | Training device |

---

## Evaluation Metrics

The following metrics are computed for each method:

- **Accuracy**: Overall classification accuracy
- **Recall (Class 0)**: Sensitivity for negative class
- **Recall (Class 1)**: Sensitivity for positive class (minority)
- **F1-Macro**: Macro-averaged F1 score
- **AUROC**: Area Under the ROC Curve
- **AUPRC**: Area Under the Precision-Recall Curve

### Ensemble Evaluation

For each method, an **ensemble** of the 5-fold models is created by averaging predictions. This ensemble approach consistently yields more stable and often superior results, especially under class imbalance.

---

## Requirements

- Python 3.8+
- PyTorch
- timm (PyTorch Image Models)
- torcheval
- scikit-learn
- numpy
- PIL
- natsort
- scipy

---

## Citation

If you use this code in your research, please cite our paper (citation will be updated upon publication).

---

## Acknowledgements

This work is supported by a French government grant managed by the Agence Nationale de la Recherche under the title "France 2030", reference ANR-21-PMRB-0009.
