"""
Model factory and loss functions for different training methods.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning loss from https://arxiv.org/pdf/2004.11362.pdf"""
    
    def __init__(self, temperature=0.1, contrast_mode='all', base_temperature=1, device='cuda:0'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device
        
    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...]
            labels: ground truth of shape [bsz]
            mask: contrastive mask of shape [bsz, bsz]
        Returns:
            A loss scalar
        """
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Handle edge cases when there is no positive pair
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastiveModelArchitecture(nn.Module):
    """Model architecture for contrastive learning methods"""
    
    def __init__(self, backbone_name="tf_efficientnet_b0.ns_jft_in1k", feature_dim=256):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.backbone.classifier = nn.Identity()
        self.feature_projector = nn.Linear(1280, feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(1280, 1)
        )
        
    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.feature_projector(x1)
        out = F.normalize(x2, dim=-1)
        x = self.classifier(x1)
        
        if self.training:
            return out, x
        else:
            return x


def create_supervised_model(backbone_name="tf_efficientnet_b0.ns_jft_in1k", num_classes=1, pretrained=True):
    """
    Create a supervised learning model.
    
    Args:
        backbone_name: Name of the timm model
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        PyTorch model
    """
    model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
    return model


def create_contrastive_model(backbone_name="tf_efficientnet_b0.ns_jft_in1k", feature_dim=256):
    """
    Create a contrastive learning model.
    
    Args:
        backbone_name: Name of the timm model
        feature_dim: Dimension of the contrastive feature space
        
    Returns:
        ContrastiveModelArchitecture instance
    """
    return ContrastiveModelArchitecture(backbone_name=backbone_name, feature_dim=feature_dim)


def get_loss_function(method_name, device='cuda:0'):
    """
    Get the appropriate loss function for a training method.
    
    Args:
        method_name: Name of the training method
        device: Device to use
        
    Returns:
        Loss function or tuple of loss functions
    """
    if method_name in ['Supervised', 'WCE']:
        return nn.BCELoss()
    elif method_name in ['SCL', 'SupMin', 'SupProto', 'SCLHN', 'SCLProCo', 'ProCo']:
        contrastive_loss = SupConLoss(device=device)
        classification_loss = nn.BCELoss()
        return contrastive_loss, classification_loss
    else:
        raise ValueError(f"Unknown method: {method_name}")
