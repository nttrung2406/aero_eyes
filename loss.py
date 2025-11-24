import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, min=-100, max=100)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_factor = torch.full_like(targets, self.alpha)
        alpha_t = torch.where(targets==1, alpha_factor, 1-alpha_factor)
        focal_loss = (1-pt)**self.gamma * alpha_t * bce_loss
        focal_loss = torch.where(torch.isnan(focal_loss), torch.zeros_like(focal_loss), focal_loss)
        return focal_loss.mean() if self.reduction=='mean' else focal_loss.sum()


class SSDMatcher:
    """Simple SSD-style anchor matcher"""
    def __init__(self, pos_threshold, neg_threshold):
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def __call__(self, match_quality_matrix):
        if match_quality_matrix.numel() == 0:
            return torch.full((match_quality_matrix.shape[1],), -1, dtype=torch.int64)

        matched_vals, matches = match_quality_matrix.max(dim=0)
        labels = matches.clone()
        labels[matched_vals < self.neg_threshold] = -1
        labels[(matched_vals >= self.neg_threshold) & (matched_vals < self.pos_threshold)] = -2
        return labels


class SSDBoxCoder:
    """Box encoder for SSD-style regression"""
    def __init__(self, weights=(10.0, 10.0, 5.0, 5.0)):
        self.weights = weights

    def encode_single(self, gt_boxes, anchors):
        wx, wy, ww, wh = self.weights
        wa = anchors[:, 2] - anchors[:, 0]
        ha = anchors[:, 3] - anchors[:, 1]
        xa = anchors[:, 0] + 0.5 * wa
        ya = anchors[:, 1] + 0.5 * ha

        wg = gt_boxes[:, 2] - gt_boxes[:, 0]
        hg = gt_boxes[:, 3] - gt_boxes[:, 1]
        xg = gt_boxes[:, 0] + 0.5 * wg
        yg = gt_boxes[:, 1] + 0.5 * hg

        dx = wx * (xg - xa) / wa
        dy = wy * (yg - ya) / ha
        dw = ww * torch.log(wg / wa)
        dh = wh * torch.log(hg / ha)

        return torch.stack((dx, dy, dw, dh), dim=1)


def get_loss_components():
    """Initialize loss components"""
    matcher = SSDMatcher(pos_threshold=0.4, neg_threshold=0.3)
    box_coder = SSDBoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
    focal_criterion = FocalLoss(alpha=0.25, gamma=3.0)
    
    return matcher, box_coder, focal_criterion