import torch
import torch.nn.functional as F
from torchvision.ops import box_iou


def compute_ssd_loss(cls_logits, bbox_regression, anchors, targets, focal_criterion, box_coder, matcher):
    """
    Compute SSD loss with robust error handling.
    This version correctly handles batched model outputs.
    - cls_logits: List of 3 tensors, each [B, 1, H, W]
    - bbox_regression: List of 3 tensors, each [B, 4, H, W]
    - anchors: List of 3 tensors, each [N, 4]
    - targets: List of B dictionaries for ground truth
    """
    device = cls_logits[0].device
    batch_size = len(targets)


    cls_preds_flat = []
    box_preds_flat = []
    for cls_level, box_level in zip(cls_logits, bbox_regression):
        B, _, H, W = cls_level.shape
        cls_preds_flat.append(cls_level.permute(0, 2, 3, 1).reshape(B, -1))
        box_preds_flat.append(box_level.permute(0, 2, 3, 1).reshape(B, -1, 4))

    cls_preds_cat = torch.cat(cls_preds_flat, dim=1)
    box_preds_cat = torch.cat(box_preds_flat, dim=1)
    anchors_cat = torch.cat(anchors, dim=0).to(device) 

    all_cls_targets = []
    all_reg_targets = []
    all_pos_masks = []
    num_pos_total = 0

    for i in range(batch_size):
        gt_boxes = targets[i]['boxes'].to(device)
        
        if gt_boxes.numel() == 0:
            matches = torch.full((anchors_cat.shape[0],), -1, dtype=torch.int64, device=device)
        else:
            # Match anchors to ground truth boxes using IoU
            match_quality_matrix = box_iou(gt_boxes, anchors_cat)
            matches = matcher(match_quality_matrix)

        pos_mask = matches >= 0
        neg_mask = matches == -1
        num_pos_total += pos_mask.sum().item()

        cls_target = torch.zeros_like(matches, dtype=torch.float32)
        cls_target[pos_mask] = 1.0
        
        if pos_mask.sum() > 0:
            matched_gt_boxes = gt_boxes[matches[pos_mask]]
            reg_target = box_coder.encode_single(matched_gt_boxes, anchors_cat[pos_mask])
        else:
            # Create an empty tensor if no positive matches
            reg_target = torch.empty((0, 4), dtype=torch.float32, device=device)

        all_cls_targets.append(cls_target)
        all_reg_targets.append(reg_target)
        all_pos_masks.append(pos_mask)

    cls_targets_flat = torch.cat(all_cls_targets, dim=0)
    pos_mask_flat = torch.cat(all_pos_masks, dim=0)
    

    loss_cls = focal_criterion(cls_preds_cat.reshape(-1), cls_targets_flat)

    # Regression loss (Smooth L1) only on positive anchors
    if num_pos_total > 0:
        reg_targets_flat = torch.cat([t for t in all_reg_targets if t.numel() > 0], dim=0)
        loss_reg = F.smooth_l1_loss(
            box_preds_cat.reshape(-1, 4)[pos_mask_flat],
            reg_targets_flat,
            reduction='sum'
        ) / num_pos_total
    else:
        loss_reg = torch.tensor(0.0, device=device)

    if torch.isnan(loss_cls) or torch.isinf(loss_cls): loss_cls = torch.tensor(0.0, device=device)
    if torch.isnan(loss_reg) or torch.isinf(loss_reg): loss_reg = torch.tensor(0.0, device=device)

    return {
        "loss_focal_cls": loss_cls,
        "loss_box_reg": loss_reg,
        "num_pos": num_pos_total,
        "num_total": cls_targets_flat.shape[0]
    }