import torch
from tqdm import tqdm

from compute_loss import compute_ssd_loss


def validate_one_epoch(model, val_loader, device, focal_criterion, box_coder, matcher):
    """
    Performs one full validation pass over the validation data.
    """
    model.eval()
    total_val_loss = 0.0
    total_val_cls_loss = 0.0
    total_val_reg_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"           [V]")
        for batch in progress_bar:
            frames = batch["frame"].to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch["targets"]]
            ref_images = batch["ref_images"].to(device)
            
            cls_logits, bbox_regression, anchors = model(frames, ref_images=ref_images)
            
            if not cls_logits:
                continue

            loss_dict = compute_ssd_loss(
                cls_logits, bbox_regression, anchors,
                targets, focal_criterion, box_coder, matcher
            )
            
            loss = loss_dict["loss_focal_cls"] + loss_dict["loss_box_reg"]
            
            total_val_loss += loss.item()
            total_val_cls_loss += loss_dict["loss_focal_cls"].item()
            total_val_reg_loss += loss_dict["loss_box_reg"].item()
            
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_cls_loss = total_val_cls_loss / len(val_loader)
    avg_val_reg_loss = total_val_reg_loss / len(val_loader)
    
    return avg_val_loss, avg_val_cls_loss, avg_val_reg_loss


def evaluate_model(model_path, use_kaggle=True):
    """Evaluate a trained model"""
    from setup import setup_training_components
    
    components = setup_training_components(use_kaggle=use_kaggle)
    
    model = components['model']
    val_loader = components['val_loader']
    device = components['device']
    focal_criterion = components['focal_criterion']
    box_coder = components['box_coder']
    matcher = components['matcher']
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Successfully loaded model weights from {model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load model weights: {e}")
        return None
    
    val_loss, val_cls_loss, val_reg_loss = validate_one_epoch(
        model, val_loader, device, focal_criterion, box_coder, matcher
    )
    
    print(f"[EVALUATION] Validation Loss: {val_loss:.4f}")
    print(f"[EVALUATION] Classification Loss: {val_cls_loss:.4f}")
    print(f"[EVALUATION] Regression Loss: {val_reg_loss:.4f}")
    
    return {
        'val_loss': val_loss,
        'val_cls_loss': val_cls_loss,
        'val_reg_loss': val_reg_loss
    }
