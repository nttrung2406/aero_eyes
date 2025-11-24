import torch
from tqdm import tqdm
import os

from setup import setup_training_components
from compute_loss import compute_ssd_loss
from validate import validate_one_epoch


def train_one_epoch(model, train_loader, optimizer, device, epoch, focal_criterion, box_coder, matcher, log_interval=1):
    """
    Performs one full training pass over the training data.
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [T]")
    
    for i, batch in enumerate(progress_bar):
        frames = batch["frame"].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch["targets"]]
        ref_images = batch["ref_images"].to(device)
        
        cls_logits, bbox_regression, anchors = model(frames, ref_images=ref_images)
        
        if not cls_logits:
            print("[WARNING] Model forward pass returned empty lists. Skipping batch.")
            continue
            
        loss_dict = compute_ssd_loss(
            cls_logits, bbox_regression, anchors,
            targets, focal_criterion, box_coder, matcher
        )
        
        loss_cls = loss_dict["loss_focal_cls"]
        loss_reg = loss_dict["loss_box_reg"]
        loss = loss_cls + loss_reg
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_reg_loss += loss_reg.item()
        
        if (i + 1) % log_interval == 0:
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Cls': f'{loss_cls.item():.4f}',
                'Reg': f'{loss_reg.item():.4f}'
            })
            
    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_reg_loss = total_reg_loss / len(train_loader)
    
    return avg_loss, avg_cls_loss, avg_reg_loss


def train_model(num_epochs=200, model_save_path=None, use_kaggle=True):
    """Main training function"""
    components = setup_training_components(use_kaggle=use_kaggle)
    
    model = components['model']
    train_loader = components['train_loader']
    val_loader = components['val_loader']
    optimizer = components['optimizer']
    scheduler = components['scheduler']
    device = components['device']
    focal_criterion = components['focal_criterion']
    box_coder = components['box_coder']
    matcher = components['matcher']
    
    if model_save_path is None:
        model_save_path = "/kaggle/working/best_ssd_model.pth" if use_kaggle else "./best_ssd_model.pth"
    
    best_val_loss = float('inf')
    log_interval = 1
    
    if os.path.exists(model_save_path):
        print(f"[INFO] Found existing model weights at '{model_save_path}'.")
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            print("[INFO] Successfully loaded model weights. Starting training.")
        except Exception as e:
            print(f"[ERROR] Could not load weights. Starting from scratch. Error: {e}")
    else:
        print("[INFO] No weights found. Starting fresh training.")

    print("[TRAINING] Starting training process...")
    for epoch in range(num_epochs):
        train_loss, train_cls_loss, train_reg_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, 
            focal_criterion, box_coder, matcher, log_interval
        )
        
        val_loss, val_cls_loss, val_reg_loss = validate_one_epoch(
            model, val_loader, device, focal_criterion, box_coder, matcher
        )
        
        scheduler.step()
        
        print(
            f"\nEpoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Reg: {train_reg_loss:.4f}) | "
            f"Val Loss: {val_loss:.4f} (Cls: {val_cls_loss:.4f}, Reg: {val_reg_loss:.4f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            torch.save(model.state_dict(), model_save_path)
            
            print(f"-> Best model saved to {model_save_path} with validation loss: {val_loss:.4f}\n")

    print("[TRAINING] Training complete.")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    print(f"Final model saved at: {model_save_path}")
    
    return model, best_val_loss
