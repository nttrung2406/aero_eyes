import torch
from torch.utils.data import random_split, DataLoader
import os

from dataclass import YoloDetectionDataset, collate_fn, get_transform
from model import OptimizedConditionedSSD
from loss import get_loss_components


def setup_environment():
    """Setup device and optimizations"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using device: {device}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    return device


def setup_paths(use_kaggle=True):
    """Setup dataset paths"""

    train_img_dir = "train_path/images"
    train_lbl_dir = "train_path/labels"
    ref_root = "train_path/samples"
    
    if not os.path.exists(train_img_dir):
        print(f"[ERROR] Images directory not found: {train_img_dir}")
    if not os.path.exists(train_lbl_dir):
        print(f"[ERROR] Labels directory not found: {train_lbl_dir}")
    if not os.path.exists(ref_root):
        print(f"[ERROR] Reference directory not found: {ref_root}")
    
    return train_img_dir, train_lbl_dir, ref_root


def setup_dataset(train_img_dir, train_lbl_dir, ref_root, batch_size=4, num_workers=4):
    """Setup dataset and dataloaders"""
    transform = get_transform()
    
    print("[DATASET] Setting up dataset...")
    try:
        dataset = YoloDetectionDataset(
            img_dir=train_img_dir,
            lbl_dir=train_lbl_dir,
            ref_root=ref_root,
            transform=transform,
            max_refs=3
        )
        if len(dataset) == 0:
            raise ValueError("Dataset is empty - check paths!")
        
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            collate_fn=collate_fn, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=collate_fn, pin_memory=True
        )
        print(f"[DATASET] Loaded: {len(train_dataset)} train, {len(val_dataset)} val images")

        return train_loader, val_loader

    except Exception as e:
        print(f"[ERROR] Failed to create dataset: {e}")
        raise e


def setup_model(device):
    """Initialize the model"""
    print("[MODEL] Initializing the corrected model...")
    try:
        model = OptimizedConditionedSSD(
            embed_dim=96,
            cond_scale=1.0, 
            device=device,
            num_feature_levels=3
        ).to(device)
        
        print("[MODEL] Model initialized successfully.")
        return model
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize model: {e}")
        raise e


def setup_optimizer_scheduler(model):
    """Initialize optimizer and scheduler"""
    print("[OPTIMIZER] Initializing optimizer and scheduler...")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=2e-4,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    return optimizer, scheduler


def setup_training_components(use_kaggle=True, batch_size=4, num_workers=4):
    """Setup all training components"""
    device = setup_environment()
    
    train_img_dir, train_lbl_dir, ref_root = setup_paths(use_kaggle)
    
    train_loader, val_loader = setup_dataset(
        train_img_dir, train_lbl_dir, ref_root, batch_size, num_workers
    )
    
    model = setup_model(device)
    
    optimizer, scheduler = setup_optimizer_scheduler(model)
    
    matcher, box_coder, focal_criterion = get_loss_components()
    
    print("[SETUP] All components are ready for training.")
    
    return {
        'device': device,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'matcher': matcher,
        'box_coder': box_coder,
        'focal_criterion': focal_criterion
    }
