import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large


class OptimizedConditionedSSD(nn.Module):
    """
    Final corrected MobileNetV3-based model.
    The feature indices and dimensions are now definitively aligned with the torchvision
    MobileNetV3-Large architecture, resolving the channel mismatch error.
    """
    def __init__(self, embed_dim=96, cond_scale=1.0, device="cuda", num_feature_levels=3):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.cond_scale = cond_scale
        self.num_feature_levels = num_feature_levels
        self.num_anchors_per_location = 6 
        
        backbone = mobilenet_v3_large(weights='DEFAULT')
        self.backbone = backbone.features

        self.feature_indices = [3, 6, 16]
        mobilenet_dims = [24, 40, 960]
        
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, embed_dim, kernel_size=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            ) for dim in mobilenet_dims
        ])

        self.ref_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_feature_levels)
        ])
        
        self.query_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_feature_levels)
        ])
        
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True, dropout=0.1)
            for _ in range(num_feature_levels)
        ])
        
        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, self.num_anchors_per_location * 1, kernel_size=1)
            ) for _ in range(num_feature_levels)
        ])
        
        self.box_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, self.num_anchors_per_location * 4, kernel_size=1)
            ) for _ in range(num_feature_levels)
        ])
        
        self.ref_embeddings = nn.ParameterList([
            nn.Parameter(torch.zeros(1, embed_dim, 1, 1))
            for _ in range(num_feature_levels)
        ])
        self.ref_embedding_valid = False

        
    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.feature_indices:
                features.append(x)
        return features
        

    def set_reference_cache(self, ref_imgs):
        with torch.no_grad():
            B, R, C, H, W = ref_imgs.shape
            ref_embeddings_accumulated = [[] for _ in range(self.num_feature_levels)]
            
            for r in range(R):
                single_ref = ref_imgs[:, r]
                ref_features = self.extract_features(single_ref)
                
                if len(ref_features) == self.num_feature_levels:
                    for level, (feat, extractor) in enumerate(zip(ref_features, self.feature_extractors)):
                        ref_embed = extractor(feat)
                        ref_embed = F.adaptive_avg_pool2d(ref_embed, 1) 
                        ref_embeddings_accumulated[level].append(ref_embed)
            
            for level in range(self.num_feature_levels):
                if ref_embeddings_accumulated[level]:
                    stacked_embeds = torch.stack(ref_embeddings_accumulated[level])
                    mean_embed = stacked_embeds.mean(dim=[0, 1])
                    
                    self.ref_embeddings[level].data = mean_embed.unsqueeze(0)
            
            self.ref_embedding_valid = True
            
    def forward(self, x, ref_images=None):
        if ref_images is not None:
            self.set_reference_cache(ref_images)
        
        features = self.extract_features(x)
        
        if len(features) != self.num_feature_levels:
            print(f"[ERROR] Expected {self.num_feature_levels} features, got {len(features)}")
            return [], [], []
        
        cls_logits_list = []
        bbox_regression_list = []
        anchors_list = []
        anchor_config = {
            "scales": torch.tensor([0.1, 0.2], device=x.device), 
            "ratios": torch.tensor([0.5, 1.0, 2.0], device=x.device)
        }
        num_anchors_per_location = len(anchor_config["scales"]) * len(anchor_config["ratios"])
        
        for level, feat in enumerate(features):
            if level >= len(self.feature_extractors):
                print(f"[ERROR] Feature extractor {level} not available")
                continue
                
            query_feat = self.feature_extractors[level](feat)
            
            if self.ref_embedding_valid:
                ref_feat_expanded = self.ref_embeddings[level].expand_as(query_feat)
                cond_feats = query_feat + self.cond_scale * ref_feat_expanded
            else:
                cond_feats = query_feat

            cls_logits = self.cls_heads[level](cond_feats)
            bbox_regression = self.box_heads[level](cond_feats)
            
            B, _, H, W = feat.shape
            stride = 320 / H 
        
            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij")
            center_x = (grid_x + 0.5) * stride
            center_y = (grid_y + 0.5) * stride
            centers = torch.stack([center_x, center_y, center_x, center_y], dim=-1).view(-1, 4)
        
            level_anchors = []
            for scale in anchor_config["scales"]:
                for ratio in anchor_config["ratios"]:
                    base_size = 320 * scale
                    w = base_size * torch.sqrt(ratio)
                    h = base_size / torch.sqrt(ratio)
                    
                    boxes = torch.cat([
                        centers[:, :2],
                        torch.full_like(centers[:, :2], w)
                    ], dim=1)
                    boxes[:, 3] = h
        
                    level_anchors.append(boxes)
            
            anchors_level = torch.cat(level_anchors, dim=0)
            anchors_level[:, :2] -= anchors_level[:, 2:] / 2
            anchors_level[:, 2:] += anchors_level[:, :2]
        
            cls_logits_list.append(cls_logits)
            bbox_regression_list.append(bbox_regression)
            anchors_list.append(anchors_level)
        
        return cls_logits_list, bbox_regression_list, anchors_list