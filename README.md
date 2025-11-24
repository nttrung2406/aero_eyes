# Zalo AI Challenge - AeroEyes 

## Model Architecture Diagram

```
Input Image (320x320)           Reference Images (Nx320x320)
       │                               │
       ▼                               ▼
┌─────────────────┐              ┌─────────────────┐
│  MobileNetV3    │              │  MobileNetV3    │
│    Backbone     │              │    Backbone     │
└─────────────────┘              └─────────────────┘
       │                               │
    Features                       Features
  [Layer 3: 24ch]                [Layer 3: 24ch]
  [Layer 6: 40ch]                [Layer 6: 40ch]  
  [Layer 16: 960ch]              [Layer 16: 960ch]
       │                               │
       ▼                               ▼
┌─────────────────┐              ┌─────────────────┐
│ Feature Extract │              │ Feature Extract │
│   (1x1 Conv)    │              │   (1x1 Conv)    │
│  → 96 channels  │              │  → 96 channels  │
└─────────────────┘              └─────────────────┘
       │                               │
       ▼                               ▼
┌─────────────────┐              ┌─────────────────┐
│ Query Features  │              │Reference Cache  │
│                 │              │(Global AvgPool) │
└─────────────────┘              └─────────────────┘
       │                               │
       └───────────┐           ┌───────┘
                   ▼           ▼
            ┌─────────────────────┐
            │   Feature Fusion    │
            │ Query + α × Ref     │
            └─────────────────────┘
                       │
                ┌──────┴──────┐
                ▼             ▼
        ┌──────────────┐ ┌──────────────┐
        │ Class Head   │ │  Box Head    │
        │(Conv layers) │ │(Conv layers) │
        │   6 × 1      │ │   6 × 4      │
        └──────────────┘ └──────────────┘
                ▼             ▼
          [Objectness]   [Box Offsets]
```

## Project Overview
This repository implements a few-shot object detection pipeline using a custom SSD (Single Shot Detector) architecture conditioned on reference images. The solution is designed for scenarios where only a few annotated examples are available for each object class, leveraging reference images to improve detection performance.

### Key Features
- MobileNetV3 Backbone
- Reference Conditioning
- Custom SSD Head
- Focal Loss for class imbalance
- Flexible Dataset Loader

## File Structure
- `dataclass.py`: Dataset class and collate function
- `model.py`: Model architecture
- `loss.py`: Loss functions and matcher
- `compute_loss.py`: SSD loss computation
- `setup.py`: Setup utilities
- `train.py`: Training loop
- `validate.py`: Validation utilities
- `docker-compose.train.yml`: Docker Compose for training
- `docker-compose.validate.yml`: Docker Compose for validation

## Model Architecture Analysis

The model implements a novel few-shot object detection architecture that conditions detection on reference images. Here's the detailed breakdown:

#### 1. Backbone Architecture
- **Base**: MobileNetV3-Large pretrained on ImageNet
- **Feature Extraction**: Multi-scale features from layers 3, 6, and 16
  - Layer 3: 24 channels (high resolution, fine details)
  - Layer 6: 40 channels (medium resolution, semantic features)
  - Layer 16: 960 channels (low resolution, deep semantics)

#### 2. Feature Processing Pipeline
```python
# Each feature level is processed through:
Conv2d(input_dim, 96, kernel_size=1) → BatchNorm2d → ReLU
```
This creates a uniform 96-dimensional embedding space across all feature levels.

#### 3. Reference Conditioning Mechanism
The model's key innovation is its reference conditioning system:

1. **Reference Encoding**: Reference images are processed through the same backbone
2. **Global Pooling**: Reference features are pooled to create compact embeddings
3. **Caching**: Reference embeddings are cached per feature level for efficiency
4. **Fusion**: Query features are combined with reference embeddings:
   ```python
   conditioned_features = query_features + α × reference_embeddings
   ```

#### 4. Detection Heads
- **Classification Head**: Predicts objectness (6 anchors × 1 class per location)
- **Regression Head**: Predicts bounding box offsets (6 anchors × 4 coordinates)
- Both heads use 3×3 convolutions followed by 1×1 output layers

#### 5. Anchor Generation
- **Multi-scale**: 2 scales per feature level [0.1, 0.2]
- **Multi-aspect**: 3 aspect ratios [0.5, 1.0, 2.0]
- **Total**: 6 anchors per spatial location
- **Grid-based**: Anchors centered on regular grid with appropriate strides

## Loss Function Analysis

The training uses a sophisticated multi-component loss system designed from scratch:

#### 1. Focal Loss for Classification
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        # Addresses extreme class imbalance in object detection
```

**Key Features**:
- **Alpha weighting**: Balances positive/negative examples (α=0.25)
- **Gamma focusing**: Down-weights easy examples (γ=2.0)
- **Robust clamping**: Prevents numerical instability
- **NaN protection**: Replaces invalid values with zeros

**Mathematical Formulation**:
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

#### 2. Anchor Matching Strategy (SSDMatcher)
- **Positive threshold**: IoU > 0.4 with ground truth
- **Negative threshold**: IoU < 0.3 with all ground truth boxes
- **Ignore region**: 0.3 ≤ IoU < 0.4 (not used in loss computation)
- **Dynamic matching**: Handles varying numbers of objects per image

#### 3. Box Regression Encoding (SSDBoxCoder)
Transforms absolute coordinates to relative offsets for stable training:

```python
dx = 10.0 × (gt_center_x - anchor_center_x) / anchor_width
dy = 10.0 × (gt_center_y - anchor_center_y) / anchor_height  
dw = 5.0 × log(gt_width / anchor_width)
dh = 5.0 × log(gt_height / anchor_height)
```

#### 4. Multi-Level Loss Computation
The loss computation (`compute_ssd_loss`) handles:
- **Batch processing**: Efficient computation across multiple images
- **Multi-scale fusion**: Combines predictions from all feature levels
- **Anchor reshaping**: Flattens spatial dimensions for matching
- **Positive normalization**: Regression loss normalized by positive anchor count
- **Error handling**: Graceful handling of edge cases (no GT, no positives)

#### 5. Loss Components Integration
Final loss combines both components:
```python
total_loss = focal_classification_loss + smooth_l1_regression_loss
```

### Key Innovations

1. **Reference Conditioning**: Unlike standard SSD, this model adapts to new object classes through reference images
2. **Robust Loss Design**: Custom focal loss with extensive error handling for few-shot scenarios
3. **Multi-scale Architecture**: Leverages MobileNet's efficiency with multi-level feature extraction
4. **Dynamic Anchor Matching**: Flexible matching strategy that works with varying object densities


## Docker Compose Usage

### Training Stage
To run training in a Docker container with GPU support:

```bash
docker compose -f docker-compose.train.yml up --build
```

This will start the training process using the configuration in `train.py`.

### Validation Stage
To run validation in a Docker container with GPU support:

```bash
docker compose -f docker-compose.validate.yml up --build
```

This will evaluate the model using the configuration in `validate.py`.

#### Notes
- Both compose files use the official PyTorch GPU image.
- The project directory is mounted into the container at `/workspace`.
- Make sure your data paths in `setup.py` are correct and accessible inside the container.
- NVIDIA GPU is required for CUDA support.

## How to Use
1. Edit `setup.py` to configure your dataset paths.
2. Use the Docker Compose commands above for training and validation.
3. Model weights are saved in the project directory.

