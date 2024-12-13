# MobileSAM: Mobile-Friendly Segment Anything Model

## Quick Overview
MobileSAM revolutionizes on-device visual understanding by shrinking SAM's size by 60x while maintaining original performance through innovative decoupled distillation. Achieves 12ms total inference time (8ms image encoder + 4ms mask decoder) on GPU, making real-time segmentation possible on mobile devices.

## Technical Deep-Dive

### Core Innovation
The breakthrough lies in solving coupled optimization between image encoder and mask decoder through decoupled distillation:
1. Replace heavy ViT-H encoder with lightweight alternative
2. Distill knowledge directly from original SAM's image encoder
3. Maintain compatibility with original mask decoder architecture

### Performance Metrics
- Model Size: 60x reduction (compared to original SAM)
- Inference Speed: ~10ms per image on GPU
- Training Efficiency: Single GPU, <24 hours
- CPU Performance: Smooth operation demonstrated

### Implementation
```python
# Basic usage example
from mobile_sam import sam_model_registry, SamPredictor

def initialize_mobile_sam():
    model_type = "vit_t"
    sam = sam_model_registry[model_type](checkpoint="mobile_sam.pth")
    return SamPredictor(sam)

def segment_image(predictor, image, box=None):
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=None, 
        point_labels=None, 
        box=box
    )
    return masks

# Advanced usage with custom parameters
def segment_with_points(predictor, image, points, labels):
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
        return_logits=True
    )
    return masks, scores, logits
