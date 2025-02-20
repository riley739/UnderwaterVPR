import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

def visualize_feature_maps(feature_map, original_image):

    # Aggregate across channels (mean)
    aggregated_map = feature_map.mean(dim=0)  # Shape: (H, W)
    
    # Normalize for visualization
    aggregated_map -= aggregated_map.min()
    aggregated_map /= aggregated_map.max()
    
    # Upsample to original image size
    # upsampled_map = resize(aggregated_map.unsqueeze(0), [480,640])
    
    # Plot
    plt.figure(figsize=(10, 5))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.axis('off')
    plt.title('Original Image')
    torch.nn.functional.interpolate
    # Heatmap
    plt.subplot(1, 2, 2)
    # plt.imshow(original_image)
    plt.imshow(aggregated_map.squeeze().cpu().numpy(), cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title('Feature Map Heatmap')
    
    plt.show()
    # plt.waitforbuttonpress()


def visualize_cls_attention(attention_weights, original_image):
    """
    Visualize the CLS token's attention as a heatmap.

    Args:
    - attention_weights: Tensor of shape (H, W), normalized attention from CLS token.
    - original_image: Original image for reference (H_in, W_in, 3).
    """
    # Normalize attention weights
    attention_weights -= attention_weights.min()
    attention_weights /= attention_weights.max()

    # # Upsample to original image size
    # upsampled_attention = resize(
    #     attention_weights.unsqueeze(0).unsqueeze(0),  # Shape (1, 1, H, W)
    #     [480,640],
    #     interpolation=InterpolationMode.BILINEAR
    # ).squeeze().cpu().numpy()

    # Plot
    plt.figure(figsize=(10, 5))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.axis('off')
    plt.title('Original Image')
    
    # Heatmap
    plt.subplot(1, 2, 2)
    # plt.imshow(original_image)
    plt.imshow(attention_weights, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title('CLS Token Heatmap')
    
    plt.show()
