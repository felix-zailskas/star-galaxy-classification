import torch
import torch.nn as nn


def GradCAM(output, feature_maps):
    # Compute gradients of the output with respect to the intermediate feature maps
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=feature_maps,
        grad_outputs=torch.ones_like(output),
        retain_graph=True,
    )

    # Compute the importance weights using global average pooling of the gradients
    importance_weights = torch.mean(
        torch.cat(
            [
                grad.view(grad.size(0), -1).mean(dim=1, keepdim=True)
                for grad in gradients
            ]
        ),
        dim=0,
        keepdim=True,
    )

    # Compute the Grad-CAM heatmaps by weighting the feature maps with the importance weights
    gradcam_heatmaps = torch.sum(importance_weights * feature_maps, dim=1, keepdim=True)
    gradcam_heatmaps = nn.functional.relu(
        gradcam_heatmaps
    )  # Apply ReLU to remove negative values

    # Normalize the Grad-CAM heatmaps
    gradcam_heatmaps_normalized = (gradcam_heatmaps - gradcam_heatmaps.min()) / (
        gradcam_heatmaps.max() - gradcam_heatmaps.min() + 1e-9
    )

    return gradcam_heatmaps_normalized
