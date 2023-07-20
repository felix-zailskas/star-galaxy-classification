# Visualize the Grad-CAM heatmaps as an overlay on the input image
import matplotlib.pyplot as plt
import numpy as np

input_image_np = input_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
gradcam_heatmaps_np = gradcam_heatmaps_normalized.squeeze().detach().cpu().numpy()

plt.imshow(input_image_np)
plt.imshow(gradcam_heatmaps_np, cmap="jet", alpha=0.5)
plt.axis("off")
plt.show()
