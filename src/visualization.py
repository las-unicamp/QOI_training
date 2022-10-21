import numpy as np
import matplotlib.pyplot as plt


def visualize(transpose_channels=False, figsize=(30, 30), **images):
    """
    Helper function for data visualization
    PyTorch CHW tensor will be converted to HWC if `transpose_channels=True`
    """
    n_images = len(images)

    plt.figure(figsize=figsize)
    for idx, (key, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.title(key.replace("_", " ").title(), fontsize=12)
        if transpose_channels:
            plt.imshow(np.transpose(image, (1, 2, 0)))
        else:
            plt.imshow(image)
        plt.axis("off")
    plt.show()
