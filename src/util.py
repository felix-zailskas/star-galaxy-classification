import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt


def display_fits_file(path: str):
    fits_file = fits.open(path)
    image = fits_file[0].data
    plt.axis("off")
    plt.imshow(
        image, cmap="gray", vmin=np.percentile(image, 5), vmax=np.percentile(image, 95)
    )
