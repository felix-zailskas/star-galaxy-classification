# Importing the required libraries
import cv2
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


# Function to combine spectral bands
def combine_spectral_bands(file_paths):
    # Reading in the first file to get the reference WCS
    first_file = fits.open(file_paths[0])
    wcs_ref = WCS(first_file[0].header)

    # Initializing the combined data array
    combined_data = np.zeros_like(first_file[0].data)

    # Combining the spectral bands
    for file_path in file_paths[1:]:
        # Reading in the fits file
        hdulist = fits.open(file_path)
        data = hdulist[0].data

        # Verifying the shape of the data
        if data.shape != combined_data.shape:
            raise ValueError(
                f"The shape of {file_path} does not match the previous files."
            )
        # check difference to reference and shift image
        wcs = WCS(hdulist[0].header)
        x, y = wcs_ref.world_to_pixel(
            wcs.pixel_to_world(wcs.wcs.crpix[0], wcs.wcs.crpix[1])
        )
        print(x, y)
        print(wcs.wcs.crpix[0], wcs.wcs.crpix[1])
        x_shift = x - wcs.wcs.crpix[0]
        y_shift = y - wcs.wcs.crpix[1]
        print(x_shift, y_shift)

        M = np.array([[1.0, 0.0, x_shift], [0.0, 1.0, y_shift]])

        shifted_data = cv2.warpAffine(
            data.astype(np.float32), M, (data.shape[1], data.shape[0])
        )
        # Adding the data to the combined array
        combined_data += shifted_data

    # Creating a new fits file to save the combined data
    combined_hdulist = fits.HDUList(
        [fits.PrimaryHDU(combined_data, header=wcs.to_header())]
    )
    combined_hdulist.writeto("../data/combined.fits", overwrite=True)

    print("Spectral bands combined and saved as 'combined.fits'.")


# Provide the file paths of the spectral band files
file_paths = [
    "../data/frame-i-008162-6-0080.fits.bz2",
    "../data/frame-r-008162-6-0080.fits.bz2",
    "../data/frame-g-008162-6-0080.fits.bz2",
]  # , '../data/frame-u-008162-6-0080.fits.bz2','../data/frame-z-008162-6-0080.fits.bz2']


# Combine the spectral bands
combine_spectral_bands(file_paths)
