# Importing the required libraries
from typing import List, Tuple

import cv2
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS


def combine_spectral_bands(file_paths: List[str], save_dir: str) -> None:
    """Aligns spectral bands provided as paths to this function. The first file in the provided list
    will be used as the reference image. Then all bands are added on top of each other to create a
    single gray-scale image. This image is saved to the provided file path.

    Args:
        file_paths (List[str]): Paths to the fits files containing the spectral bands of one image.
        save_dir (str): Path to where the file should be saved. Should end in `.fits`.

    Raises:
        ValueError: Raised when `save_dir` does not end in `.fits`.
        ValueError: Raised when the provided spectral bands have different shapes.
    """
    if not save_dir.endswith(".fits"):
        raise ValueError(
            f"File {save_dir} is not a .fits file. Function could not save output."
        )

    # Reading in the first file to get the reference WCS
    first_file = fits.open(file_paths[0])
    wcs_ref = WCS(first_file[0].header)
    # Initializing the combined data array
    combined_data = np.zeros_like(first_file[0].data)
    first_file.close()

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
        x_shift = x - wcs.wcs.crpix[0]
        y_shift = y - wcs.wcs.crpix[1]

        M = np.array([[1.0, 0.0, x_shift], [0.0, 1.0, y_shift]])

        shifted_data = cv2.warpAffine(
            data.astype(np.float32), M, (data.shape[1], data.shape[0])
        )
        # Adding the data to the combined array
        combined_data += shifted_data
        hdulist.close()

    # Creating a new fits file to save the combined data
    combined_hdulist = fits.HDUList(
        [fits.PrimaryHDU(combined_data, header=wcs.to_header())]
    )
    combined_hdulist.writeto(save_dir, overwrite=True)


def align_image_channels(
    file_paths: List[str],
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Aligns spectral bands provided as paths to this function. The first file in the provided list
    will be used as the reference image. The regions at the border of the image that becomes invalid
    due to the shift performed is cut off from the image.

    Args:
        file_paths (List[str]): Paths to the fits files containing the spectral bands of one image.

    Returns:
        Tuple[np.ndarray, Tuple[int, int, int, int]]:
            The first item in the Tuple contains a numpy array of shape (H, W, C) representing the aligned image.
                H = height of the image
                W = width of the image
                C = channels in the image
            The second item in the Tuple contains a Tuple with 4 positive integers representing the following values in order:
                1. The amount of pixels cut off from the left side of the image.
                2. The amount of pixels cut off from the right side of the image.
                3. The amount of pixels cut off from the top side of the image.
                4. The amount of pixels cut off from the bottom side of the image.
    """
    # Reading in the first file to get the reference WCS
    first_file = fits.open(file_paths[0])
    wcs_ref = WCS(first_file[0].header)

    # Initializing the channels array
    all_channels = []

    max_xshift = -np.inf
    max_yshift = -np.inf
    min_xshift = np.inf
    min_yshift = np.inf
    # Processing the spectral bands
    for file_path in file_paths:
        # Reading in the fits file
        hdulist = fits.open(file_path)
        data = hdulist[0].data

        # check difference to reference and shift image
        wcs = WCS(hdulist[0].header)
        x, y = wcs_ref.world_to_pixel(
            wcs.pixel_to_world(wcs.wcs.crpix[0], wcs.wcs.crpix[1])
        )
        x_shift = x - wcs.wcs.crpix[0]
        y_shift = y - wcs.wcs.crpix[1]

        M = np.array([[1.0, 0.0, x_shift], [0.0, 1.0, y_shift]])

        shifted_data = cv2.warpAffine(
            data.astype(np.float64), M, (data.shape[1], data.shape[0])
        )

        # save max shifts for data augmentation
        if x_shift < 0 and x_shift < min_xshift:
            min_xshift = np.floor(x_shift).astype(np.int64)
        if x_shift > 0 and x_shift > max_xshift:
            max_xshift = np.ceil(x_shift).astype(np.int64)
        if y_shift < 0 and y_shift < min_yshift:
            min_yshift = np.floor(y_shift).astype(np.int64)
        if y_shift > 0 and y_shift > max_yshift:
            max_yshift = np.ceil(y_shift).astype(np.int64)

        # Adding the data to the combined array
        all_channels.append(shifted_data)

    # cutting all channels to same dimensions
    for i, channel in enumerate(all_channels):
        all_channels[i] = channel[max_yshift:min_yshift, max_xshift:min_xshift]

    return np.stack(all_channels, axis=-1), (
        abs(max_xshift),
        abs(min_xshift),
        abs(max_yshift),
        abs(min_yshift),
    )


def extract_calibobj_coordinates(
    obj_path: str, reference_frame_path: str, offsets: Tuple[int, int, int, int]
) -> np.ndarray:
    """Extracts x and y pixel coordinates contained in a provided calibObj file in relation to a reference frame.

    Args:
        obj_path (str): Path to the calibObj file that should be used.
        reference_frame_path (str): Path to the frame used as a reference.
        offsets (Tuple[int, int, int, int]): Cutoffs applied to the reference image as provided by the `align_image_channels` function.

    Returns:
        np.ndarray: A numpy array of shape (N, 2) containing the pixel coordinates of the objects in the form [x, y].
            N = number of objects contained in the image.
    """
    # get WCS and shift values of the reference image
    ref_frame = fits.open(reference_frame_path)
    ref_wcs = WCS(ref_frame[0].header)
    (
        left_x_shift,
        right_x_shift,
        left_y_shift,
        right_y_shift,
    ) = offsets
    ref_width = ref_frame[0].data.shape[0] - left_x_shift - right_x_shift
    ref_height = ref_frame[0].data.shape[1] - left_y_shift - right_y_shift

    # open data of calibObj file
    obj_data = fits.open(obj_path)
    obj_table = Table(obj_data[1].data)
    # Extract sky coordinates and translate to pixels in reference image
    sky_coords = SkyCoord(ra=obj_table["RA"], dec=obj_table["DEC"], unit="deg")
    pixel_coords = np.array(ref_wcs.world_to_pixel(sky_coords)).T
    # get all objects within the image bounds
    x_indices = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < ref_height)
    y_indices = (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < ref_width)
    # finally shift pixel in accordance with the cutoff of the previously aligned images
    return pixel_coords[x_indices & y_indices] - np.array([left_x_shift, left_y_shift])


def align_channels_stars_galaxies(
    file_paths: List[str], star_path: str, galaxy_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    aligned_channels, offsets = align_image_channels(file_paths)
    star_coords = extract_calibobj_coordinates(star_path, file_paths[0], offsets)
    galaxy_coords = extract_calibobj_coordinates(galaxy_path, file_paths[0], offsets)
    return (aligned_channels, star_coords, galaxy_coords)
