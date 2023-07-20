import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt


def check_sky_overlap(
    sky_corners1: np.ndarray, sky_corners2: np.ndarray, wcs: WCS
) -> bool:
    """Checks whether two sets of sky coordinates overlap with each other. The sky coordinates
    should be provided as a set of four points as follows:
        [
            [MAX RA, CORRESPONDING DEC],
            [MIN RA, CORRESPONDING DEC],
            [CORRESPONDING RA, MAX DEC],
            [CORRESPONDING RA, MIN DEC]
        ]

    The function translates them into pixel coordinates based on the provided world coordinate
    system, and then checks if the rectangles overlap in pixel space.

    Args:
        sky_corners1 (np.ndarray): First set of sky coordinates.
        sky_corners2 (np.ndarray): Second set of sky coordinates.
        wcs (WCS): World coordinate system for pixel translation.

    Returns:
        bool: True if the coordinates overlap, False otherwise.
    """
    # convert sky coordinates to pixel coordinates for easier processing of the overlap
    pixel_corners1 = np.array(wcs.world_to_pixel_values(sky_corners1), dtype=int)
    pixel_corners2 = np.array(wcs.world_to_pixel_values(sky_corners2), dtype=int)

    x_min1, x_max1 = np.min(pixel_corners1[:, 0]), np.max(pixel_corners1[:, 0])
    y_min1, y_max1 = np.min(pixel_corners1[:, 1]), np.max(pixel_corners1[:, 1])

    x_min2, x_max2 = np.min(pixel_corners2[:, 0]), np.max(pixel_corners2[:, 0])
    y_min2, y_max2 = np.min(pixel_corners2[:, 1]), np.max(pixel_corners2[:, 1])

    # standard rectangle overlap check
    if x_max1 < x_min2 or x_min1 > x_max2 or y_max1 < y_min2 or y_min1 > y_max2:
        return False
    return True


def get_sky_corners(
    image: str,
    pixel_coords: Tuple[int, int, int, int] = None,
    header: fits.Header = None,
) -> np.ndarray:
    """
    Retrieves the sky coordinates of the corners of an image. These are the maximum and minimum values
    for RA and DEC.

    Args:
        image (str): Path to the FITS image file.
        pixel_coords (Tuple[int, int, int, int], optional): Tuple representing the pixel coordinates
            (min_x, max_x, min_y, max_y) of the sub-image. If None, the entire image will be used.
            Defaults to None.
        header (fits.Header, optional): FITS header object. If None, the header will be read from the
            FITS file specified by `image`. Defaults to None.

    Returns:
        np.ndarray: Array of shape (4, 2) containing the sky coordinates of the corners. Each row
            represents a corner and has the format [RA, DEC].

    Raises:
        ValueError: If the `image` path is invalid or the FITS file cannot be opened.

    """
    if header is None:
        hdulist = fits.open(image)
        header = hdulist[0].header
    wcs = WCS(header)

    if pixel_coords is None:
        pixel_coords = (0, header["NAXIS2"] - 1, 0, header["NAXIS1"] - 1)
    min_x, max_x, min_y, max_y = pixel_coords

    y_range = np.arange(min_y, max_y)
    x_range = np.arange(min_x, max_x)

    # get all pixel coordinates of the edge of the image
    pixels_left = np.column_stack((np.tile(min_x, max_y - min_y), y_range))
    pixels_top = np.column_stack((np.tile(min_y, max_x - min_x), x_range))
    pixels_right = np.column_stack((np.tile(max_x, max_y - min_y), y_range))
    pixels_bottom = np.column_stack((np.tile(max_y, max_x - min_x), x_range))
    pixel_edges = np.concatenate(
        (pixels_left, pixels_right, pixels_top, pixels_bottom), axis=0
    )

    # convert to sky coordinates
    sky_edges = np.array(wcs.all_pix2world(pixel_edges[:, 0], pixel_edges[:, 1], 0))

    # find indices of minimum and maximum RA-coordinates
    min_x_index = np.argmin(sky_edges[0])
    max_x_index = np.argmax(sky_edges[0])

    # find indices of minimum and maximum DEC-coordinates
    min_y_index = np.argmin(sky_edges[1])
    max_y_index = np.argmax(sky_edges[1])

    # create the bounding box of the sky coordinates
    sky_edges = np.array(
        [
            [sky_edges[0, min_x_index], sky_edges[1, min_x_index]],
            [sky_edges[0, max_x_index], sky_edges[1, max_x_index]],
            [sky_edges[0, min_y_index], sky_edges[1, min_y_index]],
            [sky_edges[0, max_y_index], sky_edges[1, max_y_index]],
        ]
    )
    return sky_edges


def create_image_patches(
    image: np.ndarray,
    stars: np.ndarray,
    galaxies: np.ndarray,
    patch_size: Tuple[int, int],
    reference_file: str,
    exclude_file: str = None,
):
    """
    Create patches of given size from an image, filtered by sky coordinates and excluding specific regions.

    Args:
        image (np.ndarray): Input image array of shape (height, width, channels).
        stars (np.ndarray): Array of star coordinates of shape (num_stars, 2).
        galaxies (np.ndarray): Array of galaxy coordinates of shape (num_galaxies, 2).
        patch_size (Tuple[int, int]): Size of the patches to extract (new_height, new_width).
        reference_file (str): Path to the reference FITS file for extracting sky coordinates.
        exclude_file (str, optional): Path to the FITS file containing regions to exclude.
            Defaults to None.

    Returns:
        List[Tuple[np.ndarray, np.ndarray, np.ndarray]]: A list of tuples, where each tuple contains:
            - A patch from the input image array of shape (new_height, new_width, channels).
            - An array of star coordinates within the patch, adjusted by the patch position,
              of shape (num_patch_stars, 2).
            - An array of galaxy coordinates within the patch, adjusted by the patch position,
              of shape (num_patch_galaxies, 2).

    Raises:
        ValueError: If the input image array shape is invalid or incompatible with the patch size.
        ValueError: If the star or galaxy coordinate arrays are invalid.
        ValueError: If the reference FITS file cannot be opened or its header is invalid.

    """
    old_height, old_width, channels = image.shape
    new_height, new_width = patch_size

    # prepare data needed if specific coordinates need to be excluded
    if exclude_file is not None:
        hdulist = fits.open(exclude_file)
        exclude_header = hdulist[0].header
        exclude_wcs = WCS(exclude_header)
        exclude_sky_corners = get_sky_corners(exclude_file, header=exclude_header)
        reference_header = fits.open(reference_file)[0].header

    all_patches = []

    for x in range(0, old_width - new_width, new_width):
        for y in range(0, old_height - new_height, new_height):
            # create the current patch and check if it needs to be excluded
            curr_patch = image[y : y + new_height, x : x + new_width]
            if exclude_file is not None:
                curr_sky_corners = get_sky_corners(
                    reference_file,
                    pixel_coords=(x, x + new_width, y, y + new_height),
                    header=reference_header,
                )
                if check_sky_overlap(
                    exclude_sky_corners, curr_sky_corners, exclude_wcs
                ):
                    continue
            # get the stars and gals contained in the patch and adjust their coordinates based on the current patch
            curr_stars = stars[
                (stars[:, 0] >= x)
                & (stars[:, 0] < x + new_width)
                & (stars[:, 1] >= y)
                & (stars[:, 1] < y + new_height)
            ] - np.array([x, y])
            curr_gals = galaxies[
                (galaxies[:, 0] >= x)
                & (galaxies[:, 0] < x + new_width)
                & (galaxies[:, 1] >= y)
                & (galaxies[:, 1] < y + new_height)
            ] - np.array([x, y])
            all_patches.append((curr_patch, curr_stars, curr_gals))

    return all_patches


def display_patches(patches: np.ndarray, rows: int, cols: int):
    """Displays image patches in a grid.

    Args:
        patches (np.ndarray): 1D array of patches.
        rows (int): Rows to display.
        cols (int): Columns to display.
    """
    fig, ax = plt.subplots(rows, cols, figsize=(20, 10))
    for i in range(cols):
        for j in range(rows):
            p, s, g = patches[i * rows + j]

            ax[j, i].imshow(p[:, :, :3])
            ax[j, i].scatter(s[:, 0], s[:, 1], color="green", s=1, label="Stars")
            ax[j, i].scatter(g[:, 0], g[:, 1], color="blue", s=1, label="Galaxies")
            ax[j, i].axis("off")

    plt.show()


def create_seg_map_from_patch(
    patch: np.ndarray, stars: np.ndarray, galaxies: np.ndarray
) -> np.ndarray:
    seg_map = np.zeros_like(patch)[:, :, :3]
    seg_map[:, :, 0] = 1
    if len(stars) > 0:
        stars = stars.astype(np.int64)
        seg_map[stars[:, 1], stars[:, 0], 0] = 0
        seg_map[stars[:, 1], stars[:, 0], 1] = 1
    if len(galaxies) > 0:
        galaxies = galaxies.astype(np.int64)
        seg_map[galaxies[:, 1], galaxies[:, 0], 0] = 0
        seg_map[galaxies[:, 1], galaxies[:, 0], 2] = 1
    return seg_map


def process_patches(files: List[str], save_dir: str, exclude_file: str = None):
    i = 0
    for file_name in files:
        if not Path(f"../data/processed/{file_name}.mat").exists():
            print(f"File {file_name} has not been processed. Skipping file.")
            continue

        curr_data = scipy.io.loadmat(f"../data/processed/{file_name}.mat")
        img = curr_data["image"]
        stars = curr_data["stars"]
        gals = curr_data["galaxies"]
        reference_file = f"../data/{file_name}/frame-i-{file_name}.fits.bz2"

        image_patches = create_image_patches(
            img, stars, gals, (64, 64), reference_file, exclude_file=exclude_file
        )

        for patch in image_patches:
            seg_map = create_seg_map_from_patch(patch[0], patch[1], patch[2])
            scipy.io.savemat(
                f"{save_dir}/{i}.mat",
                {
                    "image": patch[0],
                    "stars": patch[1],
                    "galaxies": patch[2],
                    "seg_map": seg_map,
                },
            )
            i += 1


def create_dataset_split(
    train_files: List[str],
    validation_files: List[str],
    test_files: List[str] = ["008162-6-0080"],
) -> None:
    train_path = "../data/train"
    validation_path = "../data/validation"
    test_path = "../data/test"
    Path(train_path).mkdir(exist_ok=True, parents=True)
    Path(validation_path).mkdir(exist_ok=True, parents=True)
    Path(test_path).mkdir(exist_ok=True, parents=True)

    exclude_img = f"../data/{test_files[0]}/frame-i-{test_files[0]}.fits.bz2"

    process_patches(train_files, train_path, exclude_file=exclude_img)
    process_patches(validation_files, validation_path, exclude_file=exclude_img)
    process_patches(test_files, test_path)
