from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue

galaxy_file = "../data/frame-r-008162-6-0080.fits.bz2"
star_file = "../data/frame-g-008162-6-0080.fits.bz2"

galaxy_data, star_data = fits.open(galaxy_file)[0].data, fits.open(star_file)[0].data

def reduce_to_tensor(image_array):

    #set everything non-zero to 1
    galaxy_tensor_improved = np.where(image_array >= 1, 1, image_array)
    galaxy_tensor_improved = np.where(galaxy_tensor_improved < 1, 0, galaxy_tensor_improved)

    #iterate through rows
    visited = {}
    height, length = galaxy_tensor_improved.shape

    result = []

    for index, value in np.ndenumerate(galaxy_tensor_improved):
        if value == 1 and index not in visited:

            detected = []
            q = Queue()

            q.put(index)

            while not q.empty():
                r, c = q.get()
                if (r, c) in visited:
                    continue
                if galaxy_tensor_improved[r,c] == 1:
                    #West
                    if c-1 >= 0: q.put((r,c-1))
                    #North-West
                    if c-1 >= 0 and r-1 >= 0: q.put((r-1,c-1))
                    #North
                    if r-1 >= 0: q.put((r-1,c))
                    #North-East
                    if r-1 >= 0 and c+1 < length: q.put((r-1,c+1))
                    #East
                    if c+1 < length: q.put((r,c+1))
                    #South-East
                    if r+1 < height and c+1 < length: q.put((r+1,c+1))
                    #South
                    if r+1 < height: q.put((r+1,c))
                    #South-West
                    if r + 1 < height and c-1 >= 0: q.put((r+1,c-1))

                    detected.append(np.array([r, c]))
                    visited[(r, c)] = True

            detected = np.array(detected)
            coordinates = detected.mean(axis=0).astype(int)
            result.append(coordinates)

    return np.array(result)



