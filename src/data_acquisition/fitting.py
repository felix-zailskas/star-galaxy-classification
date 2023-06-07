from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

r_file = "../data/frame-r-008162-6-0080.fits.bz2"
g_file = "./data/frame-g-008162-6-0080.fits.bz2"
i_file = "./data/frame-i-008162-6-0080.fits.bz2"
u_file = "./data/frame-u-008162-6-0080.fits.bz2"
z_file = "./data/frame-z-008162-6-0080.fits.bz2"

files = [r_file, g_file, i_file, u_file, z_file]

for f in files:
    hdul = fits.open(f)

    # hdul.info()
    wcs = WCS(hdul[0].header)
    coord = SkyCoord("00h00m00s +00d00m00s", frame="galactic")
    pixels = wcs.pixel_to_world(hdul[0].header["CRPIX"][0], hdul[0].header["CRPIX"][1])
    print(pixels)
    print(wcs)
