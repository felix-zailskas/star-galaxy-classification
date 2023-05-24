# Stars and Galaxies
wget https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-008162-6-gal.fits.gz -P ../../data/
wget https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-008162-6-star.fits.gz -P ../../data/

# Make a specification file ... you can fill this with files to download.

echo frame-r-008162-6-0080.fits.bz2 >  spec-list.txt
echo frame-g-008162-6-0080.fits.bz2 >> spec-list.txt
echo frame-i-008162-6-0080.fits.bz2 >> spec-list.txt
echo frame-u-008162-6-0080.fits.bz2 >> spec-list.txt
echo frame-z-008162-6-0080.fits.bz2 >> spec-list.txt
echo frame-irg-008162-6-0080.jpg    >> spec-list.txt

# All images for these stars
wget -i spec-list.txt -r --no-parent -nd -B 'https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/6/' -P ../../data/
