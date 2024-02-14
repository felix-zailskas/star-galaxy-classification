#!/bin/bash

# set to download specific image
image_dir=$1
image_name=$2
star_galaxy_name=$3
data_dir="$4$image_name"

mkdir $data_dir

# Stars and Galaxies
wget https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-$star_galaxy_name-gal.fits.gz -P $data_dir
wget https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-$star_galaxy_name-star.fits.gz -P $data_dir

# Make a specification file ... you can fill this with files to download.

echo frame-r-$image_name.fits.bz2 >  spec-list.txt
echo frame-g-$image_name.fits.bz2 >> spec-list.txt
echo frame-i-$image_name.fits.bz2 >> spec-list.txt
echo frame-u-$image_name.fits.bz2 >> spec-list.txt
echo frame-z-$image_name.fits.bz2 >> spec-list.txt
echo frame-irg-$image_name.jpg    >> spec-list.txt

# All images for these stars
wget -i spec-list.txt -r --no-parent -nd -B "https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/$image_dir" -P $data_dir
# delete temporary file needed for the downloading interface
rm spec-list.txt
