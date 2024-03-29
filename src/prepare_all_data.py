import os
import subprocess
from pathlib import Path

import scipy

from utils.data_alignment import align_channels_stars_galaxies
from utils.data_preprocessing import create_dataset_split
from utils.logger import Logger

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

logger = Logger("Data Preparation")

train_files = [
    "001043-5-0025",
    "002188-2-0037",
    "005405-6-0099",
    "001992-1-0087",
    "003184-2-0169",
    "004510-4-0030",
    "008115-2-0181",
    "002029-6-0222",
    "001035-6-0017",
]
train_dirs = [
    "1043/5/",
    "2188/2/",
    "5405/6/",
    "1992/1/",
    "3184/2/",
    "4510/4/",
    "8115/2/",
    "2029/6/",
    "1035/6/",
]

validation_files = ["003918-3-0213", "007264-3-0111"]
validation_dirs = ["3918/3/", "7264/3/"]

test_files = ["008162-6-0080"]
test_dirs = ["8162/6/"]

data_dir = "../data/"

logger.info(
    f"Downloading ({len(train_files)}) training files, ({len(validation_files)}) validation files, and ({len(test_files)}) testing files from data.sdss.org into `{data_dir}`"
)

# download all data
for file_name, img_dir in (
    list(zip(train_files, train_dirs))
    + list(zip(validation_files, validation_dirs))
    + list(zip(test_files, test_dirs))
):
    if Path().joinpath(data_dir, file_name).exists():
        logger.info(f"`{file_name}` skipped as it exists locally...")
        continue
    logger.info(f"downloading `{file_name}`...")
    subprocess.check_call(
        "scripts/downloader.sh %s %s %s %s"
        % (str(img_dir), str(file_name), str(file_name.rsplit("-", 1)[0]), data_dir),
        shell=True,
    )

logger.info("All files downloaded. Beginning image alignment.")

# process the data for alignment and star and galaxy coordinate extraction
for file_name in train_files + validation_files + test_files:
    Path(f"{data_dir}processed/").mkdir(exist_ok=True, parents=True)
    if Path(f"{data_dir}processed/{file_name}.mat").exists():
        logger.info(f"`{file_name}` skipped as it exists locally...")
        continue
    calib_obj_name = file_name.rsplit("-", 1)[0]
    file_paths = [
        f"{data_dir}{file_name}/frame-i-{file_name}.fits.bz2",
        f"{data_dir}{file_name}/frame-r-{file_name}.fits.bz2",
        f"{data_dir}{file_name}/frame-g-{file_name}.fits.bz2",
        f"{data_dir}{file_name}/frame-u-{file_name}.fits.bz2",
        f"{data_dir}{file_name}/frame-z-{file_name}.fits.bz2",
    ]
    star_path = f"{data_dir}{file_name}/calibObj-{calib_obj_name}-star.fits.gz"
    galaxy_path = f"{data_dir}{file_name}/calibObj-{calib_obj_name}-gal.fits.gz"

    logger.info(f"aligning channels of`{file_name}`...")
    image, stars, galaxies = align_channels_stars_galaxies(
        file_paths, star_path, galaxy_path
    )
    logger.info(f"saving `{file_name}`...")
    scipy.io.savemat(
        f"../data/processed/{file_name}.mat",
        {"image": image, "stars": stars, "galaxies": galaxies},
    )


logger.info("All files aligned. Creating dataset splits.")
# create actual train validation and test sets
create_dataset_split(train_files, validation_files, test_files)
