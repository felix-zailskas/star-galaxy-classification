# Classification of Stars and Galaxies using a UNet Architechture


## Dataset
Information about the dataset can be found here: https://github.com/damslab/datasets/tree/master/stars

## Creating the Environment

To create the virtual environment in this project you must have `pipenv` installed on your machine. Then run the following commands:

```
# for development environment
pipenv install --dev
# for production environment
pipenv install
```
To work within the environment you can now run:

```
# to activate the virtual environment
pipenv shell
# to run a single command
pipenv run <COMMAND>
```

## Contribution Workflow

This repository uses `pre-commit` hooks to ensure a consistent and clean file organization. Each registered hook will be executed when committing to the repository. To ensure that the hooks will be executed they need to be installed using the following command:

```
pre-commit install
```

The following hooks are registered in this Project:

<ul>
<li>
<b>Black:</b> Black is a PEP 8 compliant opinionated formatter. Black reformats entire files in place.
</li>
<li>
<b>isort:</b> isort is a Python utility / library to sort imports alphabetically, and automatically separated into sections and by type
</li>
</ul>


## Running the Application

## Data Processing
To easily download and process all needed data run the `prepare_all_data.py` script. This script downloads all training, validation and test images listed in the corresponding variables in the script. Fill the `<mode>_files` list with the names of the frames to use and the `<mode>_dirs` with the corresponding directory under https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/.
<br><br>
The script will download all needed channel files and the corresponding calibration objects storing the star and galaxy coordinates. It will then proceed to align the channels of each image and determine the star and galaxy coordinates for them. The result of this step is then stored under `/data/processed/`. After that the script splits the images into 64x64 pixel patches containing the respective stars and galaxies. The patches will be stored under `/data/train/`, `/data/validation/`, or `/data/test/` depending on the list they were stored in in the script. The patches are named after their index.

## Train
To train simply run the `train.py` script. If you are running on an MPS-enabled device, execute the following command in your terminal before training:
```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
After training, the script will generate a model file named `model.pth`.

## Results

You easily can check the results on the `predict_demo.ipynb` notebook. The notebook includes code that loads your trained model and displays the original image, the ground truth mask, and the predicted mask.

Here is an example of the results:

![Alt text](https://i.ibb.co/QJ1YhtY/488.png "Results")
